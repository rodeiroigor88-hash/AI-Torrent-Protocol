import asyncio
import os
import random
import re
import sys
import argparse
import logging
import hmac
import ipaddress
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from aiohttp import web
import aiohttp

from src.tensor_utils import MAX_COMPRESSED_BYTES, serialize_tensor, deserialize_tensor
from src import routing
from src.config import load_config, tracker_url as configured_tracker_url
from src.pow_utils import DEFAULT_EPSILON, outputs_match, relative_l2
from src.tls_utils import (
    build_client_ssl_context,
    build_server_ssl_context,
    cert_node_id,
    load_cert,
    node_sni,
    peer_node_id,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_CLIENT_END_LAYER = 7
MAX_ROUTE_ATTEMPTS = 3
# Reconexion con espera exponencial: si el tracker se cae, el cliente no debe
# rendirse con un "Enjambre vacio" al primer intento.
RECONNECT_ATTEMPTS = 5
RECONNECT_BASE_DELAY = 1.0
MAX_RECONNECT_DELAY = 15.0


class PipelineError(RuntimeError):
    """Un nodo intermedio notifico que no pudo continuar el pipeline."""

    def __init__(self, message, node_id="", hop=-1):
        super().__init__(message)
        self.node_id = node_id
        self.hop = hop


class ClientNodeModel(torch.nn.Module):
    def __init__(self, model_name: str, end_layer: int):
        super().__init__()
        logger.info(f"Cargando {model_name} (Embeddings y primeras {end_layer} capas)...")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            self.embed_tokens = base_model.model.embed_tokens
            layers = base_model.model.layers
            self.rotary_emb = base_model.model.rotary_emb
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            self.embed_tokens = base_model.transformer.wte
            self.wpe = base_model.transformer.wpe
            self.drop = base_model.transformer.drop
            layers = base_model.transformer.h
            self.rotary_emb = None
        else:
            raise ValueError("Arquitectura no soportada")

        if not 0 <= end_layer < len(layers):
            raise ValueError(
                f"end_layer={end_layer} fuera de rango; el modelo tiene {len(layers)} capas"
            )

        self.blocks = torch.nn.ModuleList([layers[i] for i in range(0, end_layer + 1)])
        self.arch = 'llama/qwen' if hasattr(base_model, 'model') else 'gpt2'
        self.end_layer = end_layer
        self.hidden_size = getattr(base_model.config, 'hidden_size', None) or \
            getattr(base_model.config, 'n_embd', None)
        del base_model

    def forward(self, input_ids):
        with torch.no_grad():
            if self.arch == 'gpt2':
                inputs_embeds = self.embed_tokens(input_ids)
                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
                position_embeds = self.wpe(position_ids)
                hidden_states = self.drop(inputs_embeds + position_embeds)
                position_embeddings = None
                attention_mask = None
            else:
                hidden_states = self.embed_tokens(input_ids)
                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                sequence_length = hidden_states.shape[1]
                attention_mask = torch.full(
                    (sequence_length, sequence_length),
                    torch.finfo(hidden_states.dtype).min,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                attention_mask = torch.triu(attention_mask, diagonal=1)[None, None, :, :]

            for block in self.blocks:
                if self.arch == 'llama/qwen':
                    output = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                    )
                else:
                    output = block(hidden_states)
                hidden_states = output[0] if isinstance(output, tuple) else output

            return hidden_states, position_ids

class AgenticChat:
    def __init__(self, model_name: str, client_port: int, next_node_url: str = None,
                 auth_token: str = None, callback_host: str = None,
                 tls_cert: str = None, tls_key: str = None, tls_ca: str = None,
                 require_client_cert: bool = True, callback_url: str = None,
                 route: list = None, end_layer: int = DEFAULT_CLIENT_END_LAYER,
                 audit_probability: float = 0.0, audit_epsilon: float = DEFAULT_EPSILON,
                 tracker_url: str = None):
        self.model_name = model_name
        self.client_port = client_port
        self.primary_node_url = next_node_url
        # El pipeline estático configurado a mano se conserva: un timeout puntual
        # no debe dejar al cliente sin destino para el resto de la sesión.
        self._static_node_url = next_node_url
        # Secreto compartido del enjambre (ver P2PNode.auth_token). Se envía a los
        # workers al reenviar tensores y se exige a quien invoque nuestro /callback.
        self.auth_token = auth_token
        # Configurable desde el panel de Ajustes: probar contra un tracker local
        # no debe exigir recompilar.
        self.tracker_url = tracker_url or configured_tracker_url()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_arch = "llama/qwen" if "Qwen" in model_name or "Llama" in model_name else "gpt2"
        self.client_model = None
        self.end_layer = end_layer
        self.pending_requests = {}
        self.generation_lock = asyncio.Lock()
        self.messages = [
            {"role": "system", "content": "You are a helpful coding assistant. If you write code, start the code block with a comment indicating the filename, e.g. `# script.py`"}
        ]
        self.session = None
        self.runner = None
        self.setup_error = None
        # [Ejecución autónoma] Bloques de código detectados en la última respuesta,
        # pendientes de confirmación del usuario antes de escribirse a disco.
        self.pending_files = None

        # --- TLS -------------------------------------------------------------
        if bool(tls_cert) != bool(tls_key):
            raise ValueError("--tls-cert y --tls-key deben usarse juntos")
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.tls_ca = tls_ca
        self.require_client_cert = require_client_cert
        self.node_id = cert_node_id(load_cert(tls_cert)) if tls_cert else str(uuid.uuid4())
        self._server_ssl_context = None
        self._client_ssl_context = None
        if tls_cert and tls_key:
            self._server_ssl_context = build_server_ssl_context(
                tls_cert, tls_key, tls_ca, require_client_cert=require_client_cert
            )
            if tls_ca:
                self._client_ssl_context = build_client_ssl_context(tls_ca, tls_cert, tls_key)
        self.scheme = "https" if self._server_ssl_context else "http"
        self.callback_host = callback_host or ('0.0.0.0' if (auth_token or tls_cert) else '127.0.0.1')
        advertised = '127.0.0.1' if self.callback_host in ('0.0.0.0', '::') else self.callback_host
        self.callback_url = callback_url or f"{self.scheme}://{advertised}:{client_port}/callback"

        # --- Enrutamiento ----------------------------------------------------
        # `route` son los saltos del pipeline SIN el callback; el callback se
        # anade siempre al final. Si solo hay `next_node_url` usamos el sobre v1
        # y el encadenamiento estatico de los `--next-node` de cada worker.
        self.route = list(route) if route else None
        self.route_exp = None
        self.route_sig = None
        self.failed_nodes = set()

        # --- Proof of Compute ------------------------------------------------
        self.audit_probability = max(0.0, min(1.0, audit_probability))
        self.audit_epsilon = audit_epsilon
        self.audit_stats = {"checked": 0, "mismatches": 0}
        self._audit_warned = False

    # ------------------------------------------------------------------ red

    def _auth_headers(self) -> dict:
        headers = {'Content-Type': 'application/msgpack'}
        if self.auth_token:
            headers['X-Auth-Token'] = self.auth_token
        return headers

    def _request_kwargs(self, url: str, node_id: str) -> dict:
        kwargs = {}
        if node_id and url.startswith('https://'):
            kwargs['server_hostname'] = node_sni(node_id)
        return kwargs

    async def _handle_callback(self, request):
        if not self._callback_authorized(request):
            return web.Response(status=401, text="Unauthorized")
        try:
            data = await request.read()
            envelope = await asyncio.to_thread(deserialize_tensor, data)
            if not isinstance(envelope, dict):
                raise ValueError("Missing request envelope")
            request_id = envelope.get('request_id')
            uuid.UUID(request_id)

            if 'error' in envelope:
                # El nodo intermedio nos avisa de que rompio el pipeline: fallamos
                # rapido en vez de esperar los 30 s del timeout.
                info = envelope['error'] if isinstance(envelope['error'], dict) else {}
                future = self.pending_requests.get(request_id)
                if future and not future.done():
                    future.set_exception(PipelineError(
                        info.get('message', 'fallo en el pipeline'),
                        node_id=info.get('node_id', ''),
                        hop=info.get('hop', -1),
                    ))
                return web.Response(status=200, text="OK")

            logits = envelope.get('payload')
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise ValueError("Invalid logits tensor")
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid callback payload")

        future = self.pending_requests.get(request_id)
        if future and not future.done():
            future.set_result(logits)
        return web.Response(status=200, text="OK")

    def _callback_authorized(self, request) -> bool:
        if peer_node_id(request):
            return True
        supplied_token = request.headers.get('X-Auth-Token', '')
        if self.auth_token:
            return hmac.compare_digest(supplied_token, self.auth_token)
        try:
            return ipaddress.ip_address(request.remote).is_loopback
        except (TypeError, ValueError):
            return False

    async def _send_to_node(self, session, url, payload, node_id=""):
        if not url:
            return False
        try:
            async with session.post(url, data=payload, headers=self._auth_headers(),
                                    **self._request_kwargs(url, node_id)) as resp:
                await resp.read()
                return resp.status in (200, 202)
        except Exception as exc:
            logger.warning(f"[Cliente] No se pudo contactar con {url}: {exc}")
            return False

    async def setup(self):
        self.client_model = ClientNodeModel(self.model_name, end_layer=self.end_layer)
        app = web.Application(client_max_size=MAX_COMPRESSED_BYTES)
        app.router.add_post('/callback', self._handle_callback)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.callback_host, self.client_port,
                           ssl_context=self._server_ssl_context)
        await site.start()
        connector = aiohttp.TCPConnector(ssl=self._client_ssl_context) if self._client_ssl_context \
            else aiohttp.TCPConnector()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), connector=connector
        )
        logger.info(f"Cliente preparado en {self.scheme}://{self.callback_host}:{self.client_port} "
                    f"(callback publicado: {self.callback_url})")

    async def teardown(self):
        if self.session:
            await self.session.close()
        if self.runner:
            await self.runner.cleanup()

    # -------------------------------------------------------------- rutas

    async def _plan_route(self, exclude=None):
        """Pide al tracker el camino completo del pipeline, no solo un nodo.

        El tracker devuelve la ruta firmada (Ed25519) para que cada nodo pueda
        verificar que no se la invento un cliente malicioso.
        """
        exclude = set(exclude or ()) | self.failed_nodes
        # El callback va DENTRO de lo que firma el tracker: si lo añadiéramos
        # después, el último salto quedaría sin firmar y el enjambre podría
        # usarse para bombardear cualquier destino.
        params = {"model_arch": self.model_arch, "start_layer": str(self.end_layer + 1),
                  "callback": self.callback_url}
        if self.tls_cert and self.node_id:
            params["callback_node_id"] = self.node_id
        if exclude:
            params["exclude"] = ",".join(sorted(x for x in exclude if x))
        try:
            async with self.session.get(f"{self.tracker_url}/plan", params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"[Cliente] El tracker no pudo planificar la ruta ({resp.status}).")
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.error(f"[Cliente] Tracker inaccesible: {exc}")
            return None

        hops = data.get("route") or []
        if not hops:
            return None
        signature = data.get("signature")
        if isinstance(signature, str):
            try:
                signature = bytes.fromhex(signature)
            except ValueError:
                logger.warning("[Cliente] El tracker devolvió una firma de ruta ilegible.")
                signature = None
        self.route = hops
        self.route_exp = data.get("exp")
        self.route_sig = signature
        return self.route

    async def _legacy_route(self):
        """Compatibilidad: el tracker antiguo solo devolvia una lista de nodos."""
        try:
            async with self.session.get(f"{self.tracker_url}/route",
                                        params={"model_arch": self.model_arch}) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.error(f"[Cliente] Tracker inaccesible: {exc}")
            return None
        nodes = data.get("nodes", [])
        if not nodes:
            return None
        node = nodes[0]
        scheme = node.get("scheme", "http")
        self.primary_node_url = f"{scheme}://{node.get('ip')}:{node.get('port')}/forward"
        return self.primary_node_url

    def _route_ends_at_our_callback(self, route) -> bool:
        try:
            return routing.hop_url(route[-1]) == routing.normalize_endpoint(
                self.callback_url, "/callback")
        except (IndexError, routing.RouteError):
            return False

    def _full_route(self):
        """Ruta completa, terminada siempre en nuestro `/callback`.

        La del tracker ya viene cerrada y firmada con el callback dentro; la de
        `--route` la escribe el usuario con solo los workers, así que hay que
        cerrarla aquí (esa nunca va firmada, y por eso solo vale en loopback o
        con `--allow-unsigned-routes`).
        """
        if not self.route:
            return None
        if self._route_ends_at_our_callback(self.route):
            return list(self.route)
        hop_entry = {"url": self.callback_url, "node_id": self.node_id} if self.tls_cert \
            else self.callback_url
        return list(self.route) + [hop_entry]

    async def _find_target_once(self) -> bool:
        if await self._plan_route():
            return True
        if await self._legacy_route():
            return True
        if self._static_node_url:
            self.primary_node_url = self._static_node_url
            return True
        return False

    async def _ensure_target(self, token_callback=None, attempts: int = 1) -> bool:
        """Busca un destino, reintentando con espera exponencial.

        `attempts=1` para la replanificación dentro de la generación (donde el
        bucle exterior ya reintenta) y `RECONNECT_ATTEMPTS` al empezar un turno,
        que es cuando merece la pena esperar a que el tracker vuelva.
        """
        if self.route or self.primary_node_url:
            return True
        if token_callback:
            token_callback("[Buscando Nodos Expertos...]\n")

        for attempt in range(max(1, attempts)):
            if await self._find_target_once():
                if attempt and token_callback:
                    token_callback("[Reconectado con la red P2P]\n")
                return True
            if attempt + 1 >= attempts:
                break
            delay = min(RECONNECT_BASE_DELAY * (2 ** attempt), MAX_RECONNECT_DELAY)
            if token_callback:
                token_callback(f"[Reconectando con la red P2P... intento "
                               f"{attempt + 2}/{attempts}, {delay:.0f}s]\n")
            await asyncio.sleep(delay)

        if token_callback:
            token_callback("[Error] No hay nodos disponibles en el enjambre. "
                           "Comprueba tu conexión o la URL del tracker en Ajustes.")
        return False

    def _build_payload(self, request_id, hidden_states, position_ids):
        payload_data = {'hidden_states': hidden_states, 'position_ids': position_ids} \
            if self.client_model.arch == 'llama/qwen' else hidden_states
        full_route = self._full_route()
        if full_route:
            envelope = routing.build_envelope(
                request_id, payload_data, route=full_route, hop=0,
                expires_at=self.route_exp, signature=self.route_sig,
            )
            first = full_route[0]
            return envelope, routing.hop_url(first), routing.hop_node_id(first)
        return ({'request_id': request_id, 'payload': payload_data},
                self.primary_node_url, "")

    async def _run_pipeline_step(self, hidden_states, position_ids, timeout=30.0):
        """Un paso completo del pipeline con reintento y replanificacion."""
        last_error = None
        for attempt in range(MAX_ROUTE_ATTEMPTS):
            if not await self._ensure_target():
                raise PipelineError("No hay nodos disponibles en el enjambre")

            request_id = str(uuid.uuid4())
            envelope, target_url, target_node_id = self._build_payload(
                request_id, hidden_states, position_ids
            )
            payload = await asyncio.to_thread(serialize_tensor, envelope)

            future = asyncio.get_running_loop().create_future()
            self.pending_requests[request_id] = future
            try:
                accepted = await self._send_to_node(self.session, target_url, payload, target_node_id)
                if not accepted:
                    last_error = PipelineError(f"El nodo {target_url} rechazó la petición")
                    self._invalidate_route(target_node_id)
                    continue
                return await asyncio.wait_for(future, timeout=timeout)
            except PipelineError as exc:
                last_error = exc
                logger.warning(f"[Cliente] Pipeline roto en el salto {exc.hop} "
                               f"(nodo {exc.node_id or 'desconocido'}): {exc}")
                self._invalidate_route(exc.node_id)
            except asyncio.TimeoutError:
                last_error = PipelineError("Timeout de red")
                self._invalidate_route(target_node_id)
            finally:
                self.pending_requests.pop(request_id, None)

        raise last_error or PipelineError("El pipeline falló sin diagnóstico")

    def _invalidate_route(self, node_id=""):
        if node_id:
            self.failed_nodes.add(node_id)
        self.route = None
        self.route_exp = None
        self.route_sig = None
        self.primary_node_url = None

    # ------------------------------------------------- proof of compute

    async def _audit_step(self, hidden_states, position_ids, reference_logits):
        """Repite el paso por una ruta alternativa y compara con tolerancia.

        La comparacion es de L2 relativa, no de igualdad: el downcast a float16
        del `secret_sauce` y la variacion de hardware hacen imposible el bit a bit.
        """
        previous_route = self.route
        previous_exp, previous_sig = self.route_exp, self.route_sig
        # El último salto es nuestro propio callback: no es un nodo auditable.
        worker_hops = list(previous_route or [])
        if worker_hops and self._route_ends_at_our_callback(worker_hops):
            worker_hops = worker_hops[:-1]
        excluded = {routing.hop_node_id(hop) for hop in worker_hops}
        excluded.discard("")
        if not excluded:
            if not self._audit_warned:
                logger.info("[Auditoría] Sin identidades de nodo en la ruta: auditoría desactivada.")
                self._audit_warned = True
            return None

        self.route = None
        alternative = await self._plan_route(exclude=excluded)
        try:
            if not alternative:
                if not self._audit_warned:
                    logger.info("[Auditoría] No hay ruta alternativa disponible para auditar.")
                    self._audit_warned = True
                return None
            audited_nodes = [routing.hop_node_id(hop) for hop in worker_hops]
            candidate = await self._run_pipeline_step(hidden_states, position_ids, timeout=30.0)
        except PipelineError as exc:
            logger.warning(f"[Auditoría] No se pudo completar la comprobación: {exc}")
            return None
        finally:
            self.route, self.route_exp, self.route_sig = previous_route, previous_exp, previous_sig

        self.audit_stats["checked"] += 1
        if outputs_match(reference_logits, candidate, self.audit_epsilon):
            return True

        divergence = relative_l2(reference_logits, candidate)
        self.audit_stats["mismatches"] += 1
        logger.error(f"[Auditoría] DIVERGENCIA detectada (L2 relativa {divergence:.4f} > "
                     f"{self.audit_epsilon}). Nodos implicados: {audited_nodes}")
        await self._report_audit_failure(audited_nodes, divergence)
        for node_id in audited_nodes:
            if node_id:
                self.failed_nodes.add(node_id)
        self._invalidate_route()
        return False

    async def _report_audit_failure(self, node_ids, divergence):
        try:
            async with self.session.post(
                f"{self.tracker_url}/report",
                json={"nodes": [n for n in node_ids if n], "divergence": divergence,
                      "model_arch": self.model_arch, "reporter": self.node_id},
            ) as resp:
                await resp.read()
        except Exception as exc:
            logger.warning(f"[Auditoría] No se pudo reportar al tracker: {exc}")

    # ----------------------------------------------------------- agente

    _FILENAME_HEADER_RE = re.compile(r'^\s*(?:#|//)\s*([\w.\-]+\.\w+)\s*$')

    def _extract_code_files(self, text: str) -> dict:
        """Busca bloques de código con un comentario de nombre de archivo en la
        primera línea (formato pedido por el system prompt: `# script.py`) y
        devuelve {filename: contenido}."""
        files = {}
        for block in re.findall(r'```[a-zA-Z0-9]*\n(.*?)```', text, re.DOTALL):
            lines = block.splitlines()
            if not lines:
                continue
            match = self._FILENAME_HEADER_RE.match(lines[0])
            if not match:
                continue
            filename = os.path.basename(match.group(1))  # evita path traversal
            content = "\n".join(lines[1:])
            files[filename] = content
        return files

    def _write_pending_files(self) -> list:
        """Escribe a disco los archivos detectados y pendientes de confirmación."""
        written = []
        for filename, content in (self.pending_files or {}).items():
            try:
                with open(filename, "x", encoding="utf-8") as f:
                    f.write(content)
                written.append(filename)
                logger.info(f"[Agente] Archivo creado: {filename}")
            except OSError as e:
                logger.error(f"[Agente] No se pudo escribir {filename}: {e}")
        return written

    async def generate_response(self, user_input: str, token_callback=None) -> str:
        async with self.generation_lock:
            return await self._generate_response(user_input, token_callback)

    async def _generate_response(self, user_input: str, token_callback=None) -> str:
        # [Ejecución autónoma] Si la última respuesta detectó archivos y está
        # esperando confirmación, esta entrada del usuario responde a esa
        # pregunta (y/n) en vez de ser un nuevo turno de conversación con el LLM.
        if self.pending_files is not None:
            answer = user_input.strip().lower()
            if answer in ("y", "yes", "s", "si", "sí"):
                written = self._write_pending_files()
                msg = f"\n[Agente] Archivo(s) creado(s): {', '.join(written)}\n" if written else \
                      "\n[Agente] No se pudo crear ningún archivo.\n"
            else:
                msg = "\n[Agente] Cancelado. No se escribió ningún archivo.\n"
            self.pending_files = None
            if token_callback: token_callback(msg)
            return msg

        self.messages.append({"role": "user", "content": user_input})

        if not await self._ensure_target(token_callback, attempts=RECONNECT_ATTEMPTS):
            return "[Error] No hay nodos disponibles en el enjambre."

        text_input = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(text_input, return_tensors="pt").input_ids

        generated_text = ""
        max_new_tokens = 256
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for _ in range(max_new_tokens):
            hidden_states, position_ids = self.client_model(input_ids)

            try:
                logits = await self._run_pipeline_step(hidden_states, position_ids)
            except PipelineError as exc:
                msg = f"\n[Error] {exc}"
                if token_callback: token_callback(msg)
                break

            if self.audit_probability and random.random() < self.audit_probability:
                await self._audit_step(hidden_states, position_ids, logits)

            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            token_str = self.tokenizer.decode(next_token_id[0])
            if token_callback: token_callback(token_str)
            generated_text += token_str

            if next_token_id.item() == self.tokenizer.eos_token_id or next_token_id.item() == im_end_id:
                break

        self.messages.append({"role": "assistant", "content": generated_text})

        # [Ejecución autónoma] Si el modelo generó bloques de código con nombre
        # de archivo (según el system prompt), preguntamos antes de escribir a disco.
        detected_files = self._extract_code_files(generated_text)
        if detected_files:
            self.pending_files = detected_files
            prompt_msg = f"\n\n[Agente] ¿Guardar en disco {', '.join(detected_files)}? (y/n): "
            if token_callback: token_callback(prompt_msg)
            generated_text += prompt_msg

        return generated_text

async def cli_main(chat):
    await chat.setup()
    print("\n" + "="*50)
    print(f"AGENTE P2P INICIADO ({chat.model_name})")
    print("="*50 + "\n")

    try:
        while True:
            user_input = input("\nTú: ")
            if user_input.lower() == 'salir': break
            print("Agente: ", end="", flush=True)
            await chat.generate_response(user_input, token_callback=lambda t: print(t, end="", flush=True))
            print()
    finally:
        if chat.audit_stats["checked"]:
            print(f"\n[Auditoría] {chat.audit_stats['checked']} comprobaciones, "
                  f"{chat.audit_stats['mismatches']} divergencias.")
        await chat.teardown()

if __name__ == "__main__":
    settings = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=settings["client_port"])
    parser.add_argument("--tracker-url", type=str, default=None,
                        help=f"URL del tracker (config: {settings['tracker_url']})")
    parser.add_argument("--next-node", type=str, default=None,
                        help="Primer worker del pipeline estático (sobre v1)")
    parser.add_argument("--route", type=str, default=None,
                        help="Pipeline completo separado por comas (sobre v2). "
                             "Cada salto puede ser URL o URL#node_id")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--end-layer", type=int, default=DEFAULT_CLIENT_END_LAYER,
                        help="Última capa que ejecuta el cliente; los workers empiezan en la siguiente")
    parser.add_argument("--auth-token", type=str, default=os.environ.get("TOKENTORRENT_AUTH_TOKEN"),
                         help="Secreto compartido del enjambre (debe coincidir con el de los workers)")
    parser.add_argument("--callback-url", type=str, default=None,
                        help="URL publica de nuestro /callback (necesaria si estamos tras NAT)")
    parser.add_argument("--audit-probability", type=float, default=0.0,
                        help="Probabilidad [0-1] de auditar un paso por una ruta alternativa")
    parser.add_argument("--audit-epsilon", type=float, default=DEFAULT_EPSILON,
                        help="Tolerancia L2 relativa al comparar dos ejecuciones")
    parser.add_argument("--tls-cert", type=str, default=os.environ.get("TOKENTORRENT_TLS_CERT"))
    parser.add_argument("--tls-key", type=str, default=os.environ.get("TOKENTORRENT_TLS_KEY"))
    parser.add_argument("--tls-ca", type=str, default=os.environ.get("TOKENTORRENT_TLS_CA"))
    parser.add_argument("--no-client-cert", action="store_true",
                        help="No exigir certificado a quien invoque nuestro /callback")
    args = parser.parse_args()

    parsed_route = None
    if args.route:
        parsed_route = []
        for raw_hop in args.route.split(","):
            raw_hop = raw_hop.strip()
            if not raw_hop:
                continue
            url, _, node_id = raw_hop.partition("#")
            parsed_route.append({"url": url, "node_id": node_id} if node_id else url)

    chat = AgenticChat(
        model_name=args.model, client_port=args.port, next_node_url=args.next_node,
        auth_token=args.auth_token, callback_url=args.callback_url, route=parsed_route,
        end_layer=args.end_layer, audit_probability=args.audit_probability,
        audit_epsilon=args.audit_epsilon, tls_cert=args.tls_cert, tls_key=args.tls_key,
        tls_ca=args.tls_ca, require_client_cert=not args.no_client_cert,
        tracker_url=args.tracker_url,
    )
    asyncio.run(cli_main(chat))
