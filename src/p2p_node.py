import asyncio
import logging
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
import torch
import uuid
import os
import psutil
import hashlib
import hmac
import ipaddress
import time
from collections import OrderedDict
from urllib.parse import urlsplit

import socket

# Importamos las utilidades que creamos
from .tensor_utils import MAX_COMPRESSED_BYTES, serialize_tensor, deserialize_tensor
from . import routing
from .config import DEFAULTS, tracker_token as configured_tracker_token, tracker_url as configured_tracker_url
from .ratelimit import ATTEST_BURST, ATTEST_RATE, RateLimiter
from .routing import RouteError
from .pow_utils import (
    DEFAULT_ATTEST_SEQ_LEN,
    attestation_message,
    canonical_challenge,
    deterministic_inference,
    tensor_digest,
)
from .tls_utils import (
    build_client_ssl_context,
    build_server_ssl_context,
    cert_node_id,
    load_cert,
    load_private_key,
    node_sni,
    peer_node_id,
    sign_blob,
)

logger = logging.getLogger(__name__)

# 15 s de latido para que el tracker pueda expulsar a un nodo fantasma a los
# 45 s (tres latidos perdidos) sin desconectar a nodos sanos por un fallo suelto.
HEARTBEAT_INTERVAL = 15
MAX_HEARTBEAT_BACKOFF = 300
# Cuantos puertos consecutivos se prueban si el configurado esta ocupado.
PORT_FALLBACK_ATTEMPTS = 10


def heartbeat_delay(consecutive_failures: int) -> float:
    """Espera antes del siguiente latido: exponencial y con techo.

    Con 0 fallos late al ritmo normal; a partir de ahi duplica para no castigar
    a un tracker que ya esta en apuros, pero sin dejar de intentarlo nunca.
    """
    if consecutive_failures <= 0:
        return float(HEARTBEAT_INTERVAL)
    return float(min(HEARTBEAT_INTERVAL * (2 ** consecutive_failures), MAX_HEARTBEAT_BACKOFF))


def find_free_port(host: str, port: int, attempts: int = PORT_FALLBACK_ATTEMPTS) -> int:
    """Primer puerto libre a partir de `port`. Devuelve `port` si no hay ninguno.

    Sin esto, un segundo proceso del nodo (o cualquier app que ya ocupe el 8001)
    hacia que el worker muriera en el arranque con un OSError.
    """
    if port == 0:
        return 0
    for offset in range(max(1, attempts)):
        candidate = port + offset
        if candidate > 65535:
            break
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Sin SO_REUSEADDR a proposito: en Windows esa opcion permite atar un
            # puerto YA ocupado, con lo que la sonda diria "libre" siempre.
            probe.bind((host if host not in ('::',) else '', candidate))
            return candidate
        except OSError:
            continue
        finally:
            probe.close()
    return port


def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()

class P2PNode:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, next_node_url: str = None,
                 operation=None, layers: str = "8-15", auth_token: str = None,
                 model_name: str = None, model_arch: str = None, enable_upnp: bool = False,
                 enable_tracker: bool = False, max_concurrency: int = 1, cache_size: int = 0,
                 tls_cert: str = None, tls_key: str = None, tls_ca: str = None,
                 require_client_cert: bool = True, tracker_key: str = None,
                 allow_unsigned_routes: bool = False, hidden_size: int = None,
                 max_ram_percent: int = None, cpu_cores: int = None,
                 cpu_pause_threshold: int = None, rate_limiter: RateLimiter = None,
                 port_fallback: bool = True, is_last: bool = False):
        self.host = host
        self.port = port
        self.next_node_url = next_node_url
        self.layers = layers
        self.auth_token = auth_token
        self.model_name = model_name
        self.enable_upnp = enable_upnp
        self.enable_tracker = enable_tracker
        self.model_arch = model_arch or "unknown"
        self.tracker_url = configured_tracker_url()
        self.is_paused = False
        self.pause_reason = ""
        self.hidden_size = hidden_size
        self.port_fallback = port_fallback
        # El tracker necesita saber quien cierra el pipeline (norm final + lm_head)
        # para no planificar una ruta que nunca produce logits.
        self.is_last = is_last

        # --- Limites de donacion de recursos (QoS) ---------------------------
        self.max_ram_percent = max_ram_percent if max_ram_percent is not None \
            else DEFAULTS["max_ram_percent"]
        self.cpu_pause_threshold = cpu_pause_threshold if cpu_pause_threshold is not None \
            else DEFAULTS["cpu_pause_threshold"]
        self.cpu_cores = cpu_cores if cpu_cores is not None else DEFAULTS["cpu_cores"]

        # --- Limitador de peticiones -----------------------------------------
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self.attest_limiter = RateLimiter(rate=ATTEST_RATE, burst=ATTEST_BURST)

        # --- Identidad y TLS -------------------------------------------------
        # La identidad del nodo es su certificado. Sin TLS caemos a un UUID
        # efimero, que NO sirve para reputacion: un tramposo lo rota reiniciando.
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.tls_ca = tls_ca
        self.require_client_cert = require_client_cert
        self._signing_key = None
        self._cert_pem = b""
        self._server_ssl_context = None
        self._client_ssl_context = None

        if tls_cert and tls_key:
            certificate = load_cert(tls_cert)
            self._signing_key = load_private_key(tls_key)
            with open(tls_cert, "rb") as handle:
                self._cert_pem = handle.read()
            self.node_id = cert_node_id(certificate) or str(uuid.uuid4())
            self._server_ssl_context = build_server_ssl_context(
                tls_cert, tls_key, tls_ca, require_client_cert=require_client_cert
            )
            if tls_ca:
                self._client_ssl_context = build_client_ssl_context(tls_ca, tls_cert, tls_key)
            else:
                logger.warning(
                    "[Nodo %s] TLS activo sin --tls-ca: las conexiones SALIENTES hacia otros "
                    "nodos del enjambre usaran las CA del sistema y fallaran la verificacion.",
                    port,
                )
        else:
            self.node_id = str(uuid.uuid4())
        self.public_key = self._cert_pem.decode("ascii") if self._cert_pem else None

        # --- Enrutamiento ----------------------------------------------------
        self.allow_unsigned_routes = allow_unsigned_routes
        self._tracker_key = routing.load_tracker_public_key(tracker_key) if tracker_key else None

        # [FASE 6.1] KV Cache Distribuido (Memoria Eidetica)
        self.kv_cache = OrderedDict()
        self.max_cache_size = max(0, cache_size)
        self._work_semaphore = asyncio.Semaphore(max_concurrency)
        self._background_tasks = set()
        self._client_session = None

        self.app = web.Application(client_max_size=MAX_COMPRESSED_BYTES)
        self.app.on_cleanup.append(self._cleanup)
        self.setup_routes()

        self.callback_handler = None
        self.operation = operation if operation else lambda x: x * 2

        self._set_low_priority()
        self._apply_resource_limits()

    @property
    def tls_enabled(self) -> bool:
        return self._server_ssl_context is not None

    @property
    def scheme(self) -> str:
        return "https" if self.tls_enabled else "http"

    def setup_routes(self):
        self.app.router.add_get('/ping', self.handle_ping)
        self.app.router.add_post('/forward', self.handle_forward)
        self.app.router.add_post('/callback', self.handle_callback)
        self.app.router.add_post('/attest', self.handle_attest)

    async def handle_ping(self, request):
        throttled = self._throttle(request)
        if throttled is not None:
            return throttled
        return web.json_response({
            "status": "alive",
            "port": self.port,
            "node_id": self.node_id,
            "layers": self.layers,
            "model_arch": self.model_arch,
            "tls": self.tls_enabled,
            "paused": self.is_paused,
            "pause_reason": self.pause_reason,
        })

    def _client_identity(self, request) -> str:
        """Identidad para el rate limit: el certificado si lo hay, si no la IP."""
        return peer_node_id(request) or (request.remote or "desconocido")

    def _throttle(self, request, limiter: RateLimiter = None):
        """Devuelve una respuesta 429 si el cliente excede su ritmo, o None."""
        limiter = limiter or self.rate_limiter
        allowed, retry_after = limiter.allow(self._client_identity(request))
        if allowed:
            return None
        logger.warning(f"[Nodo {self.port}] Rate limit para {self._client_identity(request)}; "
                       f"reintento en {retry_after}s")
        return web.Response(status=429, text="Too Many Requests",
                            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))})

    def _check_auth(self, request) -> bool:
        """Autenticacion en dos niveles.

        Un certificado de cliente valido ya fue verificado por OpenSSL contra la
        CA del enjambre durante el handshake, asi que su sola presencia basta.
        El token compartido se mantiene como compatibilidad hacia atras.
        """
        if peer_node_id(request):
            return True
        supplied_token = request.headers.get('X-Auth-Token', '')
        if self.auth_token:
            return hmac.compare_digest(supplied_token, self.auth_token)
        try:
            return ipaddress.ip_address(request.remote).is_loopback
        except (TypeError, ValueError):
            return False

    def _start_background_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    # ------------------------------------------------------------------ sobres

    def _parse_envelope(self, incoming) -> dict:
        """Normaliza los sobres v1 y v2 a una estructura interna comun."""
        if isinstance(incoming, dict) and incoming.get('v') == routing.ENVELOPE_VERSION:
            request_id = incoming.get('request_id')
            self._validate_request_id(request_id)
            if 'payload' not in incoming:
                raise ValueError("El sobre v2 no contiene payload")

            route = incoming.get('route')
            if route is None:
                return {'request_id': request_id, 'payload': incoming['payload'],
                        'route': None, 'hop': 0, 'route_exp': None, 'route_sig': None}

            authorized = routing.authorize_route(
                route,
                incoming.get('route_exp'),
                incoming.get('route_sig'),
                tracker_public_key=self._tracker_key,
                allow_unsigned=self.allow_unsigned_routes,
            )
            hop = incoming.get('hop', 0)
            if not isinstance(hop, int) or isinstance(hop, bool) or not 0 <= hop < len(authorized):
                raise ValueError("Indice de salto invalido")
            return {'request_id': request_id, 'payload': incoming['payload'],
                    'route': authorized, 'hop': hop,
                    'route_exp': incoming.get('route_exp'),
                    'route_sig': incoming.get('route_sig')}

        if isinstance(incoming, dict) and set(incoming) == {'request_id', 'payload'}:
            self._validate_request_id(incoming['request_id'])
            return {'request_id': incoming['request_id'], 'payload': incoming['payload'],
                    'route': None, 'hop': 0, 'route_exp': None, 'route_sig': None}

        return {'request_id': str(uuid.uuid4()), 'payload': incoming,
                'route': None, 'hop': 0, 'route_exp': None, 'route_sig': None}

    @staticmethod
    def _validate_request_id(request_id):
        try:
            uuid.UUID(request_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid request_id") from exc

    def _payload_cache_key(self, payload) -> str:
        """Clave de cache sobre el contenido del payload exclusivamente.

        No puede incluir el sobre: `hop` cambia en cada salto y `request_id` en
        cada token, asi que hashear el cuerpo crudo hacia que la cache no
        acertara nunca (fallo silencioso del diseno anterior).
        """
        hasher = hashlib.sha256()

        def visit(value):
            if isinstance(value, torch.Tensor):
                hasher.update(b'T')
                hasher.update(tensor_digest(value))
            elif isinstance(value, dict):
                hasher.update(b'D')
                for key in sorted(value):
                    hasher.update(key.encode('utf-8'))
                    visit(value[key])
            elif isinstance(value, list):
                hasher.update(b'L')
                for item in value:
                    visit(item)
            else:
                hasher.update(b'V')
                hasher.update(repr(value).encode('utf-8'))

        visit(payload)
        return hasher.hexdigest()

    async def handle_forward(self, request):
        """
        Recibe un tensor, aplica una operacion y lo envia al siguiente nodo (o al callback).
        """
        # El limite se aplica ANTES de autenticar: si no, un atacante sin token
        # podria pedir 401 a ritmo ilimitado, que tambien cuesta handshake y CPU.
        throttled = self._throttle(request)
        if throttled is not None:
            return throttled

        if not self._check_auth(request):
            logger.warning(f"[Nodo {self.port}] Peticion rechazada: token de autenticacion invalido o ausente.")
            return web.Response(status=401, text="Unauthorized")

        if self.is_paused:
            logger.warning(f"[Nodo {self.port}] Rechazando peticion (Auto-Pausa Activa: {self.pause_reason}).")
            return web.Response(status=503, text=f"Node is currently busy ({self.pause_reason})")

        body = await request.read()

        try:
            incoming = await asyncio.to_thread(deserialize_tensor, body)
            envelope = self._parse_envelope(incoming)
        except RouteError as exc:
            logger.warning(f"[Nodo {self.port}] Ruta rechazada: {exc}")
            return web.Response(status=403, text="Route rejected")
        except Exception as e:
            logger.error(f"[Nodo {self.port}] Error procesando tensor: {e}")
            return web.Response(status=400, text="Invalid request payload")

        request_id = envelope['request_id']
        payload = envelope['payload']

        try:
            if envelope['route'] is not None:
                target_url, target_node_id, next_index = routing.next_hop(
                    envelope['route'], envelope['hop']
                )
            elif self.next_node_url:
                target_url = routing.normalize_endpoint(self.next_node_url)
                target_node_id, next_index = "", 0
            else:
                logger.error(f"[Nodo {self.port}] No hay URL de destino configurada.")
                return web.Response(status=400, text="No forwarding target configured")
        except RouteError as exc:
            logger.error(f"[Nodo {self.port}] Destino invalido: {exc}")
            return web.Response(status=400, text="Invalid forwarding target")

        try:
            # Hashear el tensor completo cuesta milisegundos: fuera del event loop.
            cache_key = await asyncio.to_thread(self._payload_cache_key, payload) \
                if self.max_cache_size else None
            if cache_key is not None and cache_key in self.kv_cache:
                logger.info(f"[Nodo {self.port}] Cache Eidetica Hit: devolviendo calculo instantaneo sin usar CPU.")
                processed_payload = self.kv_cache[cache_key]
                self.kv_cache.move_to_end(cache_key)
                cached = True
            else:
                if isinstance(payload, dict):
                    logger.info(f"[Nodo {self.port}] Recibido payload tipo dict.")
                elif isinstance(payload, torch.Tensor):
                    logger.info(f"[Nodo {self.port}] Tensor recibido. Forma: {payload.shape}")

                async with self._work_semaphore:
                    processed_payload = await asyncio.to_thread(self.operation, payload)
                cached = False

                if cache_key is not None:
                    self.kv_cache[cache_key] = processed_payload
                    if len(self.kv_cache) > self.max_cache_size:
                        self.kv_cache.popitem(last=False)

            if envelope['route'] is not None:
                outgoing = routing.build_envelope(
                    request_id, processed_payload,
                    route=envelope['route'], hop=next_index,
                    expires_at=envelope['route_exp'], signature=envelope['route_sig'],
                )
            else:
                outgoing = {'request_id': request_id, 'payload': processed_payload}

            self._start_background_task(self._forward_to_next(
                outgoing, target_url, target_node_id,
                error_url=envelope['route'][-1] if envelope['route'] else None,
                request_id=request_id, hop_index=envelope['hop'],
            ))

            suffix = " (Cached)" if cached else ""
            return web.Response(status=202, text=f"Accepted for forwarding{suffix}")

        except Exception as e:
            logger.error(f"[Nodo {self.port}] Error procesando tensor: {e}")
            if envelope['route'] is not None:
                self._start_background_task(self._report_failure(
                    routing.hop_url(envelope['route'][-1]),
                    routing.hop_node_id(envelope['route'][-1]),
                    request_id, envelope['hop'], str(e),
                ))
            return web.Response(status=400, text="Invalid request payload")

    async def handle_callback(self, request):
        """
        Endpoint que usa el cliente original para recibir el resultado final.
        """
        throttled = self._throttle(request)
        if throttled is not None:
            return throttled
        if not self._check_auth(request):
            logger.warning(f"[Nodo {self.port}] Callback rechazado: token de autenticacion invalido o ausente.")
            return web.Response(status=401, text="Unauthorized")
        try:
            body = await request.read()
            tensor = await asyncio.to_thread(deserialize_tensor, body)
            if self.callback_handler:
                if isinstance(tensor, dict) and 'error' in tensor:
                    logger.error(f"[Nodo {self.port}] Error propagado desde el pipeline: {tensor['error']}")
                    callback_value = tensor
                elif isinstance(tensor, dict) and 'payload' in tensor:
                    callback_value = tensor.get('payload')
                else:
                    callback_value = tensor
                self.callback_handler(callback_value)
            return web.Response(status=200, text="Callback received")
        except Exception as e:
            logger.error(f"[Nodo {self.port}] Error procesando callback: {e}")
            return web.Response(status=400, text="Invalid callback payload")

    async def handle_attest(self, request):
        """[Proof of Compute] Ejecuta un reto canonico y firma el resultado.

        La firma la hace la clave privada del certificado del nodo, de modo que
        una salida incorrecta es evidencia no repudiable contra esa identidad.
        """
        # Limite propio y mucho mas estricto: cada atestacion ejecuta inferencia
        # real, asi que es el endpoint mas caro de todo el nodo.
        throttled = self._throttle(request, self.attest_limiter)
        if throttled is not None:
            return throttled

        if not self._check_auth(request):
            return web.Response(status=401, text="Unauthorized")

        try:
            body = await request.json()
            seed = body.get('seed')
            if not isinstance(seed, str) or not 0 < len(seed) <= 128:
                raise ValueError("Semilla invalida")
            seq_len = body.get('seq_len', DEFAULT_ATTEST_SEQ_LEN)
            hidden_size = body.get('hidden_size', self.hidden_size)
            if not isinstance(hidden_size, int):
                raise ValueError("hidden_size desconocido en este nodo")
            challenge = canonical_challenge(seed, hidden_size, seq_len, self.model_arch)
        except (ValueError, TypeError) as exc:
            return web.Response(status=400, text=f"Invalid attestation request: {exc}")

        def _run_challenge():
            with deterministic_inference():
                return self.operation(challenge)

        try:
            async with self._work_semaphore:
                output = await asyncio.to_thread(_run_challenge)
        except Exception as exc:
            logger.error(f"[Nodo {self.port}] Fallo ejecutando el reto de atestacion: {exc}")
            return web.Response(status=500, text="Attestation failed")

        if isinstance(output, dict):
            output = output.get('hidden_states')
        if not isinstance(output, torch.Tensor):
            return web.Response(status=500, text="Attestation produced no tensor")

        # Normalizar a float32 ANTES de firmar: un modelo en bfloat16 devolvería
        # un tensor que `tensor_digest` no puede convertir a numpy, y que el
        # serializador convertiría igualmente a float32, dejando el digest
        # firmado sin corresponder a los bytes que viajan por el cable.
        output = output.detach().to(torch.float32)
        digest = tensor_digest(output)
        signature = b""
        if self._signing_key is not None:
            signature = sign_blob(self._signing_key, attestation_message(self.node_id, seed, digest))

        response = {
            'node_id': self.node_id,
            'seed': seed,
            'layers': self.layers,
            'model_arch': self.model_arch,
            'model_name': self.model_name or "",
            'hidden_size': hidden_size,
            'digest': digest,
            'signature': signature,
            'certificate': self._cert_pem,
            'output': output,
        }
        # secret_sauce desactivado: la verificacion necesita los bytes exactos
        # que se firmaron, y el downcast a float16 los destruiria.
        payload = await asyncio.to_thread(serialize_tensor, response, False)
        return web.Response(status=200, body=payload, content_type='application/msgpack')

    # ------------------------------------------------------------------- envio

    def _get_client_session(self) -> ClientSession:
        """Sesion HTTP unica y persistente.

        Antes se abria un `ClientSession` por reenvio, lo que con TLS significa
        un handshake completo por salto y por token.
        """
        if self._client_session is None or self._client_session.closed:
            connector = TCPConnector(ssl=self._client_ssl_context) if self._client_ssl_context \
                else TCPConnector()
            self._client_session = ClientSession(
                timeout=ClientTimeout(total=30), connector=connector
            )
        return self._client_session

    def _request_kwargs(self, url: str, node_id: str) -> dict:
        """SNI logico: conectamos a la IP pero validamos el `node_id` del cert."""
        kwargs = {}
        if node_id and urlsplit(url).scheme == 'https':
            kwargs['server_hostname'] = node_sni(node_id)
        return kwargs

    async def _post(self, url: str, node_id: str, payload: bytes) -> int:
        headers = {'Content-Type': 'application/msgpack'}
        if self.auth_token:
            headers['X-Auth-Token'] = self.auth_token
        session = self._get_client_session()
        async with session.post(url, data=payload, headers=headers,
                                **self._request_kwargs(url, node_id)) as resp:
            await resp.read()
            return resp.status

    async def _forward_to_next(self, envelope, target_url: str = None, target_node_id: str = "",
                               error_url=None, request_id: str = "", hop_index: int = -1):
        """
        Serializa el sobre y lo manda al siguiente nodo o al callback.
        """
        target_url = target_url or self.next_node_url
        if not target_url:
            logger.error(f"[Nodo {self.port}] No hay URL de destino configurada.")
            return

        try:
            endpoint = routing.normalize_endpoint(target_url)
            payload = await asyncio.to_thread(serialize_tensor, envelope)
        except (TypeError, ValueError, RouteError) as exc:
            logger.error(f"[Nodo {self.port}] Destino o payload invalido: {exc}")
            await self._maybe_report(error_url, request_id, hop_index, str(exc))
            return

        try:
            status = await self._post(endpoint, target_node_id, payload)
            if status not in (200, 202):
                logger.error(f"[Nodo {self.port}] Error al enviar a {endpoint}. Status: {status}")
                await self._maybe_report(error_url, request_id, hop_index,
                                         f"El siguiente salto respondio {status}")
            else:
                logger.info(f"[Nodo {self.port}] Tensor enviado con exito a {endpoint}")
        except asyncio.TimeoutError:
            logger.error(f"[Nodo {self.port}] TIMEOUT: El nodo en {target_url} esta muerto o muy lento.")
            await self._maybe_report(error_url, request_id, hop_index, "Timeout hacia el siguiente salto")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[Nodo {self.port}] Error de conexion enviando a {target_url}: {e}")
            await self._maybe_report(error_url, request_id, hop_index, f"Error de conexion: {e}")

    async def _maybe_report(self, error_url, request_id: str, hop_index: int, message: str):
        if not error_url or not request_id:
            return
        await self._report_failure(routing.hop_url(error_url), routing.hop_node_id(error_url),
                                   request_id, hop_index, message)

    async def _report_failure(self, callback_url: str, callback_node_id: str,
                              request_id: str, hop_index: int, message: str):
        """Avisa al cliente de que el pipeline se rompio, para que no espere al timeout."""
        try:
            envelope = routing.build_error_envelope(request_id, message, hop_index, self.node_id)
            payload = serialize_tensor(envelope)
            endpoint = routing.normalize_endpoint(callback_url, "/callback")
            await self._post(endpoint, callback_node_id, payload)
            logger.info(f"[Nodo {self.port}] Fallo notificado al cliente ({message}).")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[Nodo {self.port}] No se pudo notificar el fallo al cliente: {exc}")

    def _set_low_priority(self):
        """[FASE 5.3 QoS] Reduce la prioridad del proceso para no interferir con el usuario."""
        try:
            p = psutil.Process(os.getpid())
            if os.name == 'nt':
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(10)
            logger.info(f"[Nodo {self.port}] QoS: Proceso establecido a prioridad BAJA.")
        except Exception as e:
            logger.warning(f"[Nodo {self.port}] No se pudo cambiar la prioridad del proceso: {e}")

    def _apply_resource_limits(self):
        """[QoS avanzado] Limita cuantos nucleos dona el usuario.

        `cpu_cores = 0` significa automatico: la mitad de los nucleos, para que
        el PC siga siendo usable. La afinidad ata el proceso a un subconjunto
        concreto; `set_num_threads` evita que PyTorch genere mas hilos que
        nucleos donados (con mas hilos que nucleos el rendimiento empeora).
        """
        # Primera lectura de CPU: `cpu_percent(interval=None)` siempre devuelve
        # 0.0 la primera vez, asi que se ceba aqui para que la primera
        # comprobacion de QoS ya sea un valor real y no un falso "PC ocioso".
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        total_cores = os.cpu_count() or 1
        requested = self.cpu_cores or max(1, total_cores // 2)
        self.donated_cores = max(1, min(total_cores, requested))
        try:
            torch.set_num_threads(self.donated_cores)
        except Exception as exc:
            logger.warning(f"[Nodo {self.port}] No se pudo fijar el numero de hilos: {exc}")
        if self.cpu_cores:
            # Solo se restringe la afinidad si el usuario pidio un numero exacto:
            # en modo automatico conviene dejar al planificador elegir nucleos.
            try:
                psutil.Process(os.getpid()).cpu_affinity(list(range(self.donated_cores)))
            except (AttributeError, OSError, ValueError) as exc:
                logger.warning(f"[Nodo {self.port}] Afinidad de CPU no disponible: {exc}")
        logger.info(f"[Nodo {self.port}] QoS: donando {self.donated_cores}/{total_cores} nucleos, "
                    f"tope de RAM {self.max_ram_percent}%, pausa por CPU al {self.cpu_pause_threshold}%.")

    def _ram_budget_mb(self) -> float:
        return psutil.virtual_memory().total / (1024 * 1024) * (self.max_ram_percent / 100.0)

    def _qos_check(self) -> str:
        """Motivo por el que habria que pausar, o cadena vacia si todo va bien."""
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        if cpu_usage > self.cpu_pause_threshold:
            return f"CPU al {cpu_usage:.0f}% (limite {self.cpu_pause_threshold}%)"
        if available_mb < 512:
            return f"solo quedan {available_mb:.0f}MB de RAM libre"
        try:
            own_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except psutil.Error:
            return ""
        budget_mb = self._ram_budget_mb()
        if own_mb > budget_mb:
            return f"el nodo usa {own_mb:.0f}MB, por encima del {self.max_ram_percent}% donado ({budget_mb:.0f}MB)"
        return ""

    async def _qos_monitor_loop(self):
        """[FASE 5.3 QoS] Vigila si el PC del usuario esta saturado y pausa el nodo."""
        # Inicializar metrica de CPU
        psutil.cpu_percent(interval=None)
        while True:
            await asyncio.sleep(5)
            reason = await asyncio.to_thread(self._qos_check)
            if reason:
                if not self.is_paused:
                    logger.warning(f"[Nodo {self.port}] QoS: pausando el nodo ({reason}).")
                self.is_paused = True
                self.pause_reason = reason
            elif self.is_paused:
                logger.info(f"[Nodo {self.port}] QoS: recursos disponibles de nuevo. Reanudando nodo.")
                self.is_paused = False
                self.pause_reason = ""

    async def _send_heartbeat(self) -> str:
        """Un latido. Devuelve 'ok', 'http_error' o 'network_error'."""
        payload = {
            "node_id": self.node_id,
            "ip": get_lan_ip() if self.host in ('0.0.0.0', '::') else self.host,
            "port": self.port,
            "scheme": self.scheme,
            "ram_mb": int(psutil.virtual_memory().total / (1024 * 1024)),
            "donated_ram_mb": int(self._ram_budget_mb()),
            "donated_cores": getattr(self, "donated_cores", 1),
            "layers": self.layers,
            "is_last": self.is_last,
            "public_key": self.public_key,
            "model_arch": self.model_arch,
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "tls": self.tls_enabled,
            "ts": int(time.time()),
        }
        try:
            session = self._get_client_session()
            headers = {}
            tracker_token = configured_tracker_token()
            if tracker_token:
                headers["X-Node-Token"] = tracker_token
            request_kwargs = {"json": payload}
            if headers:
                request_kwargs["headers"] = headers
            async with session.post(f"{self.tracker_url}/register", **request_kwargs) as resp:
                await resp.read()
                if resp.status == 200:
                    return 'ok'
                logger.warning(f"[Nodo {self.port}] Tracker fallo con estado {resp.status}")
                return 'http_error'
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[Nodo {self.port}] Error conectando al Tracker: {e}")
            return 'network_error'

    async def _reset_client_session(self):
        """Tira la sesion HTTP para forzar sockets nuevos.

        Cuando se cae la conexion del donante, el pool de aiohttp conserva
        sockets muertos y los reintentos siguen fallando aunque vuelva Internet.
        """
        session, self._client_session = self._client_session, None
        if session is not None and not session.closed:
            try:
                await session.close()
            except Exception:
                pass
        logger.info(f"[Nodo {self.port}] Watchdog: conexiones reiniciadas tras perder la red.")

    async def _heartbeat_loop(self):
        """Latido periodico al Tracker, con reintento exponencial y watchdog.

        Antes, si se caia Internet, el nodo se quedaba latiendo contra sockets
        muertos y nunca volvia a aparecer en el enjambre aunque volviera la red.
        """
        consecutive_failures = 0
        while True:
            # Si estamos pausados por QoS, no latimos: que el tracker no nos asigne trabajo.
            if self.is_paused:
                await asyncio.sleep(HEARTBEAT_INTERVAL * 2)
                continue

            result = await self._send_heartbeat()
            if result == 'ok':
                if consecutive_failures:
                    logger.info(f"[Nodo {self.port}] Conexion con el tracker restablecida "
                                f"tras {consecutive_failures} intentos fallidos.")
                consecutive_failures = 0
                delay = heartbeat_delay(0)
            else:
                consecutive_failures += 1
                if result == 'network_error' and consecutive_failures >= 2:
                    await self._reset_client_session()
                delay = heartbeat_delay(consecutive_failures)
                logger.warning(f"[Nodo {self.port}] Reintentando el registro en el tracker "
                               f"en {delay}s (fallo {consecutive_failures}).")

            await asyncio.sleep(delay)

    def _setup_upnp(self):
        try:
            import upnpy
            logger.info(f"[Nodo {self.port}] UPnP: Buscando router para abrir puertos automaticamente...")
            upnp = upnpy.UPnP()
            devices = upnp.discover()
            if not devices:
                logger.warning(f"[Nodo {self.port}] UPnP: No se encontraron routers compatibles en la red local.")
                return False

            device = upnp.get_igd()
            services = device.get_services()
            service = None
            if 'WANPPPConnection.1' in services:
                service = device['WANPPPConnection.1']
            elif 'WANIPConnection.1' in services:
                service = device['WANIPConnection.1']

            if not service:
                logger.warning(f"[Nodo {self.port}] UPnP: El router no soporta port mapping de red externa.")
                return False

            lan_ip = get_lan_ip()

            # Mapear el puerto TCP
            service.AddPortMapping(
                NewRemoteHost='',
                NewExternalPort=self.port,
                NewProtocol='TCP',
                NewInternalPort=self.port,
                NewInternalClient=lan_ip,
                NewEnabled=1,
                NewPortMappingDescription='TokenTorrent Node',
                NewLeaseDuration=0
            )
            logger.info(f"[Nodo {self.port}] UPnP EXITO: Puerto {self.port} abierto al mundo apuntando a {lan_ip}.")
            return True
        except ImportError:
            logger.warning(f"[Nodo {self.port}] UPnP: Libreria 'upnpy' no instalada. El nodo no sera accesible desde Internet.")
        except Exception as e:
            logger.warning(f"[Nodo {self.port}] UPnP Fallo: {e}. Puede que tu router tenga UPnP desactivado.")
        return False

    async def _start_site(self, runner):
        """Arranca el servidor buscando puerto libre si el pedido esta ocupado."""
        requested = self.port
        candidates = [requested]
        if self.port_fallback and requested:
            free = find_free_port(self.host, requested)
            if free != requested:
                candidates.append(free)

        last_error = None
        for candidate in candidates:
            site = web.TCPSite(runner, self.host, candidate, ssl_context=self._server_ssl_context)
            try:
                await site.start()
            except OSError as exc:
                last_error = exc
                logger.warning(f"[Nodo {requested}] Puerto {candidate} no disponible: {exc}")
                continue
            if candidate != requested:
                logger.warning(f"[Nodo {requested}] El puerto {requested} estaba ocupado; "
                               f"el nodo escucha en el {candidate}.")
            # Con puerto 0 el sistema asigna uno efimero: hay que leerlo del socket
            # real, o el heartbeat anunciaria el puerto 0 al tracker.
            self.port = site._server.sockets[0].getsockname()[1] if candidate == 0 else candidate
            return site

        raise last_error or OSError(f"No se pudo abrir ningun puerto desde el {requested}")

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()

        if self.enable_upnp:
            if not self.auth_token and not self.tls_enabled:
                raise ValueError("UPnP requires an authentication token or TLS")
            self._setup_upnp()

        await self._start_site(runner)
        logger.info(f"[Nodo {self.port}] Escuchando en {self.scheme}://{self.host}:{self.port} "
                    f"(node_id={self.node_id}, mTLS={'si' if self.tls_enabled and self.require_client_cert else 'no'})")

        if self.enable_tracker:
            self._start_background_task(self._heartbeat_loop())
        self._start_background_task(self._qos_monitor_loop())

        return runner

    async def _cleanup(self, _app):
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        if self._client_session is not None and not self._client_session.closed:
            await self._client_session.close()
        self._client_session = None
