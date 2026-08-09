import argparse
import asyncio
import logging
import logging.handlers
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM

from src.config import load_config
from src.p2p_node import P2PNode

logger = logging.getLogger(__name__)


def default_log_path() -> str:
    """El worker se empaqueta con --noconsole: sin fichero de log, un fallo de
    arranque (token ausente, certificado ilegible) es completamente invisible."""
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        directory = os.path.join(base, 'GhostTerminal')
    else:
        directory = os.path.join(os.path.expanduser('~'), '.ai-torrent')
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return os.path.join(os.path.abspath('.'), 'worker.log')
    return os.path.join(directory, 'worker.log')


def configure_logging(log_file: str = None, level: int = logging.INFO) -> str:
    log_file = log_file or default_log_path()
    root = logging.getLogger()
    if getattr(root, '_ai_torrent_configured', False):
        return log_file

    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.handlers.RotatingFileHandler(
            log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding='utf-8'
        ))
    except OSError as exc:  # disco lleno o ruta no escribible: seguimos con stdout
        print(f"[WARN] No se pudo abrir el log {log_file}: {exc}", file=sys.stderr)

    root.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root._ai_torrent_configured = True
    return log_file


def first_set(*values):
    """Primer valor que no sea None (la bandera CLI gana sobre la config)."""
    for value in values:
        if value is not None:
            return value
    return None


class LoggingArgumentParser(argparse.ArgumentParser):
    """argparse escribe los errores en stderr, que con --noconsole no existe."""

    def error(self, message):
        configure_logging()
        logger.error(f"Argumentos invalidos: {message}")
        super().error(message)


class GenericWorkerModel(torch.nn.Module):
    """
    Un modelo trabajador genérico que carga solo un rango de capas de un modelo HuggingFace.
    """
    def __init__(self, model_name: str, start_layer: int, end_layer: int, is_last: bool = False):
        super().__init__()
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_last = is_last

        logger.info(f"Cargando modelo base {model_name}... (Esto puede tardar)")
        # Cargamos el modelo completo (en una versión real de producción, podríamos descargar solo los safetensors necesarios)
        # pero para esta prueba, cargamos el modelo y extraemos las capas
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # hidden_size se guarda antes de liberar el modelo base: lo necesita el
        # reto de atestación (/attest) para construir la entrada canónica.
        self.hidden_size = getattr(base_model.config, 'hidden_size', None) or \
            getattr(base_model.config, 'n_embd', None)

        # Detectamos la arquitectura (Qwen/Llama usan model.layers, GPT-2 usa transformer.h)
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
            self.arch = 'llama/qwen'
            self.rotary_emb = base_model.model.rotary_emb
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            layers = base_model.transformer.h
            self.arch = 'gpt2'
            self.rotary_emb = None
        else:
            raise ValueError("Arquitectura de modelo no soportada para sharding automático.")

        if not 0 <= start_layer <= end_layer < len(layers):
            raise ValueError(
                f"Rango de capas inválido: {start_layer}-{end_layer}; "
                f"el modelo contiene {len(layers)} capas"
            )

        # Extraemos las capas que nos tocan
        self.blocks = torch.nn.ModuleList([layers[i] for i in range(start_layer, min(end_layer + 1, len(layers)))])

        # Si somos el último nodo, necesitamos la cabeza del LM y la normalización final
        if self.is_last:
            if self.arch == 'llama/qwen':
                self.norm = base_model.model.norm
            else:
                self.norm = base_model.transformer.ln_f
            self.lm_head = base_model.lm_head
        else:
            self.norm = None
            self.lm_head = None

        logger.info(f"Worker inicializado con {len(self.blocks)} capas (desde la {start_layer} hasta la {end_layer})")
        # Liberar el base_model de memoria (ayuda al recolector de basura)
        del base_model

    def forward(self, payload):
        with torch.no_grad():
            if isinstance(payload, dict):
                if 'hidden_states' not in payload:
                    raise ValueError("El payload no contiene hidden_states")
                hidden_states = payload['hidden_states']
                position_ids = payload.get('position_ids')
            else:
                hidden_states = payload
                position_ids = None
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                raise ValueError("hidden_states debe ser un tensor tridimensional")
            if self.arch == 'llama/qwen' and not isinstance(position_ids, torch.Tensor):
                raise ValueError("Los modelos Qwen/Llama requieren position_ids")

            # Match model dtype (e.g. bfloat16) to avoid mat1/mat2 dtype mismatch
            if len(list(self.blocks.parameters())) > 0:
                model_dtype = next(self.blocks.parameters()).dtype
                if hidden_states.dtype != model_dtype:
                    hidden_states = hidden_states.to(model_dtype)

            # Calculate RoPE embeddings if needed
            position_embeddings = None
            attention_mask = None
            if self.arch == 'llama/qwen' and position_ids is not None and self.rotary_emb is not None:
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

            if self.is_last:
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
                return logits

            # Retornar payload modificado
            if position_ids is not None:
                return {'hidden_states': hidden_states, 'position_ids': position_ids}
            return hidden_states

async def main():
    # Los valores por defecto salen del fichero de configuración, para que el
    # usuario pueda ajustarlos sin reescribir la entrada de auto-arranque.
    settings = load_config()

    parser = LoggingArgumentParser(description="AI Torrent Protocol - Worker Node")
    parser.add_argument("--port", type=int, default=settings["worker_port"],
                        help=f"Puerto para escuchar peticiones (config: {settings['worker_port']})")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Nombre del modelo en HuggingFace")
    parser.add_argument("--layers", type=str, required=True, help="Rango de capas (ej: 4-7)")
    parser.add_argument("--next-node", type=str, default=None,
                        help="URL del siguiente nodo (sobre v1); con rutas del tracker no hace falta")
    parser.add_argument("--is-last", action="store_true", help="Indica si este es el último nodo de la cadena")
    parser.add_argument("--auth-token", type=str, default=os.environ.get("AI_TORRENT_AUTH_TOKEN"),
                         help="Secreto compartido del enjambre; si se define, /forward y /callback lo exigen en la cabecera X-Auth-Token")
    parser.add_argument("--host", default="127.0.0.1", help="Interfaz de escucha; fuera de localhost exige TLS")
    parser.add_argument("--enable-upnp", action="store_true", help="Abre el puerto en el router; requiere autenticación")
    parser.add_argument("--enable-tracker", action="store_true", help="Publica el worker en el tracker configurado")
    parser.add_argument("--cache-size", type=int, default=0, help="Entradas de caché KV (0 = desactivada)")
    parser.add_argument("--log-file", type=str, default=None, help="Ruta del log rotatorio del worker")

    qos_group = parser.add_argument_group("Donación de recursos (QoS)")
    qos_group.add_argument("--max-ram-percent", type=int, default=None,
                           help=f"%% de la RAM total que puede ocupar el nodo (config: {settings['max_ram_percent']})")
    qos_group.add_argument("--cpu-cores", type=int, default=None,
                           help=f"Núcleos exactos a donar; 0 = la mitad (config: {settings['cpu_cores']})")
    qos_group.add_argument("--cpu-pause-threshold", type=int, default=None,
                           help=f"%% de CPU global a partir del cual el nodo se pausa (config: {settings['cpu_pause_threshold']})")
    qos_group.add_argument("--no-port-fallback", action="store_true",
                           help="Falla si el puerto está ocupado en vez de buscar el siguiente libre")

    tls_group = parser.add_argument_group("TLS / mTLS")
    tls_group.add_argument("--tls-cert", type=str, default=os.environ.get("AI_TORRENT_TLS_CERT"),
                           help="Certificado del nodo en PEM (ver src/gen_certs.py)")
    tls_group.add_argument("--tls-key", type=str, default=os.environ.get("AI_TORRENT_TLS_KEY"),
                           help="Clave privada del nodo en PEM")
    tls_group.add_argument("--tls-ca", type=str, default=os.environ.get("AI_TORRENT_TLS_CA"),
                           help="Certificado de la CA del enjambre")
    tls_group.add_argument("--no-client-cert", action="store_true",
                           help="Acepta clientes sin certificado (TLS simple en vez de mTLS)")
    tls_group.add_argument("--insecure-no-tls", action="store_true",
                           help="Permite escuchar fuera de localhost sin TLS. El tráfico y el token viajan en claro.")

    routing_group = parser.add_argument_group("Enrutamiento")
    routing_group.add_argument("--tracker-key", type=str, default=os.environ.get("AI_TORRENT_TRACKER_KEY"),
                               help="Clave pública Ed25519 del tracker para validar rutas firmadas")
    routing_group.add_argument("--allow-unsigned-routes", action="store_true",
                               help="Acepta rutas sin firmar hacia destinos remotos (riesgo de SSRF)")

    args = parser.parse_args()
    log_file = configure_logging(args.log_file)
    logger.info(f"Worker arrancando. Log en {log_file}")

    is_local = args.host in ("127.0.0.1", "::1", "localhost")
    tls_configured = bool(args.tls_cert and args.tls_key)

    if args.tls_cert and not args.tls_key:
        parser.error("--tls-cert requiere --tls-key")
    if tls_configured and not args.no_client_cert and not args.tls_ca:
        parser.error("mTLS requiere --tls-ca (usa --no-client-cert para TLS simple)")
    if not is_local and not tls_configured and not args.insecure_no_tls:
        parser.error("escuchar fuera de localhost exige TLS (--tls-cert/--tls-key) "
                     "o aceptar el riesgo explícitamente con --insecure-no-tls")
    if not is_local and not tls_configured and not args.auth_token:
        parser.error("--auth-token es obligatorio al escuchar fuera de localhost sin TLS")
    if args.enable_upnp and not (args.auth_token or tls_configured):
        parser.error("--enable-upnp requiere --auth-token o TLS")
    if not is_local and not tls_configured:
        logger.warning("Escuchando fuera de localhost SIN TLS: los tensores y el token "
                       "viajan en claro y son interceptables.")

    try:
        start_layer, end_layer = map(int, args.layers.split("-"))
    except ValueError:
        parser.error("--layers debe tener el formato inicio-fin (ej: 8-15)")

    # Inicializar el modelo
    model = GenericWorkerModel(
        model_name=args.model,
        start_layer=start_layer,
        end_layer=end_layer,
        is_last=args.is_last
    )

    # Inicializar el Nodo P2P
    node = P2PNode(
        host=args.host,
        port=args.port,
        next_node_url=args.next_node,
        operation=model,
        layers=args.layers,
        auth_token=args.auth_token,
        model_name=args.model,
        model_arch=model.arch,
        enable_upnp=args.enable_upnp,
        enable_tracker=args.enable_tracker,
        cache_size=args.cache_size,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        tls_ca=args.tls_ca,
        require_client_cert=not args.no_client_cert,
        tracker_key=args.tracker_key,
        allow_unsigned_routes=args.allow_unsigned_routes,
        hidden_size=model.hidden_size,
        is_last=args.is_last,
        # La bandera CLI gana; si no se pasa, manda el fichero de configuración.
        max_ram_percent=first_set(args.max_ram_percent, settings["max_ram_percent"]),
        cpu_cores=first_set(args.cpu_cores, settings["cpu_cores"]),
        cpu_pause_threshold=first_set(args.cpu_pause_threshold, settings["cpu_pause_threshold"]),
        port_fallback=not args.no_port_fallback,
    )

    logger.info(f"Iniciando Worker en puerto {args.port}...")
    await node.start()

    # Mantener el proceso vivo indefinidamente
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception:
        # Sin consola, una excepción no capturada desaparecería sin rastro.
        configure_logging()
        logging.getLogger(__name__).exception("El worker terminó con una excepción no controlada")
        sys.exit(1)
