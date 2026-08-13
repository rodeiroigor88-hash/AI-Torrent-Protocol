"""Tracker de referencia del enjambre (implementacion local de JARVIS).

El tracker en produccion (`jarvis.supercores.host`) vive fuera de este repo.
Este modulo implementa el mismo contrato para poder levantar un enjambre
completo en local, probar el enrutamiento firmado sin tocar el servicio real y
servir de especificacion ejecutable de lo que JARVIS debe hacer.

    python src/gen_certs.py tracker-key --out certs
    python src/tracker.py --port 5000 --signing-key certs/tracker.key

y luego, en el cliente: Ajustes -> URL del Tracker -> http://127.0.0.1:5000

Endpoints:
    POST /register  latido de un nodo
    GET  /plan      camino completo del pipeline, firmado
    GET  /route     heredado: lista de nodos sueltos
    POST /report    divergencia detectada en una auditoria
    GET  /status    vista humana del enjambre
"""

import argparse
import logging
import os
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aiohttp import web

from src import routing
from src.ratelimit import RateLimiter
from src.tls_utils import load_private_key  # noqa: F401  (paridad de API con la PKI)

logger = logging.getLogger(__name__)

# Un nodo al que le tiran del cable deja de latir: a los 45 s (tres latidos de
# 15 s perdidos) se le expulsa para no enrutar hacia un fantasma.
NODE_TIMEOUT = 45.0
ROUTE_TTL = 300           # validez de la firma de una ruta
MAX_NODES = 10_000        # tope del registro; sin el, registrar es un DoS
MAX_PLAN_HOPS = routing.MAX_ROUTE_HOPS - 1   # queda un hueco para el callback

# El registro es barato pero frecuente; planificar lo es menos.
REGISTER_RATE, REGISTER_BURST = 2.0, 10.0
PLAN_RATE, PLAN_BURST = 5.0, 30.0
REPORT_RATE, REPORT_BURST = 0.5, 5.0

# Los cuerpos que aceptamos son diminutos (register/report < 4 KB). El default
# de aiohttp (1 MiB) es un vector de agotamiento de memoria innecesario.
MAX_REQUEST_BYTES = 32 * 1024

# Campos libres del /register: si no se acotan, un nodo malicioso puede
# rellenar el registro con strings gigantescos.
MAX_MODEL_ARCH_LEN = 64
MAX_MODEL_NAME_LEN = 128
MAX_LAYERS_LEN = 32


def _parse_layers(raw):
    """'8-15' -> (8, 15). Devuelve None si no tiene ese formato."""
    if not isinstance(raw, str) or "-" not in raw:
        return None
    start, _, end = raw.partition("-")
    try:
        start, end = int(start), int(end)
    except ValueError:
        return None
    if not 0 <= start <= end <= 4096:
        return None
    return start, end


class Tracker:
    def __init__(self, signing_key=None, node_timeout: float = NODE_TIMEOUT,
                 time_source=time.time, allow_private_callbacks: bool = False):
        self.signing_key = signing_key
        self.node_timeout = node_timeout
        self._time = time_source
        # Un tracker publico jamas debe firmar un callback que apunte a
        # rangos loopback/privados/link-local: convertiria al enjambre en un
        # amplificador SSRF contra servicios internos (incluida la IP de
        # metadatos de nube 169.254.169.254). En un tracker local de pruebas
        # ese callback es legitimo, asi que se puede relajar explicitamente.
        self.allow_private_callbacks = bool(allow_private_callbacks)
        self.nodes = {}
        self.reports = {}

        self.register_limiter = RateLimiter(REGISTER_RATE, REGISTER_BURST)
        self.plan_limiter = RateLimiter(PLAN_RATE, PLAN_BURST)
        self.report_limiter = RateLimiter(REPORT_RATE, REPORT_BURST)

        self.app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        self.app.router.add_post('/register', self.handle_register)
        self.app.router.add_get('/plan', self.handle_plan)
        self.app.router.add_get('/route', self.handle_route)
        self.app.router.add_post('/report', self.handle_report)
        self.app.router.add_get('/status', self.handle_status)

    # -------------------------------------------------------------- utilidades

    def _throttle(self, request, limiter):
        identity = request.remote or "desconocido"
        allowed, retry_after = limiter.allow(identity)
        if allowed:
            return None
        logger.warning("Rate limit para %s en %s", identity, request.path)
        return web.json_response(
            {"error": "too many requests"}, status=429,
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )

    def prune(self):
        """Expulsa a los nodos fantasma. Se ejecuta en cada consulta."""
        deadline = self._time() - self.node_timeout
        stale = [node_id for node_id, node in self.nodes.items() if node["last_seen"] < deadline]
        for node_id in stale:
            logger.info("Nodo fantasma expulsado: %s (sin latido en %.0fs)",
                        node_id, self.node_timeout)
            del self.nodes[node_id]
        return stale

    def active_nodes(self, model_arch=None, exclude=()):
        self.prune()
        excluded = set(exclude or ())
        return [
            node for node in self.nodes.values()
            if node["node_id"] not in excluded
            and (model_arch is None or node["model_arch"] == model_arch)
            and not node["paused"]
        ]

    def plan_route(self, model_arch, start_layer, exclude=()):
        """Encadena nodos cuyos rangos de capas son contiguos.

        Devuelve la lista de saltos o None si la cadena no llega hasta un nodo
        final: una ruta incompleta nunca produciria logits, asi que es mejor no
        entregarla que dejar al cliente esperando un callback que no llegara.
        """
        candidates = self.active_nodes(model_arch, exclude)
        by_start = {}
        for node in candidates:
            # Ante dos nodos con el mismo rango, gana el de latido mas reciente.
            current = by_start.get(node["start_layer"])
            if current is None or node["last_seen"] > current["last_seen"]:
                by_start[node["start_layer"]] = node

        route, next_layer, used = [], start_layer, set()
        while len(route) < MAX_PLAN_HOPS:
            node = by_start.get(next_layer)
            if node is None or node["node_id"] in used:
                break
            route.append({"url": node["forward_url"], "node_id": node["node_id"]})
            used.add(node["node_id"])
            if node["is_last"]:
                return route
            next_layer = node["end_layer"] + 1
        return None

    def sign(self, route, expires_at):
        if self.signing_key is None:
            return None
        return routing.sign_route(self.signing_key, route, expires_at)

    # -------------------------------------------------------------- endpoints

    async def handle_register(self, request):
        throttled = self._throttle(request, self.register_limiter)
        if throttled is not None:
            return throttled

        try:
            body = await request.json()
        except web.HTTPRequestEntityTooLarge:
            # aiohttp lanza esta excepcion cuando el cuerpo excede
            # `client_max_size`; degradar a 400 ocultaria el motivo real.
            return web.json_response({"error": "cuerpo demasiado grande"}, status=413)
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        node_id = body.get("node_id")
        raw_layers = body.get("layers")
        # Un string larguisimo en `layers` obligaria a `_parse_layers` a fallar
        # tras trabajar sobre el; mejor cortar en la puerta.
        if isinstance(raw_layers, str) and len(raw_layers) > MAX_LAYERS_LEN:
            return web.json_response({"error": "rango de capas invalido"}, status=400)
        layers = _parse_layers(raw_layers)
        port = body.get("port")
        if not isinstance(node_id, str) or not 0 < len(node_id) <= 128:
            return web.json_response({"error": "node_id invalido"}, status=400)
        if layers is None:
            return web.json_response({"error": "rango de capas invalido"}, status=400)
        if not isinstance(port, int) or not 0 < port <= 65535:
            return web.json_response({"error": "puerto invalido"}, status=400)

        model_arch = body.get("model_arch") or "unknown"
        if not isinstance(model_arch, str) or len(model_arch) > MAX_MODEL_ARCH_LEN:
            return web.json_response({"error": "model_arch invalido"}, status=400)
        model_name = body.get("model_name")
        if model_name is not None and (
                not isinstance(model_name, str) or len(model_name) > MAX_MODEL_NAME_LEN):
            return web.json_response({"error": "model_name invalido"}, status=400)

        scheme = body.get("scheme") if body.get("scheme") in ("http", "https") else "http"
        # La IP anunciada por el nodo puede ser una IP privada inservible o una
        # mentira; la de la conexion es la unica que sabemos que es alcanzable.
        host = request.remote or body.get("ip") or "127.0.0.1"
        if body.get("ip") and str(body["ip"]).startswith("127."):
            host = body["ip"]   # enjambre local de pruebas

        if node_id not in self.nodes and len(self.nodes) >= MAX_NODES:
            return web.json_response({"error": "registro lleno"}, status=503)

        self.nodes[node_id] = {
            "node_id": node_id,
            "forward_url": f"{scheme}://{host}:{port}/forward",
            "ip": host,
            "port": port,
            "scheme": scheme,
            "layers": raw_layers,
            "start_layer": layers[0],
            "end_layer": layers[1],
            "is_last": bool(body.get("is_last")),
            "model_arch": model_arch,
            "model_name": model_name,
            "hidden_size": body.get("hidden_size"),
            "tls": bool(body.get("tls")),
            "paused": bool(body.get("paused")),
            "donated_cores": body.get("donated_cores"),
            "donated_ram_mb": body.get("donated_ram_mb"),
            "failures": self.reports.get(node_id, 0),
            "last_seen": self._time(),
        }
        return web.json_response({"status": "ok", "expires_in": self.node_timeout})

    async def handle_plan(self, request):
        throttled = self._throttle(request, self.plan_limiter)
        if throttled is not None:
            return throttled

        model_arch = request.query.get("model_arch")
        try:
            start_layer = int(request.query.get("start_layer", 8))
        except ValueError:
            return web.json_response({"error": "start_layer invalido"}, status=400)
        exclude = {item for item in request.query.get("exclude", "").split(",") if item}
        # Un nodo con auditorias fallidas no vuelve a entrar en una ruta.
        exclude |= {node_id for node_id, failures in self.reports.items() if failures > 0}

        # El callback del cliente forma parte de la ruta FIRMADA. Si el cliente
        # pudiera anadirlo despues, el ultimo worker aceptaria entregar a
        # cualquier destino: el enjambre entero seria un amplificador contra la
        # victima que eligiese un cliente malicioso.
        callback = request.query.get("callback")
        if not callback:
            return web.json_response(
                {"error": "falta el parametro callback: la ruta se firma con el destino final"},
                status=400)
        try:
            callback_hop = routing.normalize_endpoint(callback, "/callback")
        except routing.RouteError as exc:
            return web.json_response({"error": f"callback invalido: {exc}"}, status=400)
        # Un tracker publico jamas firma un callback que apunte a rangos que el
        # atacante usaria para amplificar contra la red interna: loopback,
        # RFC1918, link-local (incluida la IP de metadatos de nube).
        if not self.allow_private_callbacks:
            callback_host = urlsplit(callback_hop).hostname or ""
            if not routing.hostname_is_public_safe(callback_host):
                logger.warning("Callback rechazado por SSRF: %s", callback_hop)
                return web.json_response(
                    {"error": "callback invalido: el host apunta a un rango no publico"},
                    status=400)
        callback_node_id = request.query.get("callback_node_id") or ""

        route = self.plan_route(model_arch, start_layer, exclude)
        if not route:
            return web.json_response({"error": "no hay una cadena completa de capas",
                                      "nodes": len(self.active_nodes(model_arch))}, status=404)

        route.append({"url": callback_hop, "node_id": callback_node_id} if callback_node_id
                     else callback_hop)
        normalized = routing.validate_route(route)
        expires_at = int(self._time()) + ROUTE_TTL
        signature = self.sign(normalized, expires_at)
        return web.json_response({
            "route": normalized,
            "exp": expires_at,
            "signature": signature.hex() if signature else None,
        })

    async def handle_route(self, request):
        """Heredado: lista plana de nodos, sin camino."""
        throttled = self._throttle(request, self.plan_limiter)
        if throttled is not None:
            return throttled
        nodes = self.active_nodes(request.query.get("model_arch"))
        return web.json_response({"nodes": [
            {"node_id": node["node_id"], "ip": node["ip"], "port": node["port"],
             "scheme": node["scheme"], "layers": node["layers"]}
            for node in sorted(nodes, key=lambda item: item["start_layer"])
        ]})

    async def handle_report(self, request):
        throttled = self._throttle(request, self.report_limiter)
        if throttled is not None:
            return throttled
        try:
            body = await request.json()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "cuerpo demasiado grande"}, status=413)
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        reported = body.get("nodes")
        if not isinstance(reported, list) or len(reported) > routing.MAX_ROUTE_HOPS:
            return web.json_response({"error": "lista de nodos invalida"}, status=400)

        for node_id in reported:
            if not isinstance(node_id, str) or not node_id:
                continue
            self.reports[node_id] = self.reports.get(node_id, 0) + 1
            if node_id in self.nodes:
                self.nodes[node_id]["failures"] = self.reports[node_id]
            logger.warning("Divergencia reportada contra %s (total %d). Excluido de las rutas.",
                           node_id, self.reports[node_id])
        return web.json_response({"status": "ok", "flagged": len(reported)})

    async def handle_status(self, request):
        self.prune()
        now = self._time()
        return web.json_response({
            "nodes": [
                {"node_id": node["node_id"], "layers": node["layers"],
                 "is_last": node["is_last"], "model_arch": node["model_arch"],
                 "tls": node["tls"], "failures": node["failures"],
                 "seen_ago": round(now - node["last_seen"], 1)}
                for node in sorted(self.nodes.values(), key=lambda item: item["start_layer"])
            ],
            "flagged": self.reports,
            "node_timeout": self.node_timeout,
        })


def main():
    parser = argparse.ArgumentParser(description="Tracker de referencia del enjambre TokenTorrent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--signing-key", default=None,
                        help="Clave privada Ed25519 para firmar rutas (src/gen_certs.py tracker-key)")
    parser.add_argument("--node-timeout", type=float, default=NODE_TIMEOUT)
    parser.add_argument("--allow-private-callbacks", action="store_true",
                        help="Permite callbacks a loopback/RFC1918/link-local. Solo para "
                             "trackers locales de pruebas: en publico habilita SSRF.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s] - %(message)s')

    signing_key = None
    if args.signing_key:
        from cryptography.hazmat.primitives import serialization
        with open(args.signing_key, "rb") as handle:
            signing_key = serialization.load_pem_private_key(handle.read(), password=None)
        logger.info("Rutas firmadas con %s", args.signing_key)
    else:
        logger.warning("Sin --signing-key: las rutas van SIN FIRMAR y solo las aceptaran "
                       "nodos en loopback o arrancados con --allow-unsigned-routes.")

    tracker = Tracker(signing_key=signing_key, node_timeout=args.node_timeout,
                      allow_private_callbacks=args.allow_private_callbacks)
    if args.allow_private_callbacks:
        logger.warning("--allow-private-callbacks activo: solo apto para pruebas locales.")
    logger.info("Tracker escuchando en http://%s:%s (timeout de nodo: %.0fs)",
                args.host, args.port, args.node_timeout)
    web.run_app(tracker.app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
