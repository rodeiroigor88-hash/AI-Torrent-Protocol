"""Enrutamiento en origen (source routing) del pipeline y firma de rutas.

En el sobre v1 el camino estaba congelado en los `--next-node` de cada worker,
asi que el tracker no podia reencaminar nada. En el sobre v2 el camino viaja
con el tensor:

    {v: 2, request_id, route: [hop0, hop1, ..., callback], hop: n, payload}

Cada nodo lee `route[hop + 1]` como destino y reenvia incrementando `hop`. El
ultimo elemento de la ruta es siempre el `/callback` del cliente.

Una lista de URLs que llega por la red es un vector de SSRF y amplificacion:
un cliente malicioso podria hacer que N nodos del enjambre POSTeen contra la
victima que elija. Por eso una ruta solo se acepta si (a) todos sus saltos son
loopback, (b) el nodo arranco con `--allow-unsigned-routes`, o (c) viene
firmada por el tracker con Ed25519 y el nodo tiene su clave publica.
"""

import ipaddress
import time
from urllib.parse import urlsplit, urlunsplit

import msgpack
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

ENVELOPE_VERSION = 2
MAX_ROUTE_HOPS = 16
MAX_URL_LENGTH = 512
ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PATH_SUFFIXES = ("/forward", "/callback")


class RouteError(ValueError):
    """Ruta invalida, no autorizada o agotada."""


def normalize_endpoint(url: str, default_suffix: str = "/forward") -> str:
    """Valida una URL de salto y le anade `/forward` si no trae endpoint."""
    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        raise RouteError("URL de destino invalida")
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES or not parts.hostname:
        raise RouteError(f"Esquema o host no permitido en {url!r}")
    if parts.username or parts.password:
        raise RouteError("Las URLs con credenciales no estan permitidas")
    path = parts.path.rstrip("/")
    if not path.endswith(ALLOWED_PATH_SUFFIXES):
        path += default_suffix
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _hostname_is_loopback(hostname: str) -> bool:
    """Solo IP literales de loopback y `localhost`.

    Deliberadamente NO se resuelve por DNS: la validacion corre en el event loop
    y una ruta hostil con un nombre de resolucion lenta lo bloquearia. Ademas,
    resolver permitiria burlar la comprobacion con un dominio que apunta a
    127.0.0.1 y luego cambia (DNS rebinding).
    """
    if hostname in ("localhost", "localhost.localdomain"):
        return True
    clean_host = hostname.strip("[]")
    try:
        return ipaddress.ip_address(clean_host).is_loopback
    except ValueError:
        return False


def hostname_is_public_safe(hostname: str) -> bool:
    """True si el hostname parece un destino publico legitimo.

    Rechaza literales IP que caen en rangos que un tracker publico NUNCA
    deberia firmar como callback: loopback, RFC1918, link-local (incluida la
    IP de metadatos de nube 169.254.169.254), multicast, reservados y no
    especificados. Para nombres de dominio devuelve True: la resolucion se
    delega al momento de conexion (resolver en el tracker abre DNS rebinding
    y bloquea el event loop). Es una defensa en profundidad; el worker sigue
    siendo responsable de sus propias comprobaciones al enviar.
    """
    if not hostname:
        return False
    if hostname in ("localhost", "localhost.localdomain"):
        return False
    clean_host = hostname.strip("[]")
    try:
        addr = ipaddress.ip_address(clean_host)
        # `is_global` cubre loopback, privadas, link-local, multicast, reservadas,
        # site-local IPv6 y unspecified de una sola vez.
        return bool(addr.is_global)
    except ValueError:
        # Si contiene ':' (como IPv6 sin corchetes o mal formado) o no contiene '.'
        # (nombres de una sola etiqueta como NetBIOS, mDNS o 'fe80'), no es un FQDN publico valido.
        if ":" in hostname or "." not in hostname:
            return False
        # Es un nombre de dominio calificado (FQDN). No resolvemos aqui a proposito.
        return True


def route_is_loopback_only(route) -> bool:
    for entry in route:
        url = hop_url(entry)
        if not _hostname_is_loopback(urlsplit(url).hostname or ""):
            return False
    return True


def hop_url(entry) -> str:
    """Un salto es una URL, o un dict {'url', 'node_id'} cuando hace falta SNI."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("url"), str):
        return entry["url"]
    raise RouteError("Formato de salto invalido")


def hop_node_id(entry) -> str:
    if isinstance(entry, dict) and isinstance(entry.get("node_id"), str):
        return entry["node_id"]
    return ""


def validate_route(route) -> list:
    """Comprueba forma y tamano de la ruta y normaliza cada endpoint."""
    if not isinstance(route, list) or not route:
        raise RouteError("La ruta debe ser una lista no vacia")
    if len(route) > MAX_ROUTE_HOPS:
        raise RouteError(f"La ruta excede el maximo de {MAX_ROUTE_HOPS} saltos")

    normalized = []
    for index, entry in enumerate(route):
        default_suffix = "/callback" if index == len(route) - 1 else "/forward"
        url = normalize_endpoint(hop_url(entry), default_suffix)
        node_id = hop_node_id(entry)
        normalized.append({"url": url, "node_id": node_id} if node_id else url)
    return normalized


def canonical_route_bytes(route, expires_at: int) -> bytes:
    """Representacion canonica que firma el tracker.

    IMPORTANTE: el tracker debe firmar la ruta ya normalizada por
    `validate_route`, porque es sobre esa forma sobre la que el nodo verifica.
    Firmar `http://host:8001` cuando el nodo verificara
    `http://host:8001/forward` produce un fallo de firma.
    """
    return msgpack.packb(
        {"route": [hop_url(entry) for entry in route],
         "node_ids": [hop_node_id(entry) for entry in route],
         "exp": int(expires_at)},
        use_bin_type=True,
    )


def sign_route(private_key: ed25519.Ed25519PrivateKey, route, expires_at: int) -> bytes:
    return private_key.sign(canonical_route_bytes(route, expires_at))


def verify_route_signature(public_key: ed25519.Ed25519PublicKey, route,
                           expires_at: int, signature: bytes) -> bool:
    if not isinstance(signature, bytes):
        return False
    try:
        public_key.verify(signature, canonical_route_bytes(route, expires_at))
        return True
    except (InvalidSignature, TypeError, ValueError):
        return False


def load_tracker_public_key(path: str) -> ed25519.Ed25519PublicKey:
    with open(path, "rb") as handle:
        raw = handle.read()
    key = serialization.load_pem_public_key(raw)
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise ValueError("La clave del tracker debe ser Ed25519")
    return key


def generate_tracker_keypair():
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def authorize_route(route, expires_at, signature, tracker_public_key=None,
                    allow_unsigned: bool = False) -> list:
    """Aplica la politica de aceptacion de rutas descrita en el modulo.

    Devuelve la ruta normalizada o lanza `RouteError`.
    """
    normalized = validate_route(route)

    if tracker_public_key is not None and signature:
        if not isinstance(expires_at, int):
            raise RouteError("La ruta firmada necesita un campo de expiracion entero")
        if expires_at < time.time():
            raise RouteError("La ruta firmada ha caducado")
        if not verify_route_signature(tracker_public_key, normalized, expires_at, signature):
            raise RouteError("Firma de ruta invalida")
        return normalized

    if allow_unsigned or route_is_loopback_only(normalized):
        return normalized

    raise RouteError("Ruta sin firmar hacia destinos remotos: rechazada")


def build_envelope(request_id: str, payload, route=None, hop: int = 0,
                   expires_at: int = None, signature: bytes = None) -> dict:
    envelope = {"v": ENVELOPE_VERSION, "request_id": request_id, "payload": payload}
    if route is not None:
        envelope["route"] = route
        envelope["hop"] = hop
        if expires_at is not None:
            envelope["route_exp"] = int(expires_at)
        if signature is not None:
            envelope["route_sig"] = signature
    return envelope


def build_error_envelope(request_id: str, message: str, failed_hop: int = -1,
                         node_id: str = "") -> dict:
    return {
        "v": ENVELOPE_VERSION,
        "request_id": request_id,
        "error": {"message": str(message)[:256], "hop": int(failed_hop), "node_id": node_id},
    }


def next_hop(route, hop: int):
    """Destino del salto siguiente. Devuelve (url, node_id, nuevo_hop)."""
    index = hop + 1
    if index >= len(route):
        raise RouteError("Ruta agotada: no hay siguiente salto")
    entry = route[index]
    return hop_url(entry), hop_node_id(entry), index
