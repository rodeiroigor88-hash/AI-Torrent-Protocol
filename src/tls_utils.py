"""Utilidades de PKI y TLS/mTLS para el enjambre AI Torrent.

La identidad de un nodo es su certificado X.509, no su IP. Como los nodos
domesticos tienen IP dinamica, el certificado hoja lleva el `node_id` en un
SAN de tipo dNSName (`<node_id>.node.aitorrent`) y el cliente conecta a la IP
pero pasando `server_hostname=<node_id>.node.aitorrent`. Asi la verificacion
de hostname de OpenSSL valida la identidad del nodo, desacoplada de la IP.

Las claves son EC P-256 (ECDSA/SHA-256): universalmente soportadas en TLS y
utilizables tambien para firmar las atestaciones de Proof of Compute.
"""

import datetime
import ipaddress
import os
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

NODE_DNS_SUFFIX = "node.aitorrent"
DEFAULT_CA_DAYS = 3650
DEFAULT_LEAF_DAYS = 397


def node_sni(node_id: str) -> str:
    """Nombre DNS logico de un nodo, usado como `server_hostname` en TLS."""
    return f"{node_id}.{NODE_DNS_SUFFIX}"


def _generate_key():
    return ec.generate_private_key(ec.SECP256R1())


def generate_ca(common_name: str = "AI Torrent Swarm CA", days: int = DEFAULT_CA_DAYS):
    """Crea la CA privada del enjambre. Devuelve (clave_privada, certificado)."""
    key = _generate_key()
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Torrent Protocol"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=True, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def generate_node_cert(node_id: str, ca_key, ca_cert, days: int = DEFAULT_LEAF_DAYS,
                       extra_ips=None, extra_dns=None):
    """Emite un certificado hoja para `node_id`, firmado por la CA del enjambre.

    El certificado sirve a la vez como identidad de servidor y de cliente
    (mTLS), por eso lleva ambos ExtendedKeyUsage.
    """
    key = _generate_key()
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, node_id),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Torrent Node"),
    ])

    alt_names = [x509.DNSName(node_sni(node_id))]
    for name in extra_dns or []:
        alt_names.append(x509.DNSName(name))
    for raw_ip in extra_ips or []:
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(raw_ip)))
        except ValueError:
            continue

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    return key, cert


def write_pem(path: str, key=None, cert=None):
    """Escribe clave y/o certificado en PEM. La clave se crea con permisos 0600."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    blobs = []
    if key is not None:
        blobs.append(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    if cert is not None:
        blobs.append(cert.public_bytes(serialization.Encoding.PEM))

    # La clave privada se crea directamente con permisos restrictivos: abrirla
    # con open() y hacer chmod despues deja una ventana en la que es legible.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = 0o600 if key is not None else 0o644
    fd = os.open(path, flags, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(b"".join(blobs))


def load_cert(path: str):
    with open(path, "rb") as handle:
        return x509.load_pem_x509_certificate(handle.read())


def load_private_key(path: str):
    with open(path, "rb") as handle:
        return serialization.load_pem_private_key(handle.read(), password=None)


def cert_node_id(cert) -> str:
    """Extrae el `node_id` del SAN dNSName `<node_id>.node.aitorrent`."""
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return ""
    for name in san.get_values_for_type(x509.DNSName):
        if name.endswith("." + NODE_DNS_SUFFIX):
            return name[: -(len(NODE_DNS_SUFFIX) + 1)]
    return ""


def _apply_common(context: ssl.SSLContext, certfile: str, keyfile: str, cafile: str):
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    context.load_verify_locations(cafile=cafile)
    return context


def build_server_ssl_context(certfile: str, keyfile: str, cafile: str = None,
                             require_client_cert: bool = True) -> ssl.SSLContext:
    """Contexto TLS de servidor. Con `require_client_cert` exige mTLS."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    if require_client_cert:
        if not cafile:
            raise ValueError("mTLS requiere una CA (--tls-ca) para validar el certificado de cliente")
        context.load_verify_locations(cafile=cafile)
        context.verify_mode = ssl.CERT_REQUIRED
    else:
        context.verify_mode = ssl.CERT_NONE
    return context


def build_client_ssl_context(cafile: str, certfile: str = None, keyfile: str = None) -> ssl.SSLContext:
    """Contexto TLS de cliente: verifica la cadena contra la CA del enjambre y
    presenta el certificado propio si se le da (mTLS)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=cafile)
    if certfile and keyfile:
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


def peer_node_id(request) -> str:
    """`node_id` del certificado de cliente presentado en la conexion mTLS.

    Devuelve cadena vacia si la peticion no llego por TLS o no hubo certificado.
    """
    transport = getattr(request, "transport", None)
    if transport is None:
        return ""
    ssl_object = transport.get_extra_info("ssl_object")
    if ssl_object is None:
        return ""
    try:
        peercert = ssl_object.getpeercert()
    except ValueError:
        return ""
    if not peercert:
        return ""
    for entry_type, value in peercert.get("subjectAltName", ()):
        if entry_type == "DNS" and value.endswith("." + NODE_DNS_SUFFIX):
            return value[: -(len(NODE_DNS_SUFFIX) + 1)]
    return ""


def sign_blob(private_key, data: bytes) -> bytes:
    """Firma ECDSA/SHA-256 con la clave del certificado del nodo."""
    return private_key.sign(data, ec.ECDSA(hashes.SHA256()))


def verify_blob(cert_or_public_key, data: bytes, signature: bytes) -> bool:
    """Verifica una firma `sign_blob`. Acepta certificado o clave publica."""
    public_key = getattr(cert_or_public_key, "public_key", None)
    public_key = public_key() if callable(public_key) else cert_or_public_key
    try:
        public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False
