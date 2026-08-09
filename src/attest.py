"""Verificador de Proof of Compute: reta a un nodo y comprueba su respuesta.

Carga localmente el mismo rango de capas que dice alojar el nodo, calcula la
salida de referencia con la entrada canónica del reto y compara. Detecta al
nodo que devuelve ruido, al que se salta capas y al que aloja otro modelo.

    python src/attest.py --node https://192.168.1.40:8001 --layers 8-15 \
        --tls-ca certs/ca.crt --tls-cert certs/cliente.crt --tls-key certs/cliente.key

La comparación es de L2 relativa bajo `--epsilon`, nunca bit a bit: el downcast
a float16 y la variación de hardware lo hacen imposible.
"""

import argparse
import asyncio
import datetime
import os
import secrets
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import aiohttp
import torch
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec

from src.pow_utils import (
    DEFAULT_ATTEST_SEQ_LEN,
    DEFAULT_EPSILON,
    attestation_message,
    canonical_challenge,
    cosine_similarity,
    deterministic_inference,
    relative_l2,
    tensor_digest,
)
from src.tensor_utils import deserialize_tensor
from src.tls_utils import build_client_ssl_context, cert_node_id, node_sni, verify_blob
from src.worker import GenericWorkerModel


def verify_certificate_chain(certificate, ca_cert) -> str:
    """Valida firma y vigencia del certificado del nodo contra la CA."""
    now = datetime.datetime.now(datetime.timezone.utc)
    not_before = certificate.not_valid_before_utc
    not_after = certificate.not_valid_after_utc
    if not not_before <= now <= not_after:
        return f"certificado fuera de vigencia ({not_before} .. {not_after})"
    try:
        ca_cert.public_key().verify(
            certificate.signature,
            certificate.tbs_certificate_bytes,
            ec.ECDSA(certificate.signature_hash_algorithm),
        )
    except Exception as exc:
        return f"el certificado no está firmado por esta CA: {exc}"
    return ""


async def challenge_node(url, seed, seq_len, hidden_size, ssl_context, node_id, auth_token):
    headers = {}
    if auth_token:
        headers['X-Auth-Token'] = auth_token
    kwargs = {}
    if node_id and url.startswith('https://'):
        kwargs['server_hostname'] = node_sni(node_id)

    connector = aiohttp.TCPConnector(ssl=ssl_context) if ssl_context else aiohttp.TCPConnector()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),
                                     connector=connector) as session:
        async with session.post(f"{url.rstrip('/')}/attest",
                                json={"seed": seed, "seq_len": seq_len, "hidden_size": hidden_size},
                                headers=headers, **kwargs) as resp:
            body = await resp.read()
            if resp.status != 200:
                raise SystemExit(f"[FALLO] El nodo respondió {resp.status}: {body[:200]!r}")
    return deserialize_tensor(body)


def main():
    parser = argparse.ArgumentParser(description="Reto de atestación (Proof of Compute)")
    parser.add_argument("--node", required=True, help="URL base del nodo (sin /attest)")
    parser.add_argument("--layers", required=True, help="Rango de capas que dice alojar (ej: 8-15)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--is-last", action="store_true", help="El nodo aloja norm final + lm_head")
    parser.add_argument("--seed", default=None, help="Semilla del reto (por defecto, aleatoria)")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_ATTEST_SEQ_LEN)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--node-id", default=None, help="Identidad esperada (SNI y firma)")
    parser.add_argument("--auth-token", default=os.environ.get("TOKENTORRENT_AUTH_TOKEN"))
    parser.add_argument("--tls-ca", default=os.environ.get("TOKENTORRENT_TLS_CA"))
    parser.add_argument("--tls-cert", default=os.environ.get("TOKENTORRENT_TLS_CERT"))
    parser.add_argument("--tls-key", default=os.environ.get("TOKENTORRENT_TLS_KEY"))
    args = parser.parse_args()

    # La semilla debe ser impredecible: si el nodo la conociera de antemano
    # podría precalcular la respuesta sin alojar realmente las capas.
    seed = args.seed or secrets.token_hex(16)
    start_layer, end_layer = map(int, args.layers.split("-"))

    print(f"[*] Cargando referencia local: {args.model} capas {start_layer}-{end_layer}...")
    reference_model = GenericWorkerModel(args.model, start_layer, end_layer, is_last=args.is_last)
    hidden_size = reference_model.hidden_size

    challenge = canonical_challenge(seed, hidden_size, args.seq_len, reference_model.arch)
    with deterministic_inference():
        reference = reference_model(challenge)
    if isinstance(reference, dict):
        reference = reference['hidden_states']

    ssl_context = None
    if args.tls_ca:
        ssl_context = build_client_ssl_context(args.tls_ca, args.tls_cert, args.tls_key)

    print(f"[*] Retando a {args.node} con semilla {seed}...")
    response = asyncio.run(challenge_node(
        args.node, seed, args.seq_len, hidden_size, ssl_context, args.node_id, args.auth_token
    ))

    output = response.get('output')
    if not isinstance(output, torch.Tensor):
        raise SystemExit("[FALLO] El nodo no devolvió ningún tensor.")

    node_id = response.get('node_id', '')
    problems = []

    # 1. Integridad: el digest firmado debe corresponder al tensor recibido.
    recomputed = tensor_digest(output)
    if recomputed != response.get('digest'):
        problems.append("el digest no corresponde al tensor devuelto")

    # 2. Identidad: firma válida y certificado emitido por la CA del enjambre.
    certificate_pem = response.get('certificate') or b""
    signature = response.get('signature') or b""
    if certificate_pem and signature:
        certificate = x509.load_pem_x509_certificate(certificate_pem)
        if cert_node_id(certificate) != node_id:
            problems.append("el node_id no coincide con el certificado presentado")
        if not verify_blob(certificate, attestation_message(node_id, seed, recomputed), signature):
            problems.append("firma de atestación inválida")
        if args.tls_ca:
            with open(args.tls_ca, "rb") as handle:
                ca_cert = x509.load_pem_x509_certificate(handle.read())
            chain_error = verify_certificate_chain(certificate, ca_cert)
            if chain_error:
                problems.append(chain_error)
        if args.node_id and args.node_id != node_id:
            problems.append(f"identidad inesperada: {node_id} != {args.node_id}")
    else:
        problems.append("respuesta SIN FIRMAR: el nodo no tiene identidad criptográfica "
                        "(sin ella no hay reputación posible)")

    # 3. Cálculo: comparación numérica bajo tolerancia.
    divergence = relative_l2(output, reference)
    similarity = cosine_similarity(output, reference)
    computed_ok = output.shape == reference.shape and divergence <= args.epsilon

    print("\n=== RESULTADO DE LA ATESTACIÓN ===")
    print(f"  node_id            : {node_id or '(sin identidad)'}")
    print(f"  capas declaradas   : {response.get('layers')}")
    print(f"  modelo declarado   : {response.get('model_name')}")
    print(f"  forma esperada     : {tuple(reference.shape)}")
    print(f"  forma recibida     : {tuple(output.shape)}")
    print(f"  L2 relativa        : {divergence:.6f}  (umbral {args.epsilon})")
    print(f"  similitud coseno   : {similarity:.6f}")
    for problem in problems:
        print(f"  [!] {problem}")

    if computed_ok and not problems:
        print("\n[OK] El nodo ejecutó realmente las capas declaradas.")
        return 0
    if not computed_ok:
        print("\n[FALLO] La salida no corresponde a las capas declaradas: el nodo "
              "devuelve ruido, se salta capas o aloja otro modelo.")
    else:
        print("\n[FALLO] El cálculo es correcto pero la identidad no es verificable.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
