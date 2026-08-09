"""Generador de la PKI del enjambre: CA, certificados de nodo y clave del tracker.

Uso tipico para montar un enjambre local con mTLS:

    python src/gen_certs.py init-ca --out certs
    python src/gen_certs.py node --ca-dir certs --name worker1 --ip 192.168.1.40
    python src/gen_certs.py node --ca-dir certs --name cliente
    python src/gen_certs.py tracker-key --out certs

Cada nodo necesita su `<name>.crt` + `<name>.key` y el `ca.crt` comun. La clave
privada de la CA (`ca.key`) no debe distribuirse: solo la usa quien emite
certificados.
"""

import argparse
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cryptography.hazmat.primitives import serialization

from src.routing import generate_tracker_keypair
from src.tls_utils import (
    DEFAULT_CA_DAYS,
    DEFAULT_LEAF_DAYS,
    generate_ca,
    generate_node_cert,
    load_cert,
    load_private_key,
    node_sni,
    write_pem,
)


def cmd_init_ca(args):
    ca_key_path = os.path.join(args.out, "ca.key")
    ca_cert_path = os.path.join(args.out, "ca.crt")
    if os.path.exists(ca_key_path) and not args.force:
        raise SystemExit(f"Ya existe {ca_key_path}. Usa --force para sobrescribir "
                         f"(invalidaria todos los certificados emitidos).")

    key, cert = generate_ca(common_name=args.common_name, days=args.days)
    write_pem(ca_key_path, key=key)
    write_pem(ca_cert_path, cert=cert)
    print(f"[+] CA creada:\n    clave  {ca_key_path}  (NO distribuir)\n    cert   {ca_cert_path}")


def cmd_node(args):
    ca_key = load_private_key(os.path.join(args.ca_dir, "ca.key"))
    ca_cert = load_cert(os.path.join(args.ca_dir, "ca.crt"))

    node_id = args.node_id or str(uuid.uuid4())
    out_dir = args.out or args.ca_dir
    key_path = os.path.join(out_dir, f"{args.name}.key")
    cert_path = os.path.join(out_dir, f"{args.name}.crt")
    if os.path.exists(key_path) and not args.force:
        raise SystemExit(f"Ya existe {key_path}. Usa --force para sobrescribir.")

    key, cert = generate_node_cert(
        node_id, ca_key, ca_cert, days=args.days,
        extra_ips=args.ip, extra_dns=args.dns,
    )
    write_pem(key_path, key=key)
    write_pem(cert_path, cert=cert)
    print(f"[+] Nodo '{args.name}' emitido:\n"
          f"    node_id {node_id}\n"
          f"    SNI     {node_sni(node_id)}\n"
          f"    clave   {key_path}\n"
          f"    cert    {cert_path}")


def cmd_tracker_key(args):
    private_path = os.path.join(args.out, "tracker.key")
    public_path = os.path.join(args.out, "tracker.pub")
    if os.path.exists(private_path) and not args.force:
        raise SystemExit(f"Ya existe {private_path}. Usa --force para sobrescribir.")

    private_key, public_key = generate_tracker_keypair()
    os.makedirs(args.out, exist_ok=True)

    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(public_path, "wb") as handle:
        handle.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ))
    print(f"[+] Clave del tracker (Ed25519):\n"
          f"    privada {private_path}  (solo el tracker)\n"
          f"    publica {public_path}   (se reparte a los nodos: --tracker-key)")


def build_parser():
    parser = argparse.ArgumentParser(description="PKI del enjambre TokenTorrent")
    sub = parser.add_subparsers(dest="command", required=True)

    init_ca = sub.add_parser("init-ca", help="Crea la CA privada del enjambre")
    init_ca.add_argument("--out", default="certs", help="Directorio de salida")
    init_ca.add_argument("--common-name", default="TokenTorrent Swarm CA")
    init_ca.add_argument("--days", type=int, default=DEFAULT_CA_DAYS)
    init_ca.add_argument("--force", action="store_true")
    init_ca.set_defaults(func=cmd_init_ca)

    node = sub.add_parser("node", help="Emite el certificado de un nodo")
    node.add_argument("--ca-dir", default="certs", help="Directorio con ca.key y ca.crt")
    node.add_argument("--name", required=True, help="Nombre de fichero del certificado")
    node.add_argument("--node-id", default=None, help="Identidad del nodo (por defecto un UUID nuevo)")
    node.add_argument("--out", default=None, help="Directorio de salida (por defecto el de la CA)")
    node.add_argument("--ip", action="append", default=[], help="IP adicional como SAN (repetible)")
    node.add_argument("--dns", action="append", default=[], help="Nombre DNS adicional (repetible)")
    node.add_argument("--days", type=int, default=DEFAULT_LEAF_DAYS)
    node.add_argument("--force", action="store_true")
    node.set_defaults(func=cmd_node)

    tracker = sub.add_parser("tracker-key", help="Genera el par Ed25519 que firma las rutas")
    tracker.add_argument("--out", default="certs")
    tracker.add_argument("--force", action="store_true")
    tracker.set_defaults(func=cmd_tracker_key)

    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(getattr(args, "out", None) or getattr(args, "ca_dir", "certs"), exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
