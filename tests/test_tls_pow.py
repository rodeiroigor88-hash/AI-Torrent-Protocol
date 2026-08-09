"""Pruebas de mTLS y del reto de atestación (Proof of Compute)."""

import asyncio
import os
import ssl
import sys
import uuid

import aiohttp
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.p2p_node import P2PNode
from src.pow_utils import (
    DEFAULT_EPSILON,
    attestation_message,
    canonical_challenge,
    deterministic_inference,
    outputs_match,
    relative_l2,
    tensor_digest,
)
from src.tensor_utils import deserialize_tensor, serialize_tensor
from src.tls_utils import (
    build_client_ssl_context,
    cert_node_id,
    generate_ca,
    generate_node_cert,
    load_cert,
    node_sni,
    verify_blob,
    write_pem,
)

HIDDEN_SIZE = 16


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    """CA del enjambre + certificados para nodo y cliente, y una CA intrusa."""
    directory = tmp_path_factory.mktemp("pki")
    ca_key, ca_cert = generate_ca()
    write_pem(str(directory / "ca.key"), key=ca_key)
    write_pem(str(directory / "ca.crt"), cert=ca_cert)

    identities = {}
    for name in ("worker", "client"):
        node_id = str(uuid.uuid4())
        key, cert = generate_node_cert(node_id, ca_key, ca_cert, extra_ips=["127.0.0.1"])
        write_pem(str(directory / f"{name}.key"), key=key)
        write_pem(str(directory / f"{name}.crt"), cert=cert)
        identities[name] = node_id

    rogue_key, rogue_cert = generate_ca(common_name="CA Intrusa")
    rogue_node_id = str(uuid.uuid4())
    key, cert = generate_node_cert(rogue_node_id, rogue_key, rogue_cert)
    write_pem(str(directory / "rogue-ca.crt"), cert=rogue_cert)
    write_pem(str(directory / "rogue.key"), key=key)
    write_pem(str(directory / "rogue.crt"), cert=cert)

    return {"dir": directory, "identities": identities}


def _paths(pki, name):
    directory = pki["dir"]
    return str(directory / f"{name}.crt"), str(directory / f"{name}.key"), str(directory / "ca.crt")


def _site_port(runner):
    site = next(iter(runner.sites))
    return site._server.sockets[0].getsockname()[1]


def test_certificate_carries_the_node_identity(pki):
    certificate = load_cert(str(pki["dir"] / "worker.crt"))
    assert cert_node_id(certificate) == pki["identities"]["worker"]


def test_mtls_pipeline_rejects_outsiders_and_accepts_swarm_members(pki):
    """El certificado sustituye al token: la CA decide quién entra."""
    cert, key, ca = _paths(pki, "worker")
    client_cert, client_key, _ = _paths(pki, "client")
    worker_id = pki["identities"]["worker"]

    async def scenario():
        received = asyncio.get_running_loop().create_future()
        node = P2PNode(port=0, operation=lambda tensor: tensor * 3,
                       tls_cert=cert, tls_key=key, tls_ca=ca)
        node.callback_handler = received.set_result
        runner = await node.start()
        port = _site_port(runner)
        url = f"https://127.0.0.1:{port}/callback"

        try:
            # 1. Sin certificado de cliente: el handshake mTLS debe fallar.
            plain_context = build_client_ssl_context(ca)
            connector = aiohttp.TCPConnector(ssl=plain_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                with pytest.raises((ssl.SSLError, aiohttp.ClientError, OSError)):
                    await session.post(url, data=serialize_tensor(torch.tensor([1.0])),
                                       server_hostname=node_sni(worker_id))

            # 2. Con certificado emitido por otra CA: rechazado igualmente.
            rogue_context = build_client_ssl_context(
                ca, str(pki["dir"] / "rogue.crt"), str(pki["dir"] / "rogue.key")
            )
            connector = aiohttp.TCPConnector(ssl=rogue_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                with pytest.raises(Exception):
                    await session.post(url, data=serialize_tensor(torch.tensor([1.0])),
                                       server_hostname=node_sni(worker_id))

            # 3. Miembro legítimo del enjambre: aceptado sin token compartido.
            member_context = build_client_ssl_context(ca, client_cert, client_key)
            connector = aiohttp.TCPConnector(ssl=member_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                envelope = {"request_id": str(uuid.uuid4()),
                            "payload": torch.tensor([2.0])}
                async with session.post(url, data=serialize_tensor(envelope),
                                        server_hostname=node_sni(worker_id)) as response:
                    assert response.status == 200

            result = await asyncio.wait_for(received, timeout=5)
            assert torch.equal(result, torch.tensor([2.0]))
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_wrong_sni_fails_hostname_verification(pki):
    """La identidad va en el SAN, no en la IP: un node_id que no es el suyo falla."""
    cert, key, ca = _paths(pki, "worker")
    client_cert, client_key, _ = _paths(pki, "client")

    async def scenario():
        node = P2PNode(port=0, tls_cert=cert, tls_key=key, tls_ca=ca)
        runner = await node.start()
        port = _site_port(runner)
        try:
            context = build_client_ssl_context(ca, client_cert, client_key)
            connector = aiohttp.TCPConnector(ssl=context)
            async with aiohttp.ClientSession(connector=connector) as session:
                with pytest.raises(Exception):
                    await session.get(f"https://127.0.0.1:{port}/ping",
                                      server_hostname=node_sni(str(uuid.uuid4())))
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_attestation_is_signed_and_detects_a_lying_node(pki):
    """Un nodo honesto reproduce la referencia; uno que devuelve ruido, no."""
    cert, key, ca = _paths(pki, "worker")
    client_cert, client_key, _ = _paths(pki, "client")
    worker_id = pki["identities"]["worker"]
    seed = "semilla-de-prueba"

    def honest(payload):
        return payload["hidden_states"] * 2.0

    def liar(payload):
        return torch.randn_like(payload["hidden_states"])

    challenge = canonical_challenge(seed, HIDDEN_SIZE, 4, "llama/qwen")
    with deterministic_inference():
        reference = honest(challenge)

    async def challenge_node(node):
        runner = await node.start()
        port = _site_port(runner)
        try:
            context = build_client_ssl_context(ca, client_cert, client_key)
            connector = aiohttp.TCPConnector(ssl=context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"https://127.0.0.1:{port}/attest",
                    json={"seed": seed, "seq_len": 4},
                    server_hostname=node_sni(node.node_id),
                ) as response:
                    assert response.status == 200
                    return deserialize_tensor(await response.read())
        finally:
            await runner.cleanup()

    async def scenario():
        honest_node = P2PNode(port=0, operation=honest, tls_cert=cert, tls_key=key,
                              tls_ca=ca, model_arch="llama/qwen", hidden_size=HIDDEN_SIZE)
        response = await challenge_node(honest_node)

        assert response["node_id"] == worker_id
        digest = tensor_digest(response["output"])
        assert digest == response["digest"]
        certificate = load_cert(str(pki["dir"] / "worker.crt"))
        assert verify_blob(certificate,
                           attestation_message(worker_id, seed, digest),
                           response["signature"])
        assert outputs_match(response["output"], reference, DEFAULT_EPSILON)

        # La firma queda ligada al node_id y a la semilla: no es reutilizable.
        assert not verify_blob(certificate,
                               attestation_message(worker_id, "otra-semilla", digest),
                               response["signature"])

        lying_node = P2PNode(port=0, operation=liar, tls_cert=cert, tls_key=key,
                             tls_ca=ca, model_arch="llama/qwen", hidden_size=HIDDEN_SIZE)
        bad_response = await challenge_node(lying_node)
        assert not outputs_match(bad_response["output"], reference, DEFAULT_EPSILON)
        assert relative_l2(bad_response["output"], reference) > DEFAULT_EPSILON

    asyncio.run(scenario())


def test_full_stack_mtls_pipeline_with_a_tracker_signed_route(pki):
    """Integración de las tres piezas: mTLS + SNI + ruta firmada multi-salto."""
    from cryptography.hazmat.primitives import serialization

    from src import routing

    cert, key, ca = _paths(pki, "worker")
    directory = pki["dir"]

    private_key, public_key = routing.generate_tracker_keypair()
    tracker_pub_path = str(directory / "tracker.pub")
    with open(tracker_pub_path, "wb") as handle:
        handle.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ))

    async def scenario():
        received = asyncio.get_running_loop().create_future()
        nodes = []
        runners = []
        try:
            for operation in (lambda tensor: tensor + 1, lambda tensor: tensor * 2, None):
                node = P2PNode(port=0, operation=operation, tls_cert=cert, tls_key=key,
                               tls_ca=ca, tracker_key=tracker_pub_path)
                nodes.append(node)
                runners.append(await node.start())
            nodes[-1].callback_handler = received.set_result

            route = routing.validate_route([
                {"url": f"https://127.0.0.1:{_site_port(runners[0])}/forward",
                 "node_id": nodes[0].node_id},
                {"url": f"https://127.0.0.1:{_site_port(runners[1])}/forward",
                 "node_id": nodes[1].node_id},
                {"url": f"https://127.0.0.1:{_site_port(runners[2])}/callback",
                 "node_id": nodes[2].node_id},
            ])
            expires_at = int(__import__("time").time()) + 300
            signature = routing.sign_route(private_key, route, expires_at)
            envelope = routing.build_envelope(
                str(uuid.uuid4()), torch.tensor([2.0]), route=route, hop=0,
                expires_at=expires_at, signature=signature,
            )

            client_cert, client_key, _ = _paths(pki, "client")
            context = build_client_ssl_context(ca, client_cert, client_key)
            connector = aiohttp.TCPConnector(ssl=context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(routing.hop_url(route[0]),
                                        data=serialize_tensor(envelope),
                                        server_hostname=node_sni(nodes[0].node_id)) as response:
                    assert response.status == 202

            result = await asyncio.wait_for(received, timeout=10)
            assert torch.equal(result, torch.tensor([6.0]))
        finally:
            for runner in runners:
                await runner.cleanup()

    asyncio.run(scenario())


def test_attestation_challenge_is_deterministic():
    first = canonical_challenge("abc", HIDDEN_SIZE, 4)
    second = canonical_challenge("abc", HIDDEN_SIZE, 4)
    third = canonical_challenge("abd", HIDDEN_SIZE, 4)
    assert torch.equal(first["hidden_states"], second["hidden_states"])
    assert not torch.equal(first["hidden_states"], third["hidden_states"])


def test_skipping_one_layer_is_far_beyond_the_tolerance():
    """Justifica el umbral: saltarse trabajo se desvía órdenes de magnitud."""
    torch.manual_seed(0)
    hidden = torch.randn(1, 4, HIDDEN_SIZE)
    layer = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    with torch.no_grad():
        full = layer(layer(hidden))
        skipped = layer(hidden)
    assert relative_l2(skipped, full) > DEFAULT_EPSILON

    # Ruido de precisión float16 (el "secret sauce"): dentro de tolerancia.
    downcast = full.to(torch.float16).to(torch.float32)
    assert outputs_match(downcast, full, DEFAULT_EPSILON)
