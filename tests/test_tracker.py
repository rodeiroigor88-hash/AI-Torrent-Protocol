"""Pruebas del tracker de referencia: nodos fantasma, planificación y rate limit.

Se usa `TestServer`/`TestClient` dentro de `asyncio.run`, como el resto de la
suite, para no añadir la dependencia `pytest-aiohttp`.
"""

import asyncio
import os
import sys
import time
import uuid

import pytest
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import routing
from src import tracker as tracker_module
from src.tracker import NODE_TIMEOUT, Tracker


class FakeClock:
    """Reloj controlado: probar un timeout de 45 s no puede tardar 45 s."""

    def __init__(self, now=None):
        # Arranca en la hora real: las rutas firmadas llevan una expiración
        # absoluta que el nodo compara contra `time.time()`.
        self.now = time.time() if now is None else now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


CALLBACK = "http://127.0.0.1:8000/callback"


def heartbeat(node_id, layers, is_last=False, port=8001, arch="llama/qwen", **extra):
    payload = {"node_id": node_id, "layers": layers, "port": port, "is_last": is_last,
               "model_arch": arch, "ip": "127.0.0.1", "scheme": "http"}
    payload.update(extra)
    return payload


def run_tracker_test(scenario, signing_key=None, node_timeout=NODE_TIMEOUT,
                     allow_private_callbacks=True):
    """Levanta el tracker y ejecuta `scenario(client, tracker, clock)`.

    Por defecto se permiten callbacks a loopback: los escenarios existentes
    usan `http://127.0.0.1:...` para poder correr sin red. Los tests
    especificos de la proteccion SSRF crean el tracker con
    ``allow_private_callbacks=False`` a mano.
    """
    clock = FakeClock()
    tracker = Tracker(signing_key=signing_key, node_timeout=node_timeout, time_source=clock,
                      allow_private_callbacks=allow_private_callbacks)

    async def runner():
        server = TestServer(tracker.app)
        client = TestClient(server)
        await client.start_server()
        try:
            await scenario(client, tracker, clock)
        finally:
            await client.close()

    asyncio.run(runner())


async def register(client, **kwargs):
    response = await client.post("/register", json=heartbeat(**kwargs))
    assert response.status == 200, await response.text()
    return response


# ------------------------------------------------------- nodos fantasma

def test_ghost_nodes_are_evicted_after_the_timeout():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15")
        assert len(tracker.active_nodes()) == 1

        clock.advance(NODE_TIMEOUT - 1)
        assert len(tracker.active_nodes()) == 1, "todavía dentro de la ventana de latido"

        clock.advance(2)
        assert tracker.active_nodes() == [], "un PC apagado de golpe debe desaparecer"

    run_tracker_test(scenario)


def test_a_fresh_heartbeat_keeps_the_node_alive():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15")
        for _ in range(4):
            clock.advance(15)   # el ritmo real de latido del nodo
            await register(client, node_id="a", layers="8-15")
        assert len(tracker.active_nodes()) == 1

    run_tracker_test(scenario)


def test_paused_nodes_are_not_offered():
    async def scenario(client, tracker, clock):
        await client.post("/register", json=heartbeat(node_id="a", layers="8-15", paused=True))
        assert tracker.active_nodes() == []

    run_tracker_test(scenario)


def test_register_rejects_malformed_payloads():
    async def scenario(client, tracker, clock):
        for payload in (
            {"node_id": "", "layers": "8-15", "port": 8001},
            {"node_id": "a", "layers": "no-es-un-rango", "port": 8001},
            {"node_id": "a", "layers": "15-8", "port": 8001},
            {"node_id": "a", "layers": "8-15", "port": 0},
            {"node_id": "a", "layers": "8-15", "port": "8001"},
        ):
            response = await client.post("/register", json=payload)
            assert response.status == 400, payload

    run_tracker_test(scenario)


# ---------------------------------------------------------- planificación

def test_plan_chains_contiguous_layer_ranges():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        await register(client, node_id="b", layers="16-23", port=8002, is_last=True)

        response = await client.get("/plan", params={"model_arch": "llama/qwen", "start_layer": "8",
                                                     "callback": CALLBACK})
        assert response.status == 200
        data = await response.json()
        assert [routing.hop_node_id(hop) for hop in data["route"][:-1]] == ["a", "b"]
        assert routing.hop_url(data["route"][-1]) == CALLBACK
        assert routing.hop_url(data["route"][0]) == "http://127.0.0.1:8001/forward"

    run_tracker_test(scenario)


def test_plan_refuses_an_incomplete_chain():
    """Sin nodo final no hay logits: mejor 404 que un callback que nunca llega."""
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        assert (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).status == 404

        await register(client, node_id="b", layers="16-23", port=8002, is_last=True)
        assert (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).status == 200

    run_tracker_test(scenario)


def test_plan_detects_a_gap_in_the_layer_coverage():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        await register(client, node_id="c", layers="20-23", port=8003, is_last=True)
        assert (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).status == 404

    run_tracker_test(scenario)


def test_plan_honours_the_exclusion_list():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        await register(client, node_id="a2", layers="8-15", port=8011)
        await register(client, node_id="b", layers="16-23", port=8002, is_last=True)

        data = await (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK,
                                                        "exclude": "a"})).json()
        assert [routing.hop_node_id(hop) for hop in data["route"][:-1]] == ["a2", "b"]

    run_tracker_test(scenario)


def test_plan_does_not_route_through_an_expired_node():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        await register(client, node_id="b", layers="16-23", port=8002, is_last=True)
        clock.advance(NODE_TIMEOUT + 1)
        assert (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).status == 404

    run_tracker_test(scenario)


def test_planned_route_is_signed_and_accepted_by_a_node():
    private_key, public_key = routing.generate_tracker_keypair()

    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        await register(client, node_id="b", layers="16-23", port=8002, is_last=True)

        data = await (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).json()
        signature = bytes.fromhex(data["signature"])

        # La firma debe valer exactamente para lo que el nodo verifica al
        # recibir el sobre, incluida la normalización de la ruta.
        assert routing.authorize_route(data["route"], data["exp"], signature,
                                       tracker_public_key=public_key)

    run_tracker_test(scenario, signing_key=private_key)


def test_plan_requires_the_callback_so_the_signature_covers_it():
    """Si el cliente pudiera añadir el destino final después de la firma, el
    último worker entregaría a cualquier víctima que ese cliente eligiera."""
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001, is_last=True)
        assert (await client.get("/plan", params={"start_layer": "8"})).status == 400

        for bad in ("ftp://x/callback", "http://user:pass@x/callback", "no-es-url"):
            response = await client.get("/plan", params={"start_layer": "8", "callback": bad})
            assert response.status == 400, bad

    run_tracker_test(scenario)


def test_unsigned_tracker_returns_a_null_signature():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001, is_last=True)
        data = await (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).json()
        assert data["signature"] is None

    run_tracker_test(scenario)


def test_reported_nodes_are_dropped_from_future_routes():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        await register(client, node_id="a2", layers="8-15", port=8011)
        await register(client, node_id="b", layers="16-23", port=8002, is_last=True)

        first = await (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).json()
        chosen = routing.hop_node_id(first['route'][0])

        response = await client.post("/report", json={"nodes": [chosen], "divergence": 3.1})
        assert response.status == 200

        second = await (await client.get("/plan", params={"start_layer": "8", "callback": CALLBACK})).json()
        assert routing.hop_node_id(second['route'][0]) != chosen

    run_tracker_test(scenario)


def test_report_rejects_an_oversized_node_list():
    async def scenario(client, tracker, clock):
        response = await client.post("/report", json={"nodes": ["x"] * 100})
        assert response.status == 400

    run_tracker_test(scenario)


# ------------------------------------------------------------ rate limit

def test_register_spam_is_throttled():
    async def scenario(client, tracker, clock):
        statuses = []
        for index in range(30):
            response = await client.post("/register",
                                         json=heartbeat(node_id=f"n{index}", layers="8-15"))
            statuses.append(response.status)

        assert 429 in statuses, "un nodo malicioso no debe poder inundar el tracker"
        assert statuses.index(429) >= 10, "la ráfaga normal de arranque no debe cortarse"

    run_tracker_test(scenario)


def test_throttled_response_tells_the_client_when_to_retry():
    async def scenario(client, tracker, clock):
        tracker.plan_limiter.rate = 0
        tracker.plan_limiter.burst = 1
        tracker.plan_limiter.reset()

        assert (await client.get("/plan", params={"callback": CALLBACK})).status in (200, 404)
        response = await client.get("/plan", params={"callback": CALLBACK})
        assert response.status == 429
        assert int(response.headers["Retry-After"]) >= 1

    run_tracker_test(scenario)


def test_registry_size_is_bounded(monkeypatch):
    """El propio registro de nodos no puede crecer sin límite."""
    monkeypatch.setattr(tracker_module, "MAX_NODES", 3)

    async def scenario(client, tracker, clock):
        tracker.register_limiter.rate = 1e9   # aquí se prueba el tope del registro
        tracker.register_limiter.burst = 1e9
        tracker.register_limiter.reset()

        statuses = [(await client.post("/register",
                                       json=heartbeat(node_id=f"n{i}", layers="8-15"))).status
                    for i in range(5)]
        assert statuses.count(200) == 3
        assert statuses[-1] == 503

    run_tracker_test(scenario)


def test_status_endpoint_summarizes_the_swarm():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001)
        await register(client, node_id="b", layers="16-23", port=8002, is_last=True)

        body = await (await client.get("/status")).json()
        assert [node["node_id"] for node in body["nodes"]] == ["a", "b"]
        assert body["node_timeout"] == NODE_TIMEOUT

    run_tracker_test(scenario)


def test_node_timeout_matches_three_missed_heartbeats():
    from src.p2p_node import HEARTBEAT_INTERVAL
    assert NODE_TIMEOUT == pytest.approx(HEARTBEAT_INTERVAL * 3)


# --------------------------------------------------------------- SSRF y limites

def test_public_tracker_refuses_callback_to_private_ranges():
    """Un tracker publico no debe firmar un callback contra rangos internos:
    ese es el vector para convertir al enjambre en un amplificador SSRF."""
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001, is_last=True)
        for host in (
            "127.0.0.1",              # loopback
            "10.0.0.5",               # RFC1918
            "192.168.1.10",           # RFC1918
            "172.16.0.9",             # RFC1918
            "169.254.169.254",        # metadatos de nube (AWS/GCP/Azure)
            "localhost",              # nombre reservado
            "0.0.0.0",                # no especificado
            "::1",                    # loopback IPv6
            "fe80::1",                # link-local IPv6
        ):
            callback = f"http://{host}:8000/callback"
            response = await client.get("/plan",
                                        params={"start_layer": "8", "callback": callback})
            assert response.status == 400, f"{host} deberia rechazarse"

    run_tracker_test(scenario, allow_private_callbacks=False)


def test_public_tracker_accepts_callback_to_public_hosts():
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001, is_last=True)
        # IP publica de verdad y nombre de dominio: aceptados sin resolver DNS.
        # (Los bloques TEST-NET RFC5737 caen bajo is_private en Python, asi que
        # no sirven como ejemplo de "publico".)
        for callback in ("http://8.8.8.8:8000/callback",
                         "https://api.tokentorrent.example/callback"):
            response = await client.get("/plan",
                                        params={"start_layer": "8", "callback": callback})
            assert response.status == 200, callback

    run_tracker_test(scenario, allow_private_callbacks=False)


def test_local_tracker_can_opt_in_to_private_callbacks():
    """Un tracker local de pruebas necesita callbacks loopback: la salida
    de emergencia debe ser explicita, no por defecto."""
    async def scenario(client, tracker, clock):
        await register(client, node_id="a", layers="8-15", port=8001, is_last=True)
        assert (await client.get("/plan", params={
            "start_layer": "8", "callback": CALLBACK})).status == 200

    run_tracker_test(scenario, allow_private_callbacks=True)


def test_register_rejects_oversized_strings():
    """Un nodo malicioso no debe poder llenar el registro con strings gigantes."""
    async def scenario(client, tracker, clock):
        base = {"node_id": "a", "layers": "8-15", "port": 8001}
        for payload in (
            {**base, "model_arch": "x" * 200},
            {**base, "model_name": "x" * 500},
            {**base, "layers": "8-" + "9" * 40},
        ):
            response = await client.post("/register", json=payload)
            assert response.status == 400, payload

    run_tracker_test(scenario)


def test_register_rejects_oversized_body():
    """El default de aiohttp (1 MiB) es un vector de agotamiento de memoria
    innecesario cuando los payloads legitimos son <4 KB."""
    async def scenario(client, tracker, clock):
        # Un body de ~64 KiB supera holgadamente MAX_REQUEST_BYTES=32 KiB.
        payload = {"node_id": "a", "layers": "8-15", "port": 8001,
                   "model_arch": "llama/qwen", "padding": "x" * (64 * 1024)}
        response = await client.post("/register", json=payload)
        assert response.status == 413

    run_tracker_test(scenario)


# ------------------------------------------------------------ integración

def test_end_to_end_swarm_driven_by_the_tracker(tmp_path):
    """Cadena completa: latido -> /plan firmado -> pipeline -> callback.

    Es la prueba que ata las tres piezas: el tracker planifica y firma, los
    nodos verifican esa firma antes de reenviar, y el resultado vuelve al
    cliente sin que ningún worker sepa quién viene después.
    """
    import aiohttp
    import torch
    from cryptography.hazmat.primitives import serialization

    from src.p2p_node import P2PNode
    from src.tensor_utils import serialize_tensor

    private_key, public_key = routing.generate_tracker_keypair()
    key_path = tmp_path / "tracker.pub"
    key_path.write_bytes(public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ))

    async def scenario(client, tracker, clock):
        received = asyncio.get_running_loop().create_future()
        worker_a = P2PNode(port=0, operation=lambda tensor: tensor + 1,
                           tracker_key=str(key_path))
        worker_b = P2PNode(port=0, operation=lambda tensor: tensor * 2,
                           tracker_key=str(key_path))
        callback_node = P2PNode(port=0, tracker_key=str(key_path))
        callback_node.callback_handler = received.set_result

        runners = [await node.start() for node in (worker_a, worker_b, callback_node)]
        try:
            await register(client, node_id=worker_a.node_id, layers="8-15",
                           port=worker_a.port)
            await register(client, node_id=worker_b.node_id, layers="16-23",
                           port=worker_b.port, is_last=True)

            # El cliente pide la ruta YA CERRADA con su callback: el tracker
            # firma el camino completo, incluido el destino final.
            plan = await (await client.get("/plan", params={
                "model_arch": "llama/qwen", "start_layer": "8",
                "callback": f"http://127.0.0.1:{callback_node.port}/callback",
            })).json()
            route = plan["route"]
            signature = bytes.fromhex(plan["signature"])
            assert routing.hop_url(route[-1]).endswith("/callback")

            envelope = routing.build_envelope(
                str(uuid.uuid4()), torch.tensor([2.0]),
                route=route, hop=0, expires_at=plan["exp"], signature=signature,
            )
            async with aiohttp.ClientSession() as session:
                async with session.post(routing.hop_url(route[0]),
                                        data=serialize_tensor(envelope)) as response:
                    assert response.status == 202

            result = await asyncio.wait_for(received, timeout=10)
            assert torch.equal(result, torch.tensor([6.0]))

            # Y una ruta manipulada tras la firma no pasa el control del nodo.
            tampered = list(route)
            tampered[-1] = "http://198.51.100.7:9999/callback"
            bad_envelope = routing.build_envelope(
                str(uuid.uuid4()), torch.tensor([2.0]),
                route=tampered, hop=0, expires_at=plan["exp"], signature=signature,
            )
            async with aiohttp.ClientSession() as session:
                async with session.post(routing.hop_url(route[0]),
                                        data=serialize_tensor(bad_envelope)) as response:
                    assert response.status == 403
        finally:
            for runner in runners:
                await runner.cleanup()

    run_tracker_test(scenario, signing_key=private_key)
