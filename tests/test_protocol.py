import asyncio
import os
import sys
import uuid
import zlib

import msgpack
import torch
from aiohttp import ClientSession

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import routing
from src.p2p_node import P2PNode
from src.tensor_utils import deserialize_tensor, serialize_tensor


def test_nested_tensor_round_trip():
    value = {
        "request_id": "metadata",
        "payload": {
            "hidden_states": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.int64),
        },
    }

    restored = deserialize_tensor(serialize_tensor(value, use_secret_sauce=False))

    assert restored["request_id"] == "metadata"
    assert torch.equal(restored["payload"]["hidden_states"], value["payload"]["hidden_states"])
    assert torch.equal(restored["payload"]["position_ids"], value["payload"]["position_ids"])


def test_unknown_dtype_is_rejected():
    packed = msgpack.packb(
        {
            "_is_tensor": True,
            "shape": [1],
            "dtype": "complex128",
            "data": b"\x00" * 16,
            "secret_sauce": False,
        },
        use_bin_type=True,
    )

    try:
        deserialize_tensor(zlib.compress(packed))
    except ValueError as exc:
        assert "dtype" in str(exc)
    else:
        raise AssertionError("An unknown dtype was accepted")


def test_mismatched_tensor_length_is_rejected():
    packed = msgpack.packb(
        {
            "_is_tensor": True,
            "shape": [4],
            "dtype": "float32",
            "data": b"\x00" * 4,
            "secret_sauce": False,
        },
        use_bin_type=True,
    )

    try:
        deserialize_tensor(packed)
    except ValueError as exc:
        assert "byte length" in str(exc)
    else:
        raise AssertionError("A malformed tensor was accepted")


def test_authenticated_local_pipeline():
    async def scenario():
        token = "test-secret"
        callback_received = asyncio.get_running_loop().create_future()
        callback = P2PNode(port=0, auth_token=token)
        callback.callback_handler = callback_received.set_result
        callback_runner = await callback.start()

        def runner_url(runner, path):
            site = next(iter(runner.sites))
            port = site._server.sockets[0].getsockname()[1]
            return f"http://127.0.0.1:{port}{path}"

        worker_b = P2PNode(
            port=0,
            auth_token=token,
            operation=lambda tensor: tensor * 2,
            next_node_url=runner_url(callback_runner, "/callback"),
        )
        worker_b_runner = await worker_b.start()
        worker_a = P2PNode(
            port=0,
            auth_token=token,
            operation=lambda tensor: tensor + 1,
            next_node_url=runner_url(worker_b_runner, "/forward"),
        )
        worker_a_runner = await worker_a.start()

        try:
            endpoint = runner_url(worker_a_runner, "/forward")
            async with ClientSession() as session:
                async with session.post(endpoint, data=serialize_tensor(torch.tensor([2.0]))) as response:
                    assert response.status == 401
                async with session.post(
                    endpoint,
                    data=serialize_tensor(torch.tensor([2.0])),
                    headers={"X-Auth-Token": token},
                ) as response:
                    assert response.status == 202

            result = await asyncio.wait_for(callback_received, timeout=5)
            assert torch.equal(result, torch.tensor([6.0]))
        finally:
            await worker_a_runner.cleanup()
            await worker_b_runner.cleanup()
            await callback_runner.cleanup()

    asyncio.run(scenario())


def _runner_url(runner, path):
    site = next(iter(runner.sites))
    port = site._server.sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}{path}"


def test_source_routed_pipeline_ignores_static_next_node():
    """El camino viaja en el sobre: los workers ya no necesitan --next-node."""

    async def scenario():
        callback_received = asyncio.get_running_loop().create_future()
        client = P2PNode(port=0)
        client.callback_handler = callback_received.set_result
        client_runner = await client.start()

        # Ninguno de los dos workers sabe quién viene después.
        worker_a = P2PNode(port=0, operation=lambda tensor: tensor + 1)
        worker_a_runner = await worker_a.start()
        worker_b = P2PNode(port=0, operation=lambda tensor: tensor * 2)
        worker_b_runner = await worker_b.start()

        route = [
            _runner_url(worker_a_runner, "/forward"),
            _runner_url(worker_b_runner, "/forward"),
            _runner_url(client_runner, "/callback"),
        ]
        envelope = routing.build_envelope(str(uuid.uuid4()), torch.tensor([2.0]), route=route, hop=0)

        try:
            async with ClientSession() as session:
                async with session.post(route[0], data=serialize_tensor(envelope)) as response:
                    assert response.status == 202

            result = await asyncio.wait_for(callback_received, timeout=5)
            assert torch.equal(result, torch.tensor([6.0]))
        finally:
            await worker_a_runner.cleanup()
            await worker_b_runner.cleanup()
            await client_runner.cleanup()

    asyncio.run(scenario())


def test_broken_pipeline_notifies_the_client_instead_of_timing_out():
    """Antes el fallo se registraba y se descartaba; el cliente comía el timeout."""

    async def scenario():
        callback_received = asyncio.get_running_loop().create_future()
        client = P2PNode(port=0)
        client.callback_handler = callback_received.set_result
        client_runner = await client.start()

        worker = P2PNode(port=0, operation=lambda tensor: tensor + 1)
        worker_runner = await worker.start()

        request_id = str(uuid.uuid4())
        route = [
            _runner_url(worker_runner, "/forward"),
            "http://127.0.0.1:1/forward",  # puerto muerto
            _runner_url(client_runner, "/callback"),
        ]
        envelope = routing.build_envelope(request_id, torch.tensor([2.0]), route=route, hop=0)

        try:
            async with ClientSession() as session:
                async with session.post(route[0], data=serialize_tensor(envelope)) as response:
                    assert response.status == 202

            result = await asyncio.wait_for(callback_received, timeout=10)
            assert isinstance(result, dict) and "error" in result
            assert result["request_id"] == request_id
            assert result["error"]["node_id"] == worker.node_id
        finally:
            await worker_runner.cleanup()
            await client_runner.cleanup()

    asyncio.run(scenario())


def test_node_rejects_unsigned_route_to_a_remote_host():
    """Una ruta arbitraria desde la red convertiría al enjambre en un amplificador."""

    async def scenario():
        worker = P2PNode(port=0, operation=lambda tensor: tensor + 1)
        worker_runner = await worker.start()
        route = ["http://198.51.100.5:9000/forward", "http://198.51.100.6:8000/callback"]
        envelope = routing.build_envelope(str(uuid.uuid4()), torch.tensor([2.0]), route=route, hop=0)
        try:
            async with ClientSession() as session:
                async with session.post(_runner_url(worker_runner, "/forward"),
                                        data=serialize_tensor(envelope)) as response:
                    assert response.status == 403
        finally:
            await worker_runner.cleanup()

    asyncio.run(scenario())


def test_cache_hits_across_requests_with_different_metadata():
    """La clave de caché depende solo del payload, no del sobre."""
    calls = []

    def operation(tensor):
        calls.append(tensor)
        return tensor * 2

    async def scenario():
        callback_received = []
        client = P2PNode(port=0)
        client.callback_handler = callback_received.append
        client_runner = await client.start()

        worker = P2PNode(port=0, operation=operation, cache_size=8)
        worker_runner = await worker.start()
        callback_url = _runner_url(client_runner, "/callback")

        try:
            async with ClientSession() as session:
                for _ in range(2):
                    # request_id distinto en cada vuelta: antes rompía la caché.
                    envelope = routing.build_envelope(
                        str(uuid.uuid4()), torch.tensor([3.0]),
                        route=[_runner_url(worker_runner, "/forward"), callback_url], hop=0,
                    )
                    async with session.post(_runner_url(worker_runner, "/forward"),
                                            data=serialize_tensor(envelope)) as response:
                        assert response.status == 202
                    await asyncio.sleep(0.2)

            assert len(calls) == 1, "el segundo envío debió servirse desde la caché"
            assert len(callback_received) == 2
        finally:
            await worker_runner.cleanup()
            await client_runner.cleanup()

    asyncio.run(scenario())
