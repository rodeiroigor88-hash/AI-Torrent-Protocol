"""Pruebas del enrutamiento en origen y de la política de aceptación de rutas."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import routing
from src.routing import RouteError
from src.tensor_utils import MAX_LIST_ITEMS, deserialize_tensor, serialize_tensor

import torch


def test_normalize_endpoint_adds_forward_suffix():
    assert routing.normalize_endpoint("http://127.0.0.1:8001") == "http://127.0.0.1:8001/forward"
    assert routing.normalize_endpoint("http://127.0.0.1:8001/callback") == "http://127.0.0.1:8001/callback"


@pytest.mark.parametrize("url", [
    "ftp://127.0.0.1/forward",
    "file:///etc/passwd",
    "http://user:pass@127.0.0.1/forward",
    "not-a-url",
    "",
])
def test_normalize_endpoint_rejects_bad_urls(url):
    with pytest.raises(RouteError):
        routing.normalize_endpoint(url)


def test_route_longer_than_limit_is_rejected():
    route = [f"http://127.0.0.1:{9000 + i}/forward" for i in range(routing.MAX_ROUTE_HOPS + 1)]
    with pytest.raises(RouteError):
        routing.validate_route(route)


def test_unsigned_remote_route_is_rejected():
    """Una ruta sin firmar hacia destinos remotos es SSRF: debe rechazarse."""
    route = ["http://198.51.100.5:9000/forward", "http://127.0.0.1:8000/callback"]
    with pytest.raises(RouteError):
        routing.authorize_route(route, None, None)


def test_unsigned_loopback_route_is_allowed():
    route = ["http://127.0.0.1:9000/forward", "http://127.0.0.1:8000/callback"]
    assert routing.authorize_route(route, None, None)


def test_unsigned_remote_route_with_opt_in_is_allowed():
    route = ["http://198.51.100.5:9000/forward", "http://198.51.100.6:8000/callback"]
    assert routing.authorize_route(route, None, None, allow_unsigned=True)


def test_signed_remote_route_is_accepted_and_tampering_is_not():
    private_key, public_key = routing.generate_tracker_keypair()
    route = routing.validate_route([
        "http://198.51.100.5:9000/forward",
        "http://198.51.100.6:8000/callback",
    ])
    expires_at = int(time.time()) + 300
    signature = routing.sign_route(private_key, route, expires_at)

    assert routing.authorize_route(route, expires_at, signature, tracker_public_key=public_key)

    tampered = ["http://198.51.100.99:9000/forward", "http://198.51.100.6:8000/callback"]
    with pytest.raises(RouteError):
        routing.authorize_route(tampered, expires_at, signature, tracker_public_key=public_key)


def test_expired_signed_route_is_rejected():
    private_key, public_key = routing.generate_tracker_keypair()
    route = routing.validate_route(["http://198.51.100.5:9000/forward",
                                    "http://198.51.100.6:8000/callback"])
    expires_at = int(time.time()) - 1
    signature = routing.sign_route(private_key, route, expires_at)
    with pytest.raises(RouteError):
        routing.authorize_route(route, expires_at, signature, tracker_public_key=public_key)


def test_next_hop_walks_the_route_and_stops_at_the_end():
    route = routing.validate_route([
        "http://127.0.0.1:9001/forward",
        "http://127.0.0.1:9002/forward",
        "http://127.0.0.1:8000/callback",
    ])
    assert routing.next_hop(route, 0)[0] == "http://127.0.0.1:9002/forward"
    assert routing.next_hop(route, 1)[0] == "http://127.0.0.1:8000/callback"
    with pytest.raises(RouteError):
        routing.next_hop(route, 2)


def test_route_survives_the_wire_format():
    """El sobre v2 lleva listas, que el formato v1 no admitía."""
    envelope = routing.build_envelope(
        "5f6c1f1a-0000-4000-8000-000000000000",
        {"hidden_states": torch.ones(1, 2, 4)},
        route=[{"url": "http://127.0.0.1:9001/forward", "node_id": "abc"},
               "http://127.0.0.1:8000/callback"],
        hop=0,
    )
    restored = deserialize_tensor(serialize_tensor(envelope, use_secret_sauce=False))
    assert restored["route"][0]["node_id"] == "abc"
    assert restored["route"][1] == "http://127.0.0.1:8000/callback"
    assert torch.equal(restored["payload"]["hidden_states"], torch.ones(1, 2, 4))


def test_oversized_list_is_rejected_by_the_serializer():
    with pytest.raises(ValueError):
        serialize_tensor({"route": ["x"] * (MAX_LIST_ITEMS + 1)}, use_secret_sauce=False)
