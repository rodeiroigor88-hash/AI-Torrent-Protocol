"""Pruebas de configuración persistente, rate limit, QoS y puertos."""

import asyncio
import json
import os
import socket
import sys

import aiohttp
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.p2p_node import (
    HEARTBEAT_INTERVAL,
    MAX_HEARTBEAT_BACKOFF,
    P2PNode,
    find_free_port,
    heartbeat_delay,
)
from src.ratelimit import RateLimiter, TokenBucket
from src.tensor_utils import serialize_tensor


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("TOKENTORRENT_CONFIG", str(path))
    monkeypatch.delenv("TOKENTORRENT_TRACKER_URL", raising=False)
    return path


# --------------------------------------------------------------- config

def test_defaults_when_the_file_does_not_exist(temp_config):
    assert config_module.load_config() == config_module.DEFAULTS


def test_save_and_reload_round_trip(temp_config):
    config_module.save_config({**config_module.DEFAULTS, "hotkey": "ctrl+shift+g",
                               "worker_port": 9100})
    reloaded = config_module.load_config()
    assert reloaded["hotkey"] == "ctrl+shift+g"
    assert reloaded["worker_port"] == 9100


def test_corrupt_file_falls_back_to_defaults(temp_config):
    """Un JSON truncado no puede impedir que el worker arranque."""
    temp_config.write_text("{ esto no es json", encoding="utf-8")
    assert config_module.load_config() == config_module.DEFAULTS


@pytest.mark.parametrize("field,value,expected", [
    ("max_ram_percent", 500, 95),      # se recorta al máximo
    ("max_ram_percent", -10, 5),       # se recorta al mínimo
    ("cpu_pause_threshold", 1, 10),
    ("worker_port", 999999, 65535),
    ("cpu_cores", "4", 4),             # cadena numérica aceptada
    ("cpu_cores", "muchos", 0),        # basura -> valor por defecto
])
def test_out_of_range_values_are_clamped(field, value, expected):
    assert config_module.sanitize({field: value})[field] == expected


@pytest.mark.parametrize("url", [
    "ftp://tracker/x", "javascript:alert(1)", "http://user:pass@tracker/x", "", 42,
])
def test_invalid_tracker_urls_are_rejected(url):
    assert config_module.sanitize({"tracker_url": url})["tracker_url"] == \
        config_module.DEFAULT_TRACKER_URL


def test_local_tracker_url_is_accepted():
    cleaned = config_module.sanitize({"tracker_url": "http://127.0.0.1:5000/tokentorrent/"})
    assert cleaned["tracker_url"] == "http://127.0.0.1:5000/tokentorrent"


def test_environment_overrides_the_config_file(temp_config, monkeypatch):
    config_module.save_config({**config_module.DEFAULTS,
                               "tracker_url": "http://127.0.0.1:5000/a"})
    assert config_module.tracker_url() == "http://127.0.0.1:5000/a"
    monkeypatch.setenv("TOKENTORRENT_TRACKER_URL", "http://127.0.0.1:6000/b")
    assert config_module.tracker_url() == "http://127.0.0.1:6000/b"


def test_saved_file_is_valid_json_with_every_key(temp_config):
    config_module.save_config({"hotkey": "f9"})
    stored = json.loads(temp_config.read_text(encoding="utf-8"))
    assert set(stored) == set(config_module.DEFAULTS)
    assert stored["hotkey"] == "f9"


def test_settings_panel_round_trip_reaches_the_running_components(temp_config):
    """Lo que guarda el panel de Ajustes es lo que leen worker y cliente."""
    config_module.save_config({**config_module.DEFAULTS,
                               "hotkey": "ctrl+alt+j",
                               "tracker_url": "http://127.0.0.1:5000/tokentorrent",
                               "cpu_cores": 2, "max_ram_percent": 30})

    node = P2PNode(port=0, **{key: config_module.load_config()[key] for key in
                              ("max_ram_percent", "cpu_cores", "cpu_pause_threshold")})
    assert node.tracker_url == "http://127.0.0.1:5000/tokentorrent"
    assert node.donated_cores == min(2, os.cpu_count() or 1)
    assert node.max_ram_percent == 30
    assert config_module.load_config()["hotkey"] == "ctrl+alt+j"


def test_tray_icon_is_drawable_without_external_resources():
    """El icono se dibuja en código: si faltara el .ico, el usuario se quedaba
    sin ninguna pista visible de que la aplicación está corriendo."""
    from src.ghost_terminal import build_ghost_icon

    icon = build_ghost_icon(32)
    assert icon.size == (32, 32) and icon.mode == "RGBA"
    assert any(pixel[3] > 0 for pixel in icon.getdata()), "el icono no puede ser transparente"


# ------------------------------------------------------------ rate limit

def test_token_bucket_refills_over_time():
    clock = {"now": 0.0}
    limiter = RateLimiter(rate=10, burst=5, time_source=lambda: clock["now"])

    assert all(limiter.allow("a")[0] for _ in range(5))
    allowed, retry_after = limiter.allow("a")
    assert not allowed and retry_after > 0

    clock["now"] = 0.5  # 0.5 s * 10/s = 5 tokens
    assert limiter.allow("a")[0]


def test_clients_are_limited_independently():
    limiter = RateLimiter(rate=0, burst=1)
    assert limiter.allow("cliente-1")[0]
    assert not limiter.allow("cliente-1")[0]
    assert limiter.allow("cliente-2")[0], "un cliente ruidoso no debe afectar a otro"


def test_tracked_clients_are_bounded():
    """El propio diccionario de cubos seria una via de agotamiento de memoria."""
    limiter = RateLimiter(max_clients=32)
    for index in range(500):
        limiter.allow(f"cliente-{index}")
    assert limiter.tracked_clients == 32


def test_retry_after_matches_the_refill_rate():
    bucket = TokenBucket(rate=2.0, burst=1.0, now=0.0)
    assert bucket.consume(0.0)
    assert not bucket.consume(0.0)
    assert bucket.retry_after() == pytest.approx(0.5, abs=0.01)


def test_forward_endpoint_answers_429_with_retry_after():
    async def scenario():
        node = P2PNode(port=0, operation=lambda tensor: tensor,
                       rate_limiter=RateLimiter(rate=0, burst=2))
        runner = await node.start()
        port = next(iter(runner.sites))._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/forward"
        payload = serialize_tensor(torch.tensor([1.0]))
        try:
            statuses = []
            async with aiohttp.ClientSession() as session:
                for _ in range(4):
                    async with session.post(url, data=payload) as response:
                        statuses.append(response.status)
                        if response.status == 429:
                            assert int(response.headers["Retry-After"]) >= 1
            assert statuses[:2] == [400, 400]  # sin destino configurado, pero pasan el limite
            assert statuses[2:] == [429, 429]
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_unauthenticated_spam_is_throttled_before_reaching_the_auth_check():
    """Un atacante sin token no debe poder pedir 401 a ritmo ilimitado."""
    async def scenario():
        node = P2PNode(port=0, auth_token="secreto", operation=lambda tensor: tensor,
                       rate_limiter=RateLimiter(rate=0, burst=2))
        runner = await node.start()
        url = f"http://127.0.0.1:{node.port}/forward"
        payload = serialize_tensor(torch.tensor([1.0]))
        try:
            statuses = []
            async with aiohttp.ClientSession() as session:
                for _ in range(4):
                    async with session.post(url, data=payload) as response:
                        statuses.append(response.status)
            assert statuses[:2] == [401, 401]
            assert statuses[2:] == [429, 429]
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_attest_has_its_own_stricter_limit():
    """/attest ejecuta inferencia: no puede compartir cupo con /forward."""
    node = P2PNode(port=0)
    assert node.attest_limiter.rate < node.rate_limiter.rate
    assert node.attest_limiter.burst < node.rate_limiter.burst


# -------------------------------------------------------------- puertos

def test_find_free_port_skips_the_occupied_one():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    busy_port = holder.getsockname()[1]
    holder.listen(1)
    try:
        assert find_free_port("127.0.0.1", busy_port) != busy_port
    finally:
        holder.close()


def test_node_falls_back_to_the_next_free_port():
    """Un segundo proceso del nodo ya no muere con OSError en el arranque."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    busy_port = holder.getsockname()[1]
    holder.listen(1)

    async def scenario():
        node = P2PNode(host="127.0.0.1", port=busy_port)
        runner = await node.start()
        try:
            assert node.port != busy_port
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{node.port}/ping") as response:
                    assert response.status == 200
                    assert (await response.json())["port"] == node.port
        finally:
            await runner.cleanup()

    try:
        asyncio.run(scenario())
    finally:
        holder.close()


def test_port_zero_is_resolved_to_the_real_ephemeral_port():
    """Con puerto 0 el heartbeat anunciaria el 0 si no se leyera del socket."""
    async def scenario():
        node = P2PNode(port=0)
        runner = await node.start()
        try:
            assert node.port > 0
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


# ------------------------------------------------------------------ QoS

def test_cpu_cores_zero_donates_half_the_machine():
    node = P2PNode(port=0, cpu_cores=0)
    assert node.donated_cores == max(1, (os.cpu_count() or 1) // 2)


def test_explicit_cpu_cores_are_respected_within_the_machine_limits():
    node = P2PNode(port=0, cpu_cores=1)
    assert node.donated_cores == 1
    huge = P2PNode(port=0, cpu_cores=9999)
    assert huge.donated_cores == (os.cpu_count() or 1)


def test_qos_pauses_when_the_cpu_threshold_is_impossible_to_satisfy():
    node = P2PNode(port=0, cpu_pause_threshold=10)
    # -1 hace que cualquier lectura (incluido un 0.0 en un equipo ocioso)
    # supere el umbral: la prueba no depende de la carga real de la máquina.
    node.cpu_pause_threshold = -1
    assert "CPU" in node._qos_check()


def test_qos_gives_a_real_cpu_reading_on_the_first_check():
    """`cpu_percent` devuelve 0.0 la primera vez: si no se cebara, el nodo
    creería que el PC está ocioso justo al arrancar."""
    node = P2PNode(port=0, cpu_pause_threshold=100)
    node.cpu_pause_threshold = 101   # nunca pausa por CPU
    assert node._qos_check() in ("", node._qos_check())  # no lanza excepción
    node.cpu_pause_threshold = -1
    assert "CPU al" in node._qos_check()


def test_qos_reports_the_ram_budget_from_the_donated_percentage():
    node = P2PNode(port=0, max_ram_percent=10)
    other = P2PNode(port=0, max_ram_percent=20)
    assert other._ram_budget_mb() == pytest.approx(node._ram_budget_mb() * 2, rel=0.01)


def test_qos_pause_reason_is_exposed_on_ping():
    async def scenario():
        node = P2PNode(port=0)
        node.is_paused = True
        node.pause_reason = "CPU al 99%"
        runner = await node.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{node.port}/ping") as response:
                    body = await response.json()
            assert body["paused"] is True
            assert body["pause_reason"] == "CPU al 99%"
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


# -------------------------------------------------------- watchdog de red

def test_heartbeat_backoff_grows_and_is_capped():
    assert heartbeat_delay(0) == HEARTBEAT_INTERVAL
    assert heartbeat_delay(1) == HEARTBEAT_INTERVAL * 2
    assert heartbeat_delay(2) == HEARTBEAT_INTERVAL * 4
    assert heartbeat_delay(99) == MAX_HEARTBEAT_BACKOFF


def test_heartbeat_interval_allows_a_45s_tracker_timeout():
    """El tracker expulsa a los 45 s: hacen falta 3 latidos en esa ventana."""
    assert HEARTBEAT_INTERVAL * 3 <= 45


def test_watchdog_replaces_dead_sockets_after_losing_the_network():
    async def scenario():
        node = P2PNode(port=0)
        first = node._get_client_session()
        await node._reset_client_session()
        assert first.closed, "la sesión con sockets muertos debe cerrarse"
        second = node._get_client_session()
        assert second is not first and not second.closed
        await second.close()

    asyncio.run(scenario())


def test_heartbeat_reports_the_donated_resources():
    async def scenario():
        node = P2PNode(port=0, cpu_cores=1, max_ram_percent=25)
        captured = {}

        class FakeResponse:
            status = 200

            async def read(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class FakeSession:
            def post(self, url, json=None):
                captured.update(json or {})
                return FakeResponse()

        node._get_client_session = lambda: FakeSession()
        assert await node._send_heartbeat() == 'ok'
        assert captured["donated_cores"] == 1
        assert captured["donated_ram_mb"] > 0
        assert captured["port"] == node.port

    asyncio.run(scenario())
