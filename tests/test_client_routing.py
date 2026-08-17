"""Pruebas de la lógica de ruta del cliente sin descargar el modelo.

`AgenticChat.__init__` carga el tokenizer de HuggingFace, así que aquí se
construye la instancia sin pasar por el constructor y se rellenan solo los
atributos que participan en el enrutamiento.
"""

import asyncio
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import chat_agent as chat_agent_module
from src import routing
from src.chat_agent import MAX_ROUTE_ATTEMPTS, AgenticChat, PipelineError
from src.crossverify import AdaptiveSampler


def make_client(route=None, next_node_url=None, tls_cert=None):
    chat = object.__new__(AgenticChat)
    chat.route = list(route) if route else None
    chat.route_exp = None
    chat.route_sig = None
    chat.primary_node_url = next_node_url
    chat._static_node_url = next_node_url
    chat.callback_url = "http://127.0.0.1:8000/callback"
    chat.node_id = "cliente-1"
    chat.tls_cert = tls_cert
    chat.failed_nodes = set()
    chat.pending_requests = {}
    chat.auth_token = None
    chat.client_model = types.SimpleNamespace(arch="llama/qwen")
    chat.audit_probability = 0.0
    chat.session = None
    return chat


def test_full_route_appends_the_client_callback():
    chat = make_client(route=["http://127.0.0.1:9001/forward"])
    full = chat._full_route()
    assert full[-1] == "http://127.0.0.1:8000/callback"
    assert len(full) == 2


def test_full_route_does_not_duplicate_a_callback_already_planned():
    """La ruta del tracker ya viene cerrada y FIRMADA con el callback dentro:
    añadirlo otra vez rompería la firma y dejaría un salto muerto."""
    chat = make_client(route=["http://127.0.0.1:9001/forward",
                              "http://127.0.0.1:8000/callback"])
    full = chat._full_route()
    assert len(full) == 2
    assert full[-1] == "http://127.0.0.1:8000/callback"


def test_full_route_carries_the_client_identity_under_tls():
    chat = make_client(route=["https://10.0.0.5:9001/forward"], tls_cert="cert.pem")
    assert chat._full_route()[-1] == {"url": "http://127.0.0.1:8000/callback",
                                      "node_id": "cliente-1"}


def test_build_payload_uses_v2_when_there_is_a_route():
    chat = make_client(route=[{"url": "http://127.0.0.1:9001/forward", "node_id": "worker-a"}])
    envelope, url, node_id = chat._build_payload(
        "7bd2a1e2-0000-4000-8000-000000000000", torch.ones(1, 2, 4), torch.zeros(1, 2)
    )
    assert envelope["v"] == routing.ENVELOPE_VERSION
    assert envelope["hop"] == 0
    assert url == "http://127.0.0.1:9001/forward"
    assert node_id == "worker-a"


def test_build_payload_falls_back_to_v1_for_static_chains():
    """Con --next-node el encadenamiento sigue siendo el de los workers."""
    chat = make_client(next_node_url="http://127.0.0.1:9001/forward")
    envelope, url, node_id = chat._build_payload(
        "7bd2a1e2-0000-4000-8000-000000000000", torch.ones(1, 2, 4), torch.zeros(1, 2)
    )
    assert set(envelope) == {"request_id", "payload"}
    assert url == "http://127.0.0.1:9001/forward"
    assert node_id == ""


def test_invalidate_route_marks_the_node_and_forces_replanning():
    chat = make_client(route=["http://127.0.0.1:9001/forward"])
    chat._invalidate_route("worker-a")
    assert chat.route is None
    assert chat.primary_node_url is None
    assert "worker-a" in chat.failed_nodes


def test_static_pipeline_survives_a_transient_failure():
    """Un timeout puntual no debe dejar la sesión sin destino configurado."""
    chat = make_client(next_node_url="http://127.0.0.1:9001/forward")

    async def no_tracker():
        return None

    chat._plan_route = no_tracker
    chat._legacy_route = no_tracker

    chat._invalidate_route()
    assert chat.primary_node_url is None
    assert asyncio.run(chat._ensure_target()) is True
    assert chat.primary_node_url == "http://127.0.0.1:9001/forward"


def test_pipeline_step_replans_and_gives_up_after_the_attempt_limit():
    """Cada rechazo invalida la ruta y pide otra; no se reintenta eternamente."""
    chat = make_client()
    plans = []

    async def fake_ensure_target():
        chat.route = [{"url": "http://127.0.0.1:9001/forward", "node_id": f"worker-{len(plans)}"}]
        plans.append(chat.route)
        return True

    async def fake_send(session, url, payload, node_id=""):
        return False  # el nodo rechaza siempre

    chat._ensure_target = fake_ensure_target
    chat._send_to_node = fake_send

    with pytest.raises(PipelineError):
        asyncio.run(chat._run_pipeline_step(torch.ones(1, 2, 4), torch.zeros(1, 2)))

    assert len(plans) == MAX_ROUTE_ATTEMPTS
    assert not chat.pending_requests, "no deben quedar futuros huérfanos"


def test_reconnect_backoff_retries_before_giving_up(monkeypatch):
    """Si el tracker parpadea, el cliente reintenta en vez de rendirse."""
    chat = make_client()
    delays, messages = [], []
    attempts = {"count": 0}

    async def flaky_find():
        attempts["count"] += 1
        return attempts["count"] >= 3  # falla las dos primeras veces

    async def fake_sleep(delay):
        delays.append(delay)

    chat._find_target_once = flaky_find
    monkeypatch.setattr(chat_agent_module.asyncio, "sleep", fake_sleep)

    ok = asyncio.run(chat._ensure_target(messages.append, attempts=5))

    assert ok is True
    assert attempts["count"] == 3
    assert delays == [chat_agent_module.RECONNECT_BASE_DELAY,
                      chat_agent_module.RECONNECT_BASE_DELAY * 2]
    assert any("Reconectando con la red P2P" in message for message in messages)
    assert any("Reconectado" in message for message in messages)


def test_reconnect_gives_up_with_an_actionable_message(monkeypatch):
    chat = make_client()
    messages = []

    async def always_fails():
        return False

    async def fake_sleep(delay):
        return None

    chat._find_target_once = always_fails
    monkeypatch.setattr(chat_agent_module.asyncio, "sleep", fake_sleep)

    assert asyncio.run(chat._ensure_target(messages.append, attempts=3)) is False
    final = messages[-1]
    assert "No hay nodos disponibles" in final and "Ajustes" in final


def test_reconnect_backoff_is_capped():
    chat = make_client()
    delays = [min(chat_agent_module.RECONNECT_BASE_DELAY * (2 ** attempt),
                  chat_agent_module.MAX_RECONNECT_DELAY) for attempt in range(10)]
    assert max(delays) == chat_agent_module.MAX_RECONNECT_DELAY


def _crossverify_client(outputs):
    """Cliente con verificacion cruzada (CAPA 3).

    `outputs` mapea node_id -> tensor que devuelve ese nodo. La ruta primaria es
    worker-a; las alternativas salen del resto de claves de `outputs` segun las
    exclusiones, de modo que cada replica es una ruta DISJUNTA e independiente.
    """
    chat = make_client(route=[{"url": "http://127.0.0.1:9001/forward", "node_id": "worker-a"}])
    chat.audit_probability = 1.0
    chat.audit_epsilon = 1e-3
    chat.audit_stats = {"checked": 0, "mismatches": 0}
    chat._audit_warned = False
    chat.sampler = AdaptiveSampler()
    reported = []
    pool = [node_id for node_id in outputs if node_id != "worker-a"]

    async def fake_plan_route(exclude=None):
        exclude = exclude or set()
        for node_id in pool:
            if node_id not in exclude:
                chat.route = [{"url": f"http://127.0.0.1:9000/{node_id}", "node_id": node_id}]
                return chat.route
        chat.route = None
        return None

    async def fake_step(hidden_states, position_ids, timeout=30.0):
        return outputs[routing.hop_node_id(chat.route[0])]

    async def fake_post_report(node_ids, reason, divergence=None):
        reported.append((list(node_ids), reason))

    chat._plan_route = fake_plan_route
    chat._run_pipeline_step = fake_step
    chat._post_report = fake_post_report
    return chat, reported


def test_audit_accepts_a_unanimous_verdict():
    reference = torch.ones(1, 1, 8)
    chat, reported = _crossverify_client({
        "worker-a": reference, "worker-b": reference.clone(), "worker-c": reference.clone(),
    })

    assert asyncio.run(chat._audit_step(torch.ones(1, 2, 4), torch.zeros(1, 2), reference)) is True
    assert chat.audit_stats == {"checked": 1, "mismatches": 0}
    assert not reported
    # La ruta original se restaura: auditar no debe cambiar el pipeline en uso.
    assert routing.hop_node_id(chat.route[0]) == "worker-a"


def test_audit_attributes_the_divergence_to_the_minority_node():
    """Dos honestos coinciden; el tramposo minoritario (worker-a) queda senalado
    SIN castigar a los honestos."""
    honest = torch.ones(1, 1, 8)
    cheat = torch.zeros(1, 1, 8)
    chat, reported = _crossverify_client({
        "worker-a": cheat, "worker-b": honest, "worker-c": honest.clone(),
    })

    assert asyncio.run(chat._audit_step(torch.ones(1, 2, 4), torch.zeros(1, 2), cheat)) is False
    assert chat.audit_stats == {"checked": 1, "mismatches": 1}
    assert reported and reported[0] == (["worker-a"], "divergence")
    assert "worker-a" in chat.failed_nodes
    assert "worker-b" not in chat.failed_nodes, "un honesto no debe ser castigado"
    assert chat.route is None, "la ruta con el nodo culpable debe descartarse"


def test_audit_is_inconclusive_without_a_majority():
    """Con solo dos opiniones enfrentadas (no hay tercera ruta) no se puede
    atribuir: se declara inconcluso y NO se castiga a ciegas (mejora sobre la
    auditoria de una sola alternativa)."""
    honest = torch.ones(1, 1, 8)
    cheat = torch.zeros(1, 1, 8)
    chat, reported = _crossverify_client({"worker-a": cheat, "worker-b": honest})

    assert asyncio.run(chat._audit_step(torch.ones(1, 2, 4), torch.zeros(1, 2), cheat)) is None
    assert chat.audit_stats == {"checked": 1, "mismatches": 0}
    assert not reported
    assert "worker-a" not in chat.failed_nodes
    assert routing.hop_node_id(chat.route[0]) == "worker-a", "sin culpa clara, la ruta se conserva"


def test_audit_is_skipped_without_node_identities():
    """Sin identidad criptográfica no hay a quién atribuir la divergencia."""
    chat = make_client(route=["http://127.0.0.1:9001/forward"])
    chat.audit_probability = 1.0
    chat.audit_epsilon = 1e-3
    chat.audit_stats = {"checked": 0, "mismatches": 0}
    chat._audit_warned = False

    assert asyncio.run(chat._audit_step(torch.ones(1, 2, 4), torch.zeros(1, 2),
                                        torch.ones(1, 1, 8))) is None
    assert chat.audit_stats["checked"] == 0


def test_pipeline_step_returns_logits_on_success():
    chat = make_client(route=["http://127.0.0.1:9001/forward"])
    expected = torch.ones(1, 1, 8)

    async def fake_ensure_target():
        return True

    async def fake_send(session, url, payload, node_id=""):
        # Simula el callback del último nodo resolviendo el futuro pendiente.
        request_id = next(iter(chat.pending_requests))
        chat.pending_requests[request_id].set_result(expected)
        return True

    chat._ensure_target = fake_ensure_target
    chat._send_to_node = fake_send

    result = asyncio.run(chat._run_pipeline_step(torch.ones(1, 2, 4), torch.zeros(1, 2)))
    assert torch.equal(result, expected)
    assert not chat.pending_requests
