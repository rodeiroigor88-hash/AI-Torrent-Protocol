"""Pruebas del modelo de reputacion y scoring de nodos (CAPA 2, logica pura).

No levantan el tracker: ejercen directamente la politica de cuarentena, el
decaimiento del historial y la formula del score, de forma determinista con un
reloj explicito.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import reputation
from src.reputation import (
    ANOMALY_HALF_LIFE_SECONDS,
    NodeReputation,
    QUARANTINE_BASE_SECONDS,
    QUARANTINE_MAX_SECONDS,
    quarantine_backoff,
    score_node,
)


# ---------------------------------------------------------------- cuarentena

def test_quarantine_backoff_grows_exponentially_and_is_capped():
    assert quarantine_backoff(0) == 0.0
    assert quarantine_backoff(1) == QUARANTINE_BASE_SECONDS
    assert quarantine_backoff(2) == QUARANTINE_BASE_SECONDS * 2
    assert quarantine_backoff(3) == QUARANTINE_BASE_SECONDS * 4
    assert quarantine_backoff(100) == QUARANTINE_MAX_SECONDS


def test_record_anomaly_quarantines_then_expires():
    rep = NodeReputation(node_id="a")
    rep.record_anomaly(reputation.REASON_TIMEOUT, now=1000.0)

    assert rep.anomalies == 1
    assert rep.last_reason == reputation.REASON_TIMEOUT
    assert rep.is_quarantined(1000.0) is True
    assert rep.is_quarantined(1000.0 + QUARANTINE_BASE_SECONDS - 1) is True
    assert rep.is_quarantined(1000.0 + QUARANTINE_BASE_SECONDS + 1) is False


def test_unknown_reason_is_normalized():
    rep = NodeReputation(node_id="a")
    rep.record_anomaly("motivo-inventado", now=0.0)
    assert rep.last_reason == reputation.REASON_DIVERGENCE


def test_effective_anomalies_decay_over_time():
    rep = NodeReputation(node_id="a")
    rep.record_anomaly(reputation.REASON_DIVERGENCE, now=0.0)
    rep.record_anomaly(reputation.REASON_DIVERGENCE, now=0.0)  # 2 anomalias en t=0

    assert rep.effective_anomalies(0.0) == pytest.approx(2.0)
    assert rep.effective_anomalies(ANOMALY_HALF_LIFE_SECONDS) == pytest.approx(1.0)
    assert rep.effective_anomalies(2 * ANOMALY_HALF_LIFE_SECONDS) == pytest.approx(0.5)


# ---------------------------------------------------------------- latencia

def test_observe_latency_is_an_ewma():
    rep = NodeReputation(node_id="a")
    rep.observe_latency(100.0)
    assert rep.latency_ms == pytest.approx(100.0)     # primera muestra = valor
    rep.observe_latency(200.0)
    assert 100.0 < rep.latency_ms < 200.0             # suavizado hacia la nueva
    rep.observe_latency(-5.0)                          # muestra invalida: ignorada
    assert 100.0 < rep.latency_ms < 200.0


# ---------------------------------------------------------------- score

def test_score_prefers_lower_latency():
    fast = NodeReputation(node_id="fast", latency_ms=50.0)
    slow = NodeReputation(node_id="slow", latency_ms=500.0)
    assert score_node(fast, None, None, now=0.0) > score_node(slow, None, None, now=0.0)


def test_score_penalizes_anomalies():
    clean = NodeReputation(node_id="clean")
    dirty = NodeReputation(node_id="dirty")
    dirty.record_anomaly(reputation.REASON_DIVERGENCE, now=0.0)
    # Ya fuera de cuarentena, pero con el historial reciente aun pesando.
    later = QUARANTINE_BASE_SECONDS + 1
    assert score_node(clean, None, None, now=later) > score_node(dirty, None, None, now=later)


def test_score_rewards_donated_bandwidth():
    rep = NodeReputation(node_id="a", latency_ms=100.0)
    big = score_node(rep, donated_cores=8, donated_ram_mb=8192, now=0.0)
    small = score_node(rep, donated_cores=1, donated_ram_mb=512, now=0.0)
    assert big > small


def test_quarantined_node_scores_zero():
    rep = NodeReputation(node_id="a", latency_ms=10.0)  # latencia excelente
    rep.record_anomaly(reputation.REASON_CORRUPT, now=0.0)
    assert score_node(rep, 8, 8192, now=0.0) == 0.0     # la cuarentena manda


def test_none_reputation_is_neutral():
    score = score_node(None, None, None, now=0.0)
    assert 0.0 < score < 1.0
