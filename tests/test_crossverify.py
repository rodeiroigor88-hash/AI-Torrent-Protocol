"""Pruebas de la verificacion cruzada con atribucion (CAPA 3, logica pura).

Ejercen el motor de atribucion por mayoria y el muestreo adaptativo sin red ni
modelo, con tensores sinteticos y un RNG controlado para ser deterministas.
"""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.crossverify import (
    AdaptiveSampler,
    BASE_PROBABILITY,
    NEW_NODE_PROBABILITY,
    SUSPECT_PROBABILITY,
    TRUST_THRESHOLD,
    VerdictStatus,
    attribute,
)


class FixedRng:
    """RNG de un solo valor, para fijar el resultado de should_audit."""

    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


ONES = torch.ones(1, 1, 8)
ZEROS = torch.zeros(1, 1, 8)
TWOS = torch.full((1, 1, 8), 2.0)


# ---------------------------------------------------------------- atribucion

def test_unanimous_has_no_suspect():
    verdict = attribute([("a", ONES), ("b", ONES.clone()), ("c", ONES.clone())])
    assert verdict.status is VerdictStatus.UNANIMOUS
    assert verdict.suspect == []
    assert set(verdict.trusted) == {"a", "b", "c"}


def test_majority_pins_the_outlier():
    """Dos honestos coinciden; el tramposo minoritario queda senalado."""
    verdict = attribute([("honesto1", ONES), ("honesto2", ONES.clone()),
                         ("tramposo", ZEROS)])
    assert verdict.status is VerdictStatus.MAJORITY
    assert verdict.suspect == ["tramposo"]
    assert set(verdict.trusted) == {"honesto1", "honesto2"}
    assert verdict.attributable is True


def test_two_disagreeing_opinions_cannot_attribute():
    """El agujero de la CAPA 2: con solo dos opiniones enfrentadas no se puede
    decidir quien mintio. El dictamen es TIE, no una acusacion a ciegas."""
    verdict = attribute([("a", ONES), ("b", ZEROS)])
    assert verdict.status is VerdictStatus.TIE
    assert verdict.suspect == []
    assert verdict.attributable is False


def test_three_way_split_is_a_tie():
    verdict = attribute([("a", ONES), ("b", ZEROS), ("c", TWOS)])
    assert verdict.status is VerdictStatus.TIE
    assert verdict.attributable is False


def test_single_opinion_is_insufficient():
    verdict = attribute([("a", ONES)])
    assert verdict.status is VerdictStatus.INSUFFICIENT


def test_repeated_label_is_not_two_votes():
    """Dos ejecuciones del mismo nodo no son independientes: no deben formar
    mayoria por si solas."""
    verdict = attribute([("a", ONES), ("a", ONES.clone()), ("b", ZEROS)])
    assert verdict.total == 2
    assert verdict.status is VerdictStatus.TIE


# ------------------------------------------------------- muestreo adaptativo

def test_new_node_is_watched_aggressively():
    sampler = AdaptiveSampler()
    assert sampler.node_probability("desconocido") == NEW_NODE_PROBABILITY


def test_a_disagreement_forces_maximum_scrutiny():
    sampler = AdaptiveSampler()
    sampler.record_result("x", agreed=False)
    assert sampler.node_probability("x") == SUSPECT_PROBABILITY


def test_a_consolidated_node_relaxes_to_base():
    sampler = AdaptiveSampler()
    for _ in range(TRUST_THRESHOLD):
        sampler.record_result("x", agreed=True)
    assert sampler.node_probability("x") == BASE_PROBABILITY


def test_a_suspect_can_be_rehabilitated_after_enough_agreements():
    sampler = AdaptiveSampler()
    sampler.record_result("x", agreed=False)          # una trampa
    for _ in range(TRUST_THRESHOLD - 1):
        sampler.record_result("x", agreed=True)
    assert sampler.node_probability("x") == SUSPECT_PROBABILITY, "aun redimiendose"
    sampler.record_result("x", agreed=True)           # completa la racha
    assert sampler.node_probability("x") == BASE_PROBABILITY


def test_step_probability_is_the_weakest_link():
    sampler = AdaptiveSampler()
    for _ in range(TRUST_THRESHOLD):
        sampler.record_result("confiable", agreed=True)
    sampler.record_result("dudoso", agreed=False)
    # La ruta atraviesa un nodo confiable y uno dudoso: manda el dudoso.
    assert sampler.step_probability(["confiable", "dudoso"]) == SUSPECT_PROBABILITY


def test_should_audit_respects_the_dice():
    sampler = AdaptiveSampler()
    for _ in range(TRUST_THRESHOLD):
        sampler.record_result("x", agreed=True)       # prob = BASE_PROBABILITY
    assert sampler.should_audit(["x"], rng=FixedRng(BASE_PROBABILITY - 0.001)) is True
    assert sampler.should_audit(["x"], rng=FixedRng(BASE_PROBABILITY + 0.001)) is False
    assert sampler.should_audit([], rng=FixedRng(0.0)) is False  # sin identidades


def test_adaptive_sampling_catches_the_low_rate_cheater_far_more():
    """Un Sybil nuevo se audita ~10x mas que con probabilidad fija: hacer trampa
    a tasa baja deja de ser gratis."""
    rng = random.Random(1234)
    adaptive = AdaptiveSampler()
    audits_adaptive = sum(adaptive.should_audit(["sybil"], rng=rng) for _ in range(200))

    audits_fixed = sum((rng.random() < BASE_PROBABILITY) for _ in range(200))

    assert audits_adaptive > 3 * audits_fixed
    assert audits_adaptive > 60   # ~50% de 200, con margen holgado
