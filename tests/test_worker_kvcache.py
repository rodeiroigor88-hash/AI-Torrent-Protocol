"""Validacion de la KV-Cache incremental del worker (CAPA 1 - integracion).

Prueba que procesar SOLO la posicion nueva reutilizando las K/V cacheadas produce
exactamente la misma salida que recalcular la secuencia completa, usando un
transformer Llama real construido DESDE CONFIG (sin descargar pesos, offline).

Es la prueba que faltaba para la parte tensorial de la CAPA 1: la infraestructura
(`kv_cache.py`) se prueba aparte; aqui se valida la matematica de atencion.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM

from src.kv_cache import KVCacheStore
from src.worker import GenericWorkerModel


def _tiny_base():
    """Modelo Llama diminuto construido desde config (sin red)."""
    cfg = AutoConfig.for_model(
        "llama", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
        max_position_embeddings=64,
    )
    return AutoModelForCausalLM.from_config(cfg).eval()


def _worker(start, end, is_last, kv_store, base):
    model = GenericWorkerModel("tiny", start, end, is_last=is_last,
                               kv_store=kv_store, base_model=base)
    return model.eval()


def _payload(hidden_states, seq_len):
    hs = hidden_states[:, :seq_len, :]
    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    return hs, pos


@pytest.fixture(scope="module")
def sequence():
    torch.manual_seed(7)
    return torch.randn(1, 8, 32)   # secuencia de entrada que crece token a token


# --------------------------------------------- equivalencia nodo intermedio

def test_incremental_matches_full_recompute_for_a_middle_shard(sequence):
    base = _tiny_base()
    worker = _worker(1, 2, is_last=False, kv_store=KVCacheStore(), base=base)

    max_err = 0.0
    for seq_len in range(1, 9):
        hs, pos = _payload(sequence, seq_len)
        cached = worker.forward({"hidden_states": hs, "position_ids": pos})
        reference = worker._forward_full(hs, pos)
        assert cached["hidden_states"].shape == reference["hidden_states"].shape
        err = (cached["hidden_states"] - reference["hidden_states"]).abs().max().item()
        max_err = max(max_err, err)
    assert max_err < 1e-4, f"la salida incremental diverge del recalculo completo: {max_err}"


# --------------------------------------------- equivalencia nodo final

def test_incremental_matches_full_recompute_for_the_last_shard(sequence):
    base = _tiny_base()
    worker = _worker(2, 3, is_last=True, kv_store=KVCacheStore(), base=base)

    for seq_len in range(1, 9):
        hs, pos = _payload(sequence, seq_len)
        cached_logits = worker.forward({"hidden_states": hs, "position_ids": pos})
        full_logits = worker._forward_full(hs, pos)
        # El nodo final solo devuelve los logits de la ULTIMA posicion (lo unico
        # que usa el cliente); el recalculo completo devuelve toda la secuencia.
        assert cached_logits.shape[1] == 1
        err = (cached_logits[:, -1, :] - full_logits[:, -1, :]).abs().max().item()
        assert err < 1e-4, f"logits incrementales divergen en seq_len={seq_len}: {err}"


# --------------------------------------------- recuperacion tras un desalojo

def test_incremental_recovers_after_a_cache_eviction(sequence):
    """Si la entrada se desaloja a mitad de generacion, el siguiente token hace
    un prefill completo (MISS) y sigue coincidiendo con el recalculo."""
    base = _tiny_base()
    store = KVCacheStore()
    worker = _worker(0, 1, is_last=False, kv_store=store, base=base)

    for seq_len in range(1, 6):
        hs, pos = _payload(sequence, seq_len)
        cached = worker.forward({"hidden_states": hs, "position_ids": pos})
        reference = worker._forward_full(hs, pos)
        assert (cached["hidden_states"] - reference["hidden_states"]).abs().max().item() < 1e-4
        if seq_len == 3:
            store.clear()   # simula un desalojo por presion de memoria

    # Tras el desalojo, el token 6 provoca un MISS y debe reconstruir bien.
    hs, pos = _payload(sequence, 6)
    cached = worker.forward({"hidden_states": hs, "position_ids": pos})
    reference = worker._forward_full(hs, pos)
    assert (cached["hidden_states"] - reference["hidden_states"]).abs().max().item() < 1e-4


# --------------------------------------------- la cache desactivada no cambia nada

def test_cache_disabled_keeps_the_historic_behavior(sequence):
    base = _tiny_base()
    worker = _worker(1, 2, is_last=False, kv_store=None, base=base)
    assert worker._cache_capable is False

    hs, pos = _payload(sequence, 5)
    out = worker.forward({"hidden_states": hs, "position_ids": pos})
    reference = worker._forward_full(hs, pos)
    assert torch.equal(out["hidden_states"], reference["hidden_states"])


# --------------------------------------------- la cache realmente se puebla

def test_store_accumulates_and_rekeys_across_tokens(sequence):
    base = _tiny_base()
    store = KVCacheStore()
    worker = _worker(1, 2, is_last=False, kv_store=store, base=base)

    for seq_len in range(1, 5):
        hs, pos = _payload(sequence, seq_len)
        worker.forward({"hidden_states": hs, "position_ids": pos})
        # Tras cada token hay exactamente una entrada viva (la reindexada al
        # prefijo actual), y con hits crecientes a partir del segundo token.
        assert len(store) == 1
    assert store.stats()["hits"] >= 3, "los tokens 2..4 deben acertar en la cache"
