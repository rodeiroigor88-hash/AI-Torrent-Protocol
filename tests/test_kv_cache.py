"""Pruebas de la infraestructura de cache KV (CAPA 1).

Son pruebas de LOGICA PURA: ejercen el ciclo de vida del buffer y el deshaucio
(LRU / TTL / memoria) y las primitivas de *Speculative Decoding* (fork/truncate)
sin cargar ningun modelo ni levantar la red. La correctitud de la matematica de
atencion (que cada token procese solo la posicion nueva) es responsabilidad de la
integracion en el worker y se valida aparte con un modelo real.
"""

import os
import sys
import threading

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kv_cache import KVCacheStore, RequestCacheEntry


class ManualClock:
    """Reloj controlado: probar un TTL de 120 s no puede tardar 120 s."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _entry(key: str, nfloats: int = 100) -> RequestCacheEntry:
    """Entrada con un tensor de salida de tamano conocido (nfloats*4 bytes)."""
    return RequestCacheEntry(
        key=key,
        past_key_values=[(torch.zeros(1, 2, 4, 3), torch.zeros(1, 2, 4, 3))],
        output_buffer=torch.zeros(1, nfloats, 1),
        seq_len=4,
    )


# ---------------------------------------------------------------- hit / miss

def test_get_records_hit_and_miss():
    store = KVCacheStore()
    assert store.get("ausente") is None
    store.put("a", _entry("a"))
    assert store.get("a") is not None
    stats = store.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1


def test_rekey_moves_the_entry_under_a_new_key():
    """La cache por prefijo cambia de clave cada token pero reusa el buffer."""
    store = KVCacheStore()
    store.put("prefijo-t0", _entry("prefijo-t0"))
    moved = store.rekey("prefijo-t0", "prefijo-t1")
    assert moved is not None
    assert "prefijo-t0" not in store
    assert store.get("prefijo-t1") is not None
    assert store.rekey("no-existe", "otro") is None


def test_discard_frees_the_entry():
    store = KVCacheStore()
    store.put("a", _entry("a"))
    assert store.discard("a") is True
    assert store.discard("a") is False
    assert len(store) == 0


# ---------------------------------------------------------------- deshaucio

def test_lru_evicts_the_least_recently_used():
    store = KVCacheStore(max_entries=2, ttl_seconds=0)
    store.put("a", _entry("a"))
    store.put("b", _entry("b"))
    store.get("a")                 # "a" pasa a ser la mas reciente
    store.put("c", _entry("c"))    # desborda: debe caer "b"
    assert "a" in store
    assert "c" in store
    assert "b" not in store
    assert store.stats()["evictions_lru"] == 1


def test_ttl_expiry_on_access_and_on_sweep():
    clock = ManualClock()
    store = KVCacheStore(max_entries=0, ttl_seconds=100, time_source=clock)
    store.put("a", _entry("a"))

    clock.advance(50)
    assert store.get("a") is not None, "dentro de la ventana TTL"

    clock.advance(101)
    assert store.get("a") is None, "caducada: get la trata como fallo y la libera"
    assert store.stats()["evictions_ttl"] == 1

    # Y el barredor libera las abandonadas sin necesidad de un get.
    store.put("b", _entry("b"))
    clock.advance(101)
    assert store.evict_expired() == 1
    assert len(store) == 0


def test_memory_budget_evicts_until_under_the_cap():
    # Cada entrada de salida pesa 100*4 = 400 bytes (mas las K/V, ~192 bytes).
    store = KVCacheStore(max_entries=0, ttl_seconds=0, max_total_bytes=1500)
    for name in ("a", "b", "c", "d", "e"):
        store.put(name, _entry(name))
    assert store.total_nbytes() <= 1500
    assert store.stats()["evictions_mem"] >= 1
    assert len(store) >= 1, "nunca se vacia por completo por presupuesto"


# ----------------------------------------------------- speculative decoding

def test_fork_is_independent_from_the_confirmed_entry():
    original = _entry("a")
    forked = original.fork()
    forked.output_buffer.add_(1.0)          # avanzar la rama tentativa in-place
    assert torch.equal(original.output_buffer, torch.zeros(1, 100, 1)), \
        "la rama especulativa no debe tocar el estado confirmado"
    assert not torch.equal(forked.output_buffer, original.output_buffer)


def test_truncate_drops_tentative_positions():
    entry = RequestCacheEntry(
        key="a",
        past_key_values=[(torch.zeros(1, 2, 5, 3), torch.zeros(1, 2, 5, 3))],
        output_buffer=torch.zeros(1, 5, 4),
        seq_len=5,
    )
    entry.truncate(3)   # solo 3 de los 5 tokens propuestos se aceptaron
    assert entry.seq_len == 3
    assert entry.output_buffer.shape == (1, 3, 4)
    k, v = entry.past_key_values[0]
    assert k.shape == (1, 2, 3, 3)
    assert v.shape == (1, 2, 3, 3)


def test_truncate_is_a_noop_when_not_shorter():
    entry = _entry("a")           # seq_len = 4
    entry.truncate(4)
    assert entry.seq_len == 4
    with pytest.raises(ValueError):
        entry.truncate(-1)


def test_commit_replaces_the_confirmed_branch():
    store = KVCacheStore()
    store.put("a", _entry("a"))
    forked = store.get("a").fork()
    forked.output_buffer.add_(2.0)
    forked.seq_len = 5
    store.commit("a", forked)
    confirmed = store.get("a")
    assert confirmed.seq_len == 5
    assert torch.equal(confirmed.output_buffer, torch.full((1, 100, 1), 2.0))


# ---------------------------------------------------------------- concurrencia

def test_store_is_thread_safe_under_concurrent_access():
    """La inferencia corre en un hilo aparte: el almacen debe soportar acceso
    concurrente sin corromperse ni lanzar."""
    store = KVCacheStore(max_entries=0, ttl_seconds=0)
    errors = []

    def worker(base: int) -> None:
        try:
            for i in range(200):
                key = f"n{base}-{i}"
                store.put(key, _entry(key))
                store.get(key)
                if i % 2 == 0:
                    store.discard(key)
        except Exception as exc:  # pragma: no cover - solo salta si hay carrera
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"acceso concurrente rompio el almacen: {errors}"
    # 8 hilos * 100 entradas impares que NO se descartan = 800 vivas.
    assert len(store) == 800
