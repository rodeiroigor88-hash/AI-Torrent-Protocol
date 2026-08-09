"""Proof of Compute: atestacion firmada y auditoria por muestreo.

Que es esto y que NO es
-----------------------
Una prueba criptografica de que un stack de capas transformer se ejecuto
correctamente (zkML) esta hoy varios ordenes de magnitud fuera de presupuesto
para un modelo de 0.5B en CPU domestica. Lo que implementa este modulo es
**verificacion probabilistica con responsabilidad atribuible**:

* `/attest` — el nodo procesa una entrada canonica derivada de una semilla y
  devuelve la salida junto a una firma ECDSA hecha con la clave privada de su
  certificado. La firma hace la evidencia no repudiable, que es lo que permite
  degradar o expulsar a un nodo tramposo (su identidad es el certificado, no un
  UUID que pueda rotar reiniciando el proceso).
* Auditoria por muestreo — el cliente, con probabilidad `p`, repite un paso por
  una ruta alternativa y compara. La probabilidad de detectar a un tramposo en
  `n` pasos es `1 - (1 - p)^n`.

Deteccion garantizada: devolver ruido, saltarse capas, cargar otro modelo.
NO cubierto: un nodo con los pesos que calcula honestamente pero lento; un
Sybil que pasa la atestacion y luego hace trampa a tasa baja; la colusion
entre auditado y auditor.

Sobre el determinismo
---------------------
`tensor_utils` degrada a float16 por defecto (`secret_sauce`), asi que la
comparacion bit a bit es imposible. Toda verificacion serializa con
`use_secret_sauce=False` y compara bajo tolerancia relativa. El valor de
`DEFAULT_EPSILON` se fija con `tests/test_phase2_5_benchmark.py`, que mide la
divergencia real entre fp32/fp16/bf16 sobre la misma entrada.
"""

import hashlib
import contextlib

import torch

MAX_ATTEST_SEQ_LEN = 32
DEFAULT_ATTEST_SEQ_LEN = 8

# Tolerancia de la distancia L2 relativa entre dos ejecuciones honestas de las
# mismas capas en maquinas distintas. Un nodo que se salta una sola capa se
# desvia ordenes de magnitud por encima de este umbral.
DEFAULT_EPSILON = 2e-2


@contextlib.contextmanager
def deterministic_inference():
    """Fuerza ejecucion reproducible: un solo hilo y algoritmos deterministas.

    En CPU la fuente principal de divergencia es el orden de reduccion entre
    hilos, de ahi `set_num_threads(1)`.
    """
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with torch.no_grad():
            yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)
        torch.set_num_threads(previous_threads)


def seed_to_int(seed: str) -> int:
    """Deriva una semilla entera estable a partir de una cadena arbitraria."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def canonical_challenge(seed: str, hidden_size: int, seq_len: int = DEFAULT_ATTEST_SEQ_LEN,
                        arch: str = "llama/qwen") -> dict:
    """Construye la entrada canonica del reto, identica en retador y retado."""
    if not isinstance(hidden_size, int) or not 0 < hidden_size <= 65536:
        raise ValueError("hidden_size invalido para el reto de atestacion")
    if not isinstance(seq_len, int) or not 0 < seq_len <= MAX_ATTEST_SEQ_LEN:
        raise ValueError(f"seq_len debe estar entre 1 y {MAX_ATTEST_SEQ_LEN}")

    generator = torch.Generator().manual_seed(seed_to_int(seed))
    hidden_states = torch.randn(
        (1, seq_len, hidden_size), generator=generator, dtype=torch.float32
    )
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
    if arch == "gpt2":
        return {"hidden_states": hidden_states}
    return {"hidden_states": hidden_states, "position_ids": position_ids}


def tensor_digest(tensor: torch.Tensor) -> bytes:
    """SHA-256 sobre forma, dtype y bytes del tensor: lo que se firma."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Se esperaba un tensor para el digest")
    contiguous = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(tuple(contiguous.shape)).encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(contiguous.numpy().tobytes())
    return hasher.digest()


def attestation_message(node_id: str, seed: str, digest: bytes) -> bytes:
    """Mensaje firmado: liga la salida a la identidad del nodo y al reto."""
    return b"ai-torrent-attest-v1|" + node_id.encode("utf-8") + b"|" + \
           seed.encode("utf-8") + b"|" + digest


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Distancia L2 relativa. Es la metrica de comparacion, no la igualdad."""
    if candidate.shape != reference.shape:
        return float("inf")
    candidate = candidate.detach().to(torch.float32)
    reference = reference.detach().to(torch.float32)
    denominator = torch.linalg.vector_norm(reference).item()
    if denominator == 0.0:
        return torch.linalg.vector_norm(candidate).item()
    return (torch.linalg.vector_norm(candidate - reference).item() / denominator)


def cosine_similarity(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    if candidate.shape != reference.shape:
        return -1.0
    flat_candidate = candidate.detach().to(torch.float32).flatten()
    flat_reference = reference.detach().to(torch.float32).flatten()
    return torch.nn.functional.cosine_similarity(
        flat_candidate, flat_reference, dim=0, eps=1e-12
    ).item()


def outputs_match(candidate: torch.Tensor, reference: torch.Tensor,
                  epsilon: float = DEFAULT_EPSILON) -> bool:
    if not isinstance(candidate, torch.Tensor) or not isinstance(reference, torch.Tensor):
        return False
    if candidate.shape != reference.shape:
        return False
    if not torch.isfinite(candidate).all():
        return False
    return relative_l2(candidate, reference) <= epsilon
