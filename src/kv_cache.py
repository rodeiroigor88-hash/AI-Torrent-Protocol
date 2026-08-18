"""Infraestructura de cache KV por peticion para la capa del nodo (CAPA 1).

Este modulo NO ejecuta inferencia: gestiona el CICLO DE VIDA de los buffers de
atencion (K/V) y de los estados ocultos de salida que un worker acumula a lo
largo de una generacion, para que cada token nuevo procese solo la POSICION
nueva en vez de recalcular todo el prefijo (coste O(n^2) -> O(n)).

Se separa a proposito de ``worker.py`` y ``p2p_node.py`` por dos motivos:

* La logica de deshaucio (LRU + TTL + tope de memoria) es pura y se puede probar
  offline, sin cargar ningun modelo ni levantar la red.
* El buffer se disena desde ya para *Speculative Decoding*: un nodo de poca
  potencia puede FORKAR el cache confirmado, avanzar tokens tentativos de forma
  asincrona y luego adoptarlos (``commit``) o descartarlos (``rollback``) sin
  corromper el estado confirmado.

Terminologia:

* :class:`RequestCacheEntry` -- el buffer de UNA generacion en curso.
* :class:`KVCacheStore` -- el almacen de todas las entradas vivas del nodo, con
  deshaucio por cantidad (LRU), por antiguedad (TTL) y por presupuesto de RAM.

El almacen es agnostico a COMO se identifica una peticion: solo ve una ``str``.
Hoy el llamador (el worker) usa una clave direccionada por contenido del
prefijo, porque el cliente acuna un ``request_id`` nuevo por token y ese id no
sirve para reutilizar cache entre tokens.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

try:  # torch es opcional: las pruebas de logica pura no deben exigirlo.
    import torch
except Exception:  # pragma: no cover - entorno sin torch
    torch = None  # type: ignore


# ---------------------------------------------------------------- utilidades

def _tensor_nbytes(obj: Any) -> int:
    """Bytes aproximados de un tensor o de una estructura anidada de tensores.

    Recorre listas, tuplas y dicts. Para objetos opacos (p.ej. un ``Cache`` de
    HuggingFace) intenta leer un atributo ``nbytes``; si no existe devuelve 0,
    de modo que la contabilidad de memoria nunca lanza por un tipo inesperado.
    """
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_nbytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_tensor_nbytes(value) for value in obj.values())
    nbytes = getattr(obj, "nbytes", None)
    return int(nbytes) if isinstance(nbytes, int) else 0


def _clone(obj: Any) -> Any:
    """Copia profunda de tensores anidados para aislar una rama especulativa.

    Los tensores se ``detach().clone()`` para que la rama tentativa no comparta
    almacenamiento con el estado confirmado. Los contenedores se recorren. Un
    objeto opaco con ``.clone()`` se clona con el; si no es clonable, se comparte
    (comportamiento documentado: el llamador que use cache opaco debe pasar su
    propia ``clone_fn`` al forkar).
    """
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, list):
        return [_clone(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_clone(item) for item in obj)
    if isinstance(obj, dict):
        return {key: _clone(value) for key, value in obj.items()}
    clone = getattr(obj, "clone", None)
    if callable(clone):
        try:
            return clone()
        except Exception:  # pragma: no cover - objeto con clone() roto
            return obj
    return obj


def _crop_seq(obj: Any, seq_len: int, seq_dim: int) -> Any:
    """Recorta la dimension de secuencia de un tensor (o estructura) a ``seq_len``.

    Sirve para descartar posiciones tentativas no aceptadas en *Speculative
    Decoding*. Solo recorta tensores cuya dimension ``seq_dim`` sea mayor que
    ``seq_len``; el resto se deja intacto para no romper estructuras heterogeneas.
    """
    if torch is not None and isinstance(obj, torch.Tensor):
        if obj.dim() > abs(seq_dim) - (1 if seq_dim < 0 else 0) and obj.size(seq_dim) > seq_len:
            index = [slice(None)] * obj.dim()
            index[seq_dim] = slice(0, seq_len)
            return obj[tuple(index)]
        return obj
    if isinstance(obj, list):
        return [_crop_seq(item, seq_len, seq_dim) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_crop_seq(item, seq_len, seq_dim) for item in obj)
    if isinstance(obj, dict):
        return {key: _crop_seq(value, seq_len, seq_dim) for key, value in obj.items()}
    return obj


# --------------------------------------------------------------- entrada

@dataclass
class RequestCacheEntry:
    """Buffer de una generacion: K/V de atencion + estados de salida acumulados.

    ``past_key_values`` es opaco a proposito (un ``Cache`` de HuggingFace, una
    lista de tuplas ``(k, v)``, o lo que use el modelo). El almacen solo necesita
    poder medir su tamano, clonarlo y liberarlo.

    :param key: clave con la que la entrada esta indexada en el almacen.
    :param past_key_values: objeto de cache de atencion del modelo (opaco).
    :param output_buffer: estados ocultos de salida acumulados ``[1, seq, hidden]``.
    :param seq_len: numero de posiciones ya procesadas y confirmadas.
    :param kv_seq_dim: eje de secuencia dentro de los tensores K/V (por defecto
        ``-2`` para la forma habitual ``[batch, heads, seq, head_dim]``).
    :param output_seq_dim: eje de secuencia del ``output_buffer`` (por defecto 1
        para ``[batch, seq, hidden]``).
    """

    key: str
    past_key_values: Any = None
    output_buffer: Any = None
    seq_len: int = 0
    kv_seq_dim: int = -2
    output_seq_dim: int = 1
    created_at: float = 0.0
    last_access: float = 0.0

    def touch(self, now: float) -> None:
        """Marca la entrada como usada ahora (para LRU y TTL)."""
        self.last_access = now

    def nbytes(self) -> int:
        """Huella de memoria aproximada (K/V + estados de salida), en bytes."""
        return _tensor_nbytes(self.past_key_values) + _tensor_nbytes(self.output_buffer)

    def is_expired(self, now: float, ttl_seconds: float) -> bool:
        """True si no se accede a la entrada desde hace mas de ``ttl_seconds``."""
        return ttl_seconds > 0 and (now - self.last_access) > ttl_seconds

    # -------------------------------------------------- speculative decoding

    def fork(self, clone_fn: Optional[Callable[[Any], Any]] = None) -> "RequestCacheEntry":
        """Devuelve una copia independiente para validar tokens tentativos.

        La rama forkada no comparte almacenamiento mutable con la confirmada, de
        modo que un nodo puede avanzarla de forma asincrona y luego adoptarla o
        descartarla sin efectos colaterales.

        :param clone_fn: clonador alternativo para caches opacos (p.ej. el
            ``.clone()`` especifico de un ``Cache`` de HuggingFace). Si es None se
            usa el clonador torch-aware por defecto.
        """
        cloner = clone_fn or _clone
        return RequestCacheEntry(
            key=self.key,
            past_key_values=cloner(self.past_key_values),
            output_buffer=cloner(self.output_buffer),
            seq_len=self.seq_len,
            kv_seq_dim=self.kv_seq_dim,
            output_seq_dim=self.output_seq_dim,
            created_at=self.created_at,
            last_access=self.last_access,
        )

    def truncate(self, seq_len: int) -> None:
        """Recorta la entrada a ``seq_len`` posiciones (descarta las tentativas).

        Usado tras una ronda de *Speculative Decoding* cuando solo ``seq_len`` de
        los tokens propuestos se aceptaron: se conservan las K/V de las posiciones
        aceptadas y se tiran las del resto.
        """
        if seq_len < 0:
            raise ValueError("seq_len no puede ser negativo")
        if seq_len >= self.seq_len:
            return
        self.past_key_values = _crop_seq(self.past_key_values, seq_len, self.kv_seq_dim)
        self.output_buffer = _crop_seq(self.output_buffer, seq_len, self.output_seq_dim)
        self.seq_len = seq_len


# --------------------------------------------------------------- almacen

class KVCacheStore:
    """Almacen thread-safe de buffers KV con deshaucio por LRU, TTL y memoria.

    La inferencia del worker corre en un hilo aparte (``asyncio.to_thread``), asi
    que todas las operaciones del almacen se serializan con un ``RLock``. El
    deshaucio protege contra fugas de memoria de tres formas independientes y
    acumulables:

    * **LRU**: nunca mas de ``max_entries`` generaciones vivas a la vez.
    * **TTL**: una generacion abandonada (cliente que se fue) caduca a los
      ``ttl_seconds`` y se libera en el siguiente barrido.
    * **Memoria**: si ``max_total_bytes`` esta fijado, se expulsan las entradas
      menos recientes hasta bajar del presupuesto.

    Todos los limites son opcionales (``0``/``None`` = sin limite en ese eje),
    pero se recomienda fijar al menos ``max_entries`` en produccion.
    """

    def __init__(
        self,
        *,
        max_entries: int = 64,
        ttl_seconds: float = 120.0,
        max_total_bytes: Optional[int] = None,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 0:
            raise ValueError("max_entries no puede ser negativo")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds no puede ser negativo")
        if max_total_bytes is not None and max_total_bytes < 0:
            raise ValueError("max_total_bytes no puede ser negativo")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.max_total_bytes = max_total_bytes
        self._time = time_source

        self._entries: "OrderedDict[str, RequestCacheEntry]" = OrderedDict()
        self._lock = threading.RLock()

        # Telemetria: util para exponer un hit-rate real y para las pruebas.
        self.hits = 0
        self.misses = 0
        self.evictions_lru = 0
        self.evictions_ttl = 0
        self.evictions_mem = 0

    # ------------------------------------------------------------ consulta

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def total_nbytes(self) -> int:
        """Suma de la huella de memoria de todas las entradas vivas."""
        with self._lock:
            return sum(entry.nbytes() for entry in self._entries.values())

    def get(self, key: str) -> Optional[RequestCacheEntry]:
        """Devuelve la entrada de ``key`` (marcandola como reciente) o None.

        Actualiza los contadores ``hits``/``misses`` para poder medir el
        hit-rate real de la cache del nodo.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            now = self._time()
            if entry.is_expired(now, self.ttl_seconds):
                # Caducada: se trata como fallo y se libera de inmediato.
                del self._entries[key]
                self.evictions_ttl += 1
                self.misses += 1
                return None
            entry.touch(now)
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def stats(self) -> Dict[str, int]:
        """Instantanea de la telemetria (para logs y para /status)."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "evictions_lru": self.evictions_lru,
                "evictions_ttl": self.evictions_ttl,
                "evictions_mem": self.evictions_mem,
            }

    # ---------------------------------------------------------- mutacion

    def put(self, key: str, entry: RequestCacheEntry) -> None:
        """Inserta o reemplaza la entrada de ``key`` y aplica los limites.

        Fija ``key``/``created_at``/``last_access`` en la entrada para que su
        metadato coincida con su posicion en el almacen.
        """
        now = self._time()
        with self._lock:
            entry.key = key
            if not entry.created_at:
                entry.created_at = now
            entry.last_access = now
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._enforce_limits()

    def rekey(self, old_key: str, new_key: str) -> Optional[RequestCacheEntry]:
        """Reindexa una entrada bajo una clave nueva de forma atomica.

        Imprescindible para la cache direccionada por prefijo: al generar el
        token N+1, la clave (hash del prefijo) cambia, pero el buffer es el
        mismo. Devuelve la entrada movida, o None si ``old_key`` no existe.
        """
        with self._lock:
            entry = self._entries.pop(old_key, None)
            if entry is None:
                return None
            entry.key = new_key
            entry.touch(self._time())
            self._entries[new_key] = entry
            self._entries.move_to_end(new_key)
            self._enforce_limits()
            return entry

    def discard(self, key: str) -> bool:
        """Libera la entrada de ``key`` (generacion terminada o abortada).

        Devuelve True si existia. Llamar a esto en cuanto una generacion acaba es
        la primera linea de defensa contra fugas de memoria.
        """
        with self._lock:
            existed = self._entries.pop(key, None) is not None
            return existed

    def commit(self, key: str, entry: RequestCacheEntry) -> None:
        """Adopta una rama especulativa como el nuevo estado confirmado de ``key``."""
        self.put(key, entry)

    def clear(self) -> None:
        """Vacia el almacen entero (p.ej. al apagar el nodo)."""
        with self._lock:
            self._entries.clear()

    # ---------------------------------------------------------- deshaucio

    def evict_expired(self) -> int:
        """Libera todas las entradas caducadas por TTL. Devuelve cuantas.

        Pensado para llamarse periodicamente desde un barredor del nodo, de modo
        que una generacion abandonada no ocupe RAM hasta el proximo ``get``.
        """
        if self.ttl_seconds <= 0:
            return 0
        now = self._time()
        with self._lock:
            stale = [key for key, entry in self._entries.items()
                     if entry.is_expired(now, self.ttl_seconds)]
            for key in stale:
                del self._entries[key]
            self.evictions_ttl += len(stale)
            return len(stale)

    def _enforce_limits(self) -> None:
        """Aplica los topes de cantidad y de memoria. Se llama con el lock tomado."""
        # Tope por cantidad: expulsa las menos recientes (frente del OrderedDict).
        if self.max_entries:
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions_lru += 1
        # Tope por memoria: sigue expulsando las menos recientes hasta encajar.
        if self.max_total_bytes:
            total = sum(entry.nbytes() for entry in self._entries.values())
            while total > self.max_total_bytes and len(self._entries) > 1:
                _, evicted = self._entries.popitem(last=False)
                total -= evicted.nbytes()
                self.evictions_mem += 1
