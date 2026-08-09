"""Limitador de peticiones (token bucket) para los endpoints del enjambre.

Un nodo malicioso puede tumbar a otro nodo -o al tracker- a base de peticiones
masivas: aunque el payload se rechace, cada una cuesta un handshake TLS, una
descompresion y una validacion. El semaforo de trabajo limita el computo
simultaneo, pero no el ritmo de llegada.

El estado se guarda en un `OrderedDict` con tope de clientes: un diccionario sin
limite indexado por IP seria, el mismo, una via de agotamiento de memoria.

El modulo es independiente de aiohttp en su nucleo (`TokenBucket`,
`RateLimiter`) para que el tracker JARVIS pueda reutilizarlo tal cual.
"""

import time
from collections import OrderedDict

# Ritmo normal: un cliente pide un forward por token generado, con varios
# saltos, asi que se deja holgura. Lo que se corta es el spam de miles/segundo.
DEFAULT_RATE = 25.0        # peticiones por segundo sostenidas
DEFAULT_BURST = 100.0      # rafaga permitida

# /attest carga capas y ejecuta inferencia: es caro y no necesita ritmo alto.
ATTEST_RATE = 0.2          # una cada 5 segundos
ATTEST_BURST = 3.0

MAX_TRACKED_CLIENTS = 4096


class TokenBucket:
    """Cubo de tokens clasico: se rellena a `rate`/s hasta `burst`."""

    __slots__ = ("rate", "burst", "tokens", "updated_at")

    def __init__(self, rate: float, burst: float, now: float = None):
        self.rate = float(rate)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.updated_at = now if now is not None else time.monotonic()

    def consume(self, now: float, amount: float = 1.0) -> bool:
        elapsed = max(0.0, now - self.updated_at)
        self.updated_at = now
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after(self, amount: float = 1.0) -> float:
        """Segundos que faltan para volver a tener credito."""
        if self.rate <= 0:
            return 1.0
        missing = max(0.0, amount - self.tokens)
        return round(missing / self.rate, 3)


class RateLimiter:
    def __init__(self, rate: float = DEFAULT_RATE, burst: float = DEFAULT_BURST,
                 max_clients: int = MAX_TRACKED_CLIENTS, time_source=time.monotonic):
        self.rate = rate
        self.burst = burst
        self.max_clients = max(1, max_clients)
        self._time = time_source
        self._buckets = OrderedDict()

    def allow(self, identity: str, amount: float = 1.0):
        """Devuelve (permitido, segundos_de_espera)."""
        now = self._time()
        bucket = self._buckets.get(identity)
        if bucket is None:
            bucket = TokenBucket(self.rate, self.burst, now)
            self._buckets[identity] = bucket
            if len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(identity)

        if bucket.consume(now, amount):
            return True, 0.0
        return False, max(0.001, bucket.retry_after(amount))

    def reset(self, identity: str = None):
        if identity is None:
            self._buckets.clear()
        else:
            self._buckets.pop(identity, None)

    @property
    def tracked_clients(self) -> int:
        return len(self._buckets)
