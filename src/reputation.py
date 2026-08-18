"""Reputacion y scoring de nodos para el enrutamiento inteligente (CAPA 2).

Separa la POLITICA (como se puntua y penaliza un nodo) de la MECANICA del tracker
(registro, expulsion de fantasmas, firma de rutas). Asi el algoritmo de seleccion
se prueba de forma aislada y determinista, sin levantar el servidor HTTP.

El score combina las tres senales pedidas por la matriz de la CAPA 2::

    Score = f(Latencia_Ping, Historial_Anomalias, Ancho_Banda_Disponible)

* **Latencia**: mas baja -> mejor. Se normaliza contra un techo de referencia.
* **Anomalias**: cada divergencia/timeout/formato-corrupto resta, pero el peso
  DECAE con el tiempo para que un nodo pueda REHABILITARSE si vuelve a portarse
  bien (no un veto permanente).
* **Ancho de banda**: se aproxima por los recursos donados (nucleos + RAM),
  porque un nodo con mas capacidad sostiene mas tokens por segundo.

Penalizacion automatica (Slash / Quarantine): un nodo que devuelve formatos
corruptos o supera el timeout critico entra en cuarentena por un tiempo que crece
exponencialmente con la reincidencia. Mientras esta en cuarentena NO se enruta
hacia el; al expirar vuelve, pero con el score penalizado hasta que acumule
exitos y el historial de anomalias decaiga.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# --- Motivos de penalizacion -------------------------------------------------
REASON_DIVERGENCE = "divergence"   # la auditoria detecto una salida alterada
REASON_TIMEOUT = "timeout"         # el nodo no respondio a tiempo
REASON_CORRUPT = "corrupt"         # devolvio un formato ilegible
VALID_REASONS = frozenset({REASON_DIVERGENCE, REASON_TIMEOUT, REASON_CORRUPT})

# --- Cuarentena --------------------------------------------------------------
# Backoff exponencial: 1a anomalia -> BASE, 2a -> 2*BASE, ... con techo en MAX.
QUARANTINE_BASE_SECONDS = 60.0
QUARANTINE_MAX_SECONDS = 3600.0

# --- Decaimiento del historial de anomalias ----------------------------------
# Vida media: cada ANOMALY_HALF_LIFE_SECONDS el peso efectivo de las anomalias
# pasadas se reduce a la mitad. Sin esto un nodo penalizado nunca se recuperaria.
ANOMALY_HALF_LIFE_SECONDS = 1800.0

# --- Latencia EWMA -----------------------------------------------------------
# Peso de la ultima muestra frente al historico (0..1). Mas alto = mas reactivo.
LATENCY_EWMA_ALPHA = 0.3

# --- Normalizacion del score -------------------------------------------------
REFERENCE_LATENCY_MS = 250.0   # latencia "buena" de referencia para normalizar
REFERENCE_CORES = 4.0          # nucleos donados que ya cuentan como "capaz"
REFERENCE_RAM_MB = 4096.0      # RAM donada que ya cuenta como "capaz"

# Pesos de las tres componentes (deben sumar 1.0).
WEIGHT_LATENCY = 0.4
WEIGHT_REPUTATION = 0.4
WEIGHT_BANDWIDTH = 0.2


def quarantine_backoff(anomalies: int) -> float:
    """Duracion de cuarentena, en segundos, para la ``n``-esima anomalia.

    Crece exponencialmente con la reincidencia y se acota en
    :data:`QUARANTINE_MAX_SECONDS` para no expulsar a un nodo para siempre por un
    mal dia.
    """
    if anomalies <= 0:
        return 0.0
    grown = QUARANTINE_BASE_SECONDS * (2 ** (anomalies - 1))
    return float(min(grown, QUARANTINE_MAX_SECONDS))


@dataclass
class NodeReputation:
    """Historial de comportamiento de un nodo, persistente entre latidos.

    Vive fuera de ``Tracker.nodes`` (que se poda cuando el nodo deja de latir)
    para que un tramposo no borre su historial simplemente reapareciendo con el
    mismo ``node_id`` tras una expulsion.
    """

    node_id: str
    successes: int = 0
    anomalies: int = 0
    last_anomaly_at: float = 0.0
    quarantined_until: float = 0.0
    latency_ms: Optional[float] = None
    last_reason: str = ""

    # ------------------------------------------------------------ señales

    def observe_latency(self, sample_ms: float) -> None:
        """Integra una muestra de latencia en la media exponencial (EWMA)."""
        if sample_ms is None or sample_ms < 0:
            return
        if self.latency_ms is None:
            self.latency_ms = float(sample_ms)
        else:
            self.latency_ms = (LATENCY_EWMA_ALPHA * float(sample_ms)
                               + (1 - LATENCY_EWMA_ALPHA) * self.latency_ms)

    def record_success(self) -> None:
        """Registra un paso servido correctamente (mejora la reputacion)."""
        self.successes += 1

    def record_anomaly(self, reason: str, now: float) -> None:
        """Registra una anomalia y pone al nodo en cuarentena con backoff.

        :param reason: uno de :data:`VALID_REASONS`; cualquier otro se normaliza a
            ``divergence`` para no perder la senal.
        :param now: instante actual (se inyecta para poder probar con reloj falso).
        """
        self.anomalies += 1
        self.last_anomaly_at = now
        self.last_reason = reason if reason in VALID_REASONS else REASON_DIVERGENCE
        self.quarantined_until = now + quarantine_backoff(self.anomalies)

    # ------------------------------------------------------------ consultas

    def is_quarantined(self, now: float) -> bool:
        """True si el nodo esta en cuarentena en el instante ``now``."""
        return now < self.quarantined_until

    def effective_anomalies(self, now: float) -> float:
        """Peso de las anomalias tras aplicar el decaimiento por vida media.

        Un nodo que lleva mucho sin fallar ve su penalizacion tender a cero, de
        modo que puede recuperar un buen score.
        """
        if self.anomalies <= 0:
            return 0.0
        elapsed = max(0.0, now - self.last_anomaly_at)
        decay = 0.5 ** (elapsed / ANOMALY_HALF_LIFE_SECONDS)
        return self.anomalies * decay


def _latency_component(latency_ms: Optional[float]) -> float:
    """Componente de latencia normalizada a (0, 1]; latencia desconocida = neutra."""
    if latency_ms is None or latency_ms < 0:
        return 0.5  # sin datos: ni premia ni castiga
    return REFERENCE_LATENCY_MS / (REFERENCE_LATENCY_MS + latency_ms)


def _reputation_component(effective_anomalies: float) -> float:
    """Componente de reputacion en (0, 1]: cae con las anomalias efectivas."""
    return 1.0 / (1.0 + effective_anomalies)


def _bandwidth_component(donated_cores, donated_ram_mb) -> float:
    """Componente de capacidad en [0, 1] a partir de los recursos donados.

    Recursos desconocidos cuentan como neutros (0.5) para no penalizar a un nodo
    legitimo que simplemente no los declaro.
    """
    if not donated_cores and not donated_ram_mb:
        return 0.5
    cores = min(1.0, (donated_cores or 0) / REFERENCE_CORES)
    ram = min(1.0, (donated_ram_mb or 0) / REFERENCE_RAM_MB)
    return (cores + ram) / 2.0


def score_node(
    rep: Optional[NodeReputation],
    donated_cores,
    donated_ram_mb,
    now: float,
) -> float:
    """Puntua un nodo combinando latencia, reputacion y ancho de banda.

    Devuelve un valor en ``[0, 1]`` (mayor = mejor candidato). Un nodo en
    cuarentena obtiene ``0.0`` para que quede el ultimo aunque no haya sido
    excluido todavia por el filtro de rutas.

    :param rep: historial del nodo, o None si aun no tiene (se trata como neutro).
    :param now: instante actual (inyectable para pruebas deterministas).
    """
    if rep is not None and rep.is_quarantined(now):
        return 0.0

    latency_ms = rep.latency_ms if rep is not None else None
    effective = rep.effective_anomalies(now) if rep is not None else 0.0

    return (
        WEIGHT_LATENCY * _latency_component(latency_ms)
        + WEIGHT_REPUTATION * _reputation_component(effective)
        + WEIGHT_BANDWIDTH * _bandwidth_component(donated_cores, donated_ram_mb)
    )
