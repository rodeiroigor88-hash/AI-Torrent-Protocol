"""Verificacion redundante cruzada con atribucion de culpa (CAPA 3).

La auditoria por muestreo de la CAPA anterior repite un paso por UNA ruta
alternativa y compara. Tiene dos agujeros que este modulo cierra:

1. **Atribucion.** Con solo dos opiniones que discrepan es imposible saber cual
   mintio: la culpa recaia sobre la ruta primaria entera, castigando a nodos
   honestos. Con TRES o mas ejecuciones redundantes e independientes, el
   resultado minoritario delata al tramposo (voto por mayoria), asumiendo que
   los honestos son mayoria -- que es justo el caso del tramposo a tasa baja.

2. **El tramposo de tasa baja / Sybil nuevo.** Un muestreo de probabilidad fija
   rara vez pilla a quien hace trampa el 5% de las veces, y un Sybil recien
   llegado empieza con credibilidad gratis. El :class:`AdaptiveSampler` sube la
   probabilidad de auditoria para nodos NUEVOS o ya SOSPECHOSOS, de modo que
   ganarse la confianza cuesta tiempo y hacer trampa sale caro.

Este modulo es LOGICA PURA: opera sobre tensores de salida ya calculados, sin
red ni modelo, para poder probarse de forma determinista.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from src.pow_utils import DEFAULT_EPSILON, outputs_match

# Etiqueta de quien produjo un resultado (un node_id, o una tupla de node_ids si
# la opinion viene de una ruta entera). Debe ser hashable para deduplicar rutas.
Label = Hashable
LabeledResult = Tuple[Label, Any]


class VerdictStatus(str, Enum):
    """Resultado de comparar un conjunto de ejecuciones redundantes."""

    UNANIMOUS = "unanimous"    # todas coinciden: nadie sospechoso
    MAJORITY = "majority"      # hay una mayoria clara; el resto es sospechoso
    TIE = "tie"                # sin mayoria: no se puede atribuir (inconcluso)
    INSUFFICIENT = "insufficient"  # menos de 2 opiniones distintas: nada que cruzar


@dataclass
class Verdict:
    """Dictamen de una verificacion cruzada.

    :param status: ver :class:`VerdictStatus`.
    :param trusted: etiquetas que forman la mayoria (se consideran correctas).
    :param suspect: etiquetas fuera de la mayoria (candidatas a cuarentena).
    :param majority_size: tamano del grupo mayoritario.
    :param total: numero de opiniones distintas consideradas.
    """

    status: VerdictStatus
    trusted: List[Label] = field(default_factory=list)
    suspect: List[Label] = field(default_factory=list)
    majority_size: int = 0
    total: int = 0

    @property
    def attributable(self) -> bool:
        """True si el dictamen senala a un culpable concreto."""
        return self.status is VerdictStatus.MAJORITY and bool(self.suspect)


def _dedupe(results: Sequence[LabeledResult]) -> List[LabeledResult]:
    """Descarta opiniones de la MISMA etiqueta: dos ejecuciones del mismo nodo no
    son independientes y no deben contar como dos votos."""
    seen: set = set()
    unique: List[LabeledResult] = []
    for label, output in results:
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, output))
    return unique


def attribute(results: Sequence[LabeledResult], epsilon: float = DEFAULT_EPSILON) -> Verdict:
    """Agrupa ejecuciones redundantes por acuerdo y atribuye la culpa al outlier.

    El agrupamiento es voraz (la tolerancia L2 no es perfectamente transitiva):
    cada resultado entra en el primer grupo cuyo representante iguala dentro de
    ``epsilon``, o abre un grupo nuevo. Gana el grupo mas grande; si ninguno pasa
    de la mitad, el dictamen es TIE (no se puede atribuir con seguridad y hay que
    escalar a una opinion mas).

    :param results: pares ``(etiqueta, tensor_de_salida)``. Etiquetas repetidas se
        colapsan (no son opiniones independientes).
    """
    unique = _dedupe(results)
    total = len(unique)
    if total < 2:
        return Verdict(VerdictStatus.INSUFFICIENT, total=total)

    clusters: List[Dict[str, Any]] = []
    for label, output in unique:
        for cluster in clusters:
            if outputs_match(cluster["repr"], output, epsilon):
                cluster["labels"].append(label)
                break
        else:
            clusters.append({"repr": output, "labels": [label]})

    clusters.sort(key=lambda c: len(c["labels"]), reverse=True)
    largest = clusters[0]
    majority_size = len(largest["labels"])

    if len(clusters) == 1:
        return Verdict(VerdictStatus.UNANIMOUS, trusted=list(largest["labels"]),
                       suspect=[], majority_size=majority_size, total=total)

    # Mayoria estricta: mas de la mitad en un solo grupo. Un empate (p.ej. 2 vs 2,
    # o 1/1/1) no permite decidir quien miente sin arriesgarse a castigar honestos.
    if majority_size * 2 > total:
        suspect = [label for cluster in clusters[1:] for label in cluster["labels"]]
        return Verdict(VerdictStatus.MAJORITY, trusted=list(largest["labels"]),
                       suspect=suspect, majority_size=majority_size, total=total)

    return Verdict(VerdictStatus.TIE, majority_size=majority_size, total=total)


# --------------------------------------------------------------- muestreo

# Probabilidades por defecto del muestreo adaptativo (0..1).
BASE_PROBABILITY = 0.05        # nodo consolidado y sin incidencias
NEW_NODE_PROBABILITY = 0.5     # nodo aun sin historial suficiente: se vigila mucho
SUSPECT_PROBABILITY = 1.0      # nodo con una discrepancia reciente: se comprueba siempre
MAX_PROBABILITY = 1.0
# Cuantas comprobaciones superadas hacen falta para dejar de ser "nuevo".
TRUST_THRESHOLD = 20


@dataclass
class _NodeStats:
    checks: int = 0
    agreements: int = 0
    disagreements: int = 0
    consecutive_agreements: int = 0


class AdaptiveSampler:
    """Decide cuando auditar un paso, sesgando el muestreo hacia lo dudoso.

    Mantiene estadistica LOCAL por nodo (el cliente ve por que rutas paso y si
    la comprobacion cuadro), asi que no depende de la reputacion del tracker.
    La probabilidad de auditar un paso es la del nodo MAS sospechoso de su ruta:
    una cadena es tan fiable como su eslabon mas debil.
    """

    def __init__(
        self,
        *,
        base_probability: float = BASE_PROBABILITY,
        new_node_probability: float = NEW_NODE_PROBABILITY,
        suspect_probability: float = SUSPECT_PROBABILITY,
        max_probability: float = MAX_PROBABILITY,
        trust_threshold: int = TRUST_THRESHOLD,
    ) -> None:
        # Monotonia obligatoria: un nodo nuevo o sospechoso nunca debe auditarse
        # MENOS que uno consolidado, aunque el usuario configure una base alta.
        self.base_probability = base_probability
        self.new_node_probability = max(new_node_probability, base_probability)
        self.suspect_probability = max(suspect_probability, self.new_node_probability)
        self.max_probability = max_probability
        self.trust_threshold = trust_threshold
        self._stats: Dict[Label, _NodeStats] = {}

    def node_probability(self, node_id: Label) -> float:
        """Probabilidad de auditar un paso que atraviesa ``node_id``."""
        stats = self._stats.get(node_id)
        if stats is None or stats.checks == 0:
            # Nodo desconocido: maxima vigilancia hasta que se gane la confianza.
            return min(self.max_probability, self.new_node_probability)
        if stats.disagreements > 0 and stats.consecutive_agreements < self.trust_threshold:
            # Discrepo y aun no ha encadenado suficientes aciertos para redimirse.
            return min(self.max_probability, self.suspect_probability)
        if stats.agreements < self.trust_threshold:
            return min(self.max_probability, self.new_node_probability)
        return self.base_probability

    def step_probability(self, node_ids: Sequence[Label]) -> float:
        """Probabilidad de auditar un paso: la del nodo mas sospechoso de la ruta."""
        candidates = [n for n in node_ids if n]
        if not candidates:
            return 0.0
        return max(self.node_probability(node_id) for node_id in candidates)

    def should_audit(self, node_ids: Sequence[Label],
                     rng: Optional[random.Random] = None) -> bool:
        """Lanza el dado (sesgado) para decidir si auditar este paso."""
        probability = self.step_probability(node_ids)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        roll = (rng or random).random()
        return roll < probability

    def record_result(self, node_id: Label, agreed: bool) -> None:
        """Actualiza la estadistica de un nodo tras una comprobacion cruzada."""
        stats = self._stats.get(node_id)
        if stats is None:
            stats = _NodeStats()
            self._stats[node_id] = stats
        stats.checks += 1
        if agreed:
            stats.agreements += 1
            stats.consecutive_agreements += 1
        else:
            stats.disagreements += 1
            stats.consecutive_agreements = 0

    def record_route(self, node_ids: Sequence[Label], agreed: bool) -> None:
        """Aplica :meth:`record_result` a cada nodo de una ruta."""
        for node_id in node_ids:
            if node_id:
                self.record_result(node_id, agreed)

    def stats_for(self, node_id: Label) -> Dict[str, int]:
        """Instantanea de la estadistica de un nodo (para logs y pruebas)."""
        stats = self._stats.get(node_id) or _NodeStats()
        return {
            "checks": stats.checks,
            "agreements": stats.agreements,
            "disagreements": stats.disagreements,
            "consecutive_agreements": stats.consecutive_agreements,
        }
