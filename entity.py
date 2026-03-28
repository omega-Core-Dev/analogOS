"""
analogOS · analog_core/entity.py
Entidade base do framework — unidade fundamental do espaço analógico.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """
    Unidade fundamental do analogOS.

    Parâmetros
    ----------
    id      : identificador único da entidade
    value   : dado associado (token, número, string, qualquer coisa)
    vector  : representação vetorial no espaço analógico
    tags    : etiquetas semânticas — filtro antes do cálculo vetorial
    meta    : metadados opcionais de domínio
    """

    id: str
    value: Any
    vector: np.ndarray = field(default_factory=lambda: np.zeros(8))
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def distance_to(self, other: Entity) -> float:
        """
        Distância euclidiana entre dois vetores.
        Retorna 0.0 se os vetores tiverem dimensões diferentes.
        """
        if self.vector.shape != other.vector.shape:
            return 0.0
        return float(np.linalg.norm(self.vector - other.vector))

    def similarity_to(self, other: Entity) -> float:
        """
        Similaridade cosseno entre dois vetores — [0, 1].
        Complementar à distância: quanto maior, mais similar.
        """
        a, b = self.vector, other.vector
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def __repr__(self) -> str:
        return f"Entity(id={self.id!r}, value={self.value!r}, tags={self.tags})"
