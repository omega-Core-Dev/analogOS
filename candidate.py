"""
analogOS · primitives/candidate.py
candidate() — Filtra entidades que receberam sinal suficiente.
Complexidade: O(k).
"""

from analog_core.entity import Entity


def candidate(
    entities: list[Entity],
    signals: dict[str, float],
    threshold: float = 0.4,
) -> list[Entity]:
    """
    Parâmetros
    ----------
    entities  : lista completa de entidades
    signals   : mapa { entity_id → signal_strength } de broadcast()
    threshold : sinal mínimo para ser elegível [0, 1]
                0 → todos candidatos (broadcast puro)
                1 → ninguém candidato (sinal fica na fonte)

    Retorna
    -------
    Lista de Entity onde signals[e.id] >= threshold, ordenada por sinal desc.
    """
    eligible = [
        e for e in entities
        if signals.get(e.id, 0.0) >= threshold
    ]
    return sorted(eligible, key=lambda e: signals[e.id], reverse=True)
