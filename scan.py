"""
analogOS · primitives/scan.py
scan() — Percorre o espaço de entidades e constrói mapa de referência.
Complexidade: O(n) — pago uma vez, reutilizado n vezes.
"""

from typing import Callable
from analog_core.entity import Entity


ReferenceMap = dict[str, Entity]


def scan(
    entities: list[Entity],
    key_fn: Callable[[Entity], str] | None = None,
) -> ReferenceMap:
    """
    Parâmetros
    ----------
    entities : lista de Entity do domínio
    key_fn   : função opcional que extrai a chave do mapa (default: entity.id)

    Retorna
    -------
    ReferenceMap : { chave → Entity }
    """
    if key_fn is None:
        key_fn = lambda e: e.id

    return {key_fn(e): e for e in entities}
