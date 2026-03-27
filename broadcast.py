"""
analogOS · primitives/broadcast.py
broadcast() — Fonte emite sinal; intensidade decai com distância.
Complexidade: O(k), k = entidades no mapa.
"""

from analog_core.entity import Entity

FALLOFF_MODES = ("sqrt", "linear", "quadratic")


def broadcast(
    source: Entity,
    ref_map: dict[str, Entity],
    intensity: float = 1.0,
    falloff: str = "sqrt",
    noise: float = 0.0,
) -> dict[str, float]:
    """
    Parâmetros
    ----------
    source    : entidade emissora do sinal
    ref_map   : mapa construído por scan()
    intensity : força inicial do sinal [0, ∞)
    falloff   : modo de decaimento — 'sqrt' | 'linear' | 'quadratic'
    noise     : perturbação aleatória [0, 1)

    Retorna
    -------
    signals : { entity_id → signal_strength }
    """
    import random

    if falloff not in FALLOFF_MODES:
        raise ValueError(f"falloff deve ser um de {FALLOFF_MODES}")

    signals: dict[str, float] = {}

    distances = {eid: source.distance_to(e) for eid, e in ref_map.items()}
    d_max = max(distances.values()) or 1.0  # evita divisão por zero

    for eid, dist in distances.items():
        d_norm = dist / d_max  # normaliza para [0, 1]

        if falloff == "quadratic":
            strength = intensity / (dist ** 2 + 1)
        elif falloff == "sqrt":
            strength = intensity * (1 - (d_norm ** 0.5))
        else:  # linear
            strength = intensity * (1 - d_norm)

        if noise > 0:
            strength += random.uniform(-noise, noise)

        signals[eid] = max(0.0, strength)

    return signals
