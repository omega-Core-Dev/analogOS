"""
analogOS · primitives/broadcast.py
broadcast() — Fonte emite sinal; intensidade decai com distância.
Complexidade: O(k), k = entidades no mapa.

v0.2.0 — distâncias vetorizadas com numpy (elimina loop Python).
         Speedup esperado: 10-50x em datasets grandes.
"""

import numpy as np
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
    if falloff not in FALLOFF_MODES:
        raise ValueError(f"falloff deve ser um de {FALLOFF_MODES}")

    ids     = list(ref_map.keys())
    vectors = np.stack([ref_map[eid].vector for eid in ids])  # shape (n, dim)

    # ── distâncias vetorizadas — O(n) em numpy, sem loop Python ──────────────
    diff      = vectors - source.vector          # broadcast numpy
    distances = np.linalg.norm(diff, axis=1)     # shape (n,)

    d_max  = distances.max() or 1.0
    d_norm = distances / d_max                   # normaliza [0, 1]

    # ── falloff vetorizado ────────────────────────────────────────────────────
    if falloff == "quadratic":
        strengths = intensity / (distances ** 2 + 1)
    elif falloff == "sqrt":
        strengths = intensity * (1 - d_norm ** 0.5)
    else:  # linear
        strengths = intensity * (1 - d_norm)

    # ── noise opcional ────────────────────────────────────────────────────────
    if noise > 0:
        strengths = strengths + np.random.uniform(-noise, noise, size=len(ids))

    strengths = np.maximum(0.0, strengths)

    return dict(zip(ids, strengths.tolist()))
