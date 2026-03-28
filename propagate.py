"""
analogOS · primitives/propagate.py
propagate() — Candidatos repassam sinal para vizinhos.
Formação emergente de clusters.

v0.2.0 — vetorizado com numpy.
         Elimina loop Python duplo O(k·n) — speedup esperado 10-50x.
"""

import numpy as np
from analog_core.entity import Entity


def propagate(
    candidates: list[Entity],
    all_entities: list[Entity],
    signals: dict[str, float],
    social_factor: float = 0.5,
    radius: float | None = None,
) -> dict[str, float]:
    """
    Parâmetros
    ----------
    candidates    : entidades elegíveis de candidate()
    all_entities  : espaço completo de entidades
    signals       : sinais atuais (base para o repasse)
    social_factor : eficiência de repasse [0, 1]
    radius        : distância máxima de vizinhança
                    None → sem limite

    Retorna
    -------
    propagated : { entity_id → signal_strength }
    """
    if not candidates:
        return dict(signals)

    candidate_ids  = {e.id for e in candidates}

    # separa targets — entidades que NÃO são candidatos
    targets        = [e for e in all_entities if e.id not in candidate_ids]

    if not targets:
        return dict(signals)

    # ── matrizes numpy ────────────────────────────────────────────────────────
    cand_vectors   = np.stack([e.vector for e in candidates])   # (k, dim)
    target_vectors = np.stack([e.vector for e in targets])      # (t, dim)
    cand_signals   = np.array([signals.get(e.id, 0.0)
                                for e in candidates])           # (k,)

    # ── distâncias: (k, t) — cada candidato vs cada target ───────────────────
    # broadcast numpy: (k, 1, dim) - (1, t, dim) → (k, t, dim)
    diff      = cand_vectors[:, np.newaxis, :] - target_vectors[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)                    # (k, t)

    # ── máscara de radius ─────────────────────────────────────────────────────
    if radius is not None:
        mask = distances <= radius                               # (k, t) bool
    else:
        mask = np.ones_like(distances, dtype=bool)

    # ── repasse vetorizado ────────────────────────────────────────────────────
    # repasse[k, t] = cand_signal[k] * social_factor / (dist[k,t] + 1)
    repasse = (cand_signals[:, np.newaxis] * social_factor
               / (distances + 1))                               # (k, t)

    repasse = repasse * mask                                     # aplica radius

    # soma contribuições de todos os candidatos para cada target
    total_repasse = repasse.sum(axis=0)                         # (t,)

    # ── monta dict final ──────────────────────────────────────────────────────
    propagated = dict(signals)
    for i, target in enumerate(targets):
        propagated[target.id] = (propagated.get(target.id, 0.0)
                                 + float(total_repasse[i]))

    return propagated
