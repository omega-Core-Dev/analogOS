"""
analogOS · primitives/propagate.py
propagate() — Candidatos repassam sinal para vizinhos.
Formação emergente de clusters.
Complexidade: O(k·r), r = vizinhos no raio.
"""

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
                    None → sem limite (cuidado: O(n²))

    Retorna
    -------
    propagated : { entity_id → signal_strength } — sinais após propagação
                 entidades já candidatas mantêm seu sinal original
    """
    propagated: dict[str, float] = dict(signals)  # copia base

    candidate_ids = {e.id for e in candidates}

    for cand in candidates:
        cand_signal = signals.get(cand.id, 0.0)

        for target in all_entities:
            if target.id in candidate_ids:
                continue  # candidatos não recebem repasse entre si

            dist = cand.distance_to(target)

            if radius is not None and dist > radius:
                continue  # fora do raio de vizinhança

            repasse = cand_signal * social_factor / (dist + 1)
            propagated[target.id] = propagated.get(target.id, 0.0) + repasse

    return propagated
