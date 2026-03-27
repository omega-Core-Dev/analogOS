"""
analogOS · primitives/compose.py
compose() — Agrega sinais do cluster em saída unificada.
Complexidade: O(k).
"""

from analog_core.entity import Entity

COMPOSE_MODES = ("weighted_sum", "vote", "top_k", "concat")


def compose(
    cluster: list[Entity],
    signals: dict[str, float],
    mode: str = "weighted_sum",
    top_k: int = 3,
) -> dict:
    """
    Parâmetros
    ----------
    cluster : entidades do cluster (candidatos + propagados acima de threshold)
    signals : sinais finais após propagate()
    mode    : modo de composição
              'weighted_sum' → Σ(sinal_i · valor_i) — padrão ML
              'vote'         → maioria ponderada por sinal — consenso
              'top_k'        → k entidades mais fortes — retrieval
              'concat'       → concatenação ordenada por sinal — simbólico
    top_k   : usado apenas no modo 'top_k'

    Retorna
    -------
    Output dict com:
        'mode'   : modo usado
        'result' : saída composta
        'ranked' : lista [(entity, signal)] ordenada por sinal desc
    """
    if mode not in COMPOSE_MODES:
        raise ValueError(f"mode deve ser um de {COMPOSE_MODES}")

    ranked = sorted(cluster, key=lambda e: signals.get(e.id, 0.0), reverse=True)

    if mode == "weighted_sum":
        total_signal = sum(signals.get(e.id, 0.0) for e in ranked) or 1.0
        result = sum(
            signals.get(e.id, 0.0) / total_signal * (e.value if isinstance(e.value, (int, float)) else 1.0)
            for e in ranked
        )

    elif mode == "vote":
        votes: dict = {}
        for e in ranked:
            v = str(e.value)
            votes[v] = votes.get(v, 0.0) + signals.get(e.id, 0.0)
        result = max(votes, key=votes.get)

    elif mode == "top_k":
        result = [(e, signals.get(e.id, 0.0)) for e in ranked[:top_k]]

    else:  # concat
        result = [e.value for e in ranked]

    return {
        "mode": mode,
        "result": result,
        "ranked": [(e, signals.get(e.id, 0.0)) for e in ranked],
    }
