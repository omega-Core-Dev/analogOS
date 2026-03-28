"""
analogOS · benchmark.py
Mede: precisão do top-k, velocidade do pipeline, impacto do filtro de etiqueta.

Roda no Colab ou local:
    python benchmark.py
"""
from __future__ import annotations

import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Entity:
    id: str
    value: Any
    vector: np.ndarray = field(default_factory=lambda: np.zeros(8))
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def distance_to(self, other: Entity) -> float:
        if self.vector.shape != other.vector.shape:
            return 0.0
        return float(np.linalg.norm(self.vector - other.vector))

# ── Primitivas inline ─────────────────────────────────────────────────────────
def scan(entities, key_fn=None):
    if key_fn is None:
        key_fn = lambda e: e.id
    return {key_fn(e): e for e in entities}

def broadcast(source, ref_map, intensity=1.0, falloff='sqrt', noise=0.0):
    """Original — loop Python."""
    distances = {eid: source.distance_to(e) for eid, e in ref_map.items()}
    d_max = max(distances.values()) or 1.0
    signals = {}
    for eid, dist in distances.items():
        d_norm = dist / d_max
        if falloff == 'quadratic':
            strength = intensity / (dist ** 2 + 1)
        elif falloff == 'sqrt':
            strength = intensity * (1 - (d_norm ** 0.5))
        else:
            strength = intensity * (1 - d_norm)
        if noise > 0:
            strength += random.uniform(-noise, noise)
        signals[eid] = max(0.0, strength)
    return signals

def broadcast_v2(source, ref_map, intensity=1.0, falloff='sqrt', noise=0.0):
    """v0.2.0 — distancias vetorizadas com numpy. Sem loop Python."""
    ids     = list(ref_map.keys())
    vectors = np.stack([ref_map[eid].vector for eid in ids])
    diff      = vectors - source.vector
    distances = np.linalg.norm(diff, axis=1)
    d_max   = distances.max() or 1.0
    d_norm  = distances / d_max
    if falloff == 'quadratic':
        strengths = intensity / (distances ** 2 + 1)
    elif falloff == 'sqrt':
        strengths = intensity * (1 - d_norm ** 0.5)
    else:
        strengths = intensity * (1 - d_norm)
    if noise > 0:
        strengths = strengths + np.random.uniform(-noise, noise, size=len(ids))
    strengths = np.maximum(0.0, strengths)
    return dict(zip(ids, strengths.tolist()))

def candidate(entities, signals, threshold=0.4, source_tags=None):
    def tag_ok(e):
        if source_tags is None:
            return True
        return any(t in source_tags for t in e.tags)
    eligible = [e for e in entities
                if signals.get(e.id, 0.0) >= threshold and tag_ok(e)]
    return sorted(eligible, key=lambda e: signals[e.id], reverse=True)

def propagate(candidates, all_entities, signals, social_factor=0.5, radius=None):
    propagated = dict(signals)
    candidate_ids = {e.id for e in candidates}
    for cand in candidates:
        cand_signal = signals.get(cand.id, 0.0)
        for target in all_entities:
            if target.id in candidate_ids:
                continue
            dist = cand.distance_to(target)
            if radius is not None and dist > radius:
                continue
            repasse = cand_signal * social_factor / (dist + 1)
            propagated[target.id] = propagated.get(target.id, 0.0) + repasse
    return propagated

def compose(cluster, signals, mode='top_k', top_k=3):
    ranked = sorted(cluster, key=lambda e: signals.get(e.id, 0.0), reverse=True)
    if mode == 'top_k':
        result = [(e, signals.get(e.id, 0.0)) for e in ranked[:top_k]]
    elif mode == 'vote':
        votes = {}
        for e in ranked:
            v = str(e.value)
            votes[v] = votes.get(v, 0.0) + signals.get(e.id, 0.0)
        result = max(votes, key=votes.get)
    else:
        result = [e.value for e in ranked]
    return {'mode': mode, 'result': result, 'ranked': [(e, signals.get(e.id, 0.0)) for e in ranked]}

# ══════════════════════════════════════════════════════════════════════════════
# GERADOR DE DATASET SINTÉTICO
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_TAGS = {
    'cognitivo':  ['conceito', 'cognitivo', 'mente', 'aprendizado'],
    'ML':         ['ML', 'matematica', 'algoritmo', 'modelo'],
    'biologico':  ['biologia', 'celula', 'organismo', 'planta'],
    'alimentar':  ['fruta', 'alimento', 'comida', 'nutricao'],
    'fisico':     ['fisica', 'energia', 'frequencia', 'onda'],
}

DOMAIN_VECTORS = {
    'cognitivo': np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
    'ML':        np.array([0.8, 0.7, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]),
    'biologico': np.array([0.1, 0.1, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1]),
    'alimentar': np.array([0.1, 0.1, 0.8, 0.9, 0.1, 0.1, 0.1, 0.1]),
    'fisico':    np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6]),
}

def generate_dataset(n_per_domain=20, noise=0.15, seed=42):
    np.random.seed(seed)
    entities = []
    for domain, base_vec in DOMAIN_VECTORS.items():
        tags = DOMAIN_TAGS[domain]
        for i in range(n_per_domain):
            vec = base_vec + np.random.normal(0, noise, size=8)
            vec = np.clip(vec, 0, 1)
            eid = f"{domain}_{i}"
            entities.append(Entity(
                id=eid,
                value=f"{domain}_token_{i}",
                vector=vec,
                tags=[tags[i % len(tags)]],
            ))
    return entities

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1 — PRECISÃO (com vs sem filtro de etiqueta)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_precision(entities, n_queries=50):
    print("\n" + "="*60)
    print("BENCHMARK 1 — PRECISÃO: com vs sem filtro de etiqueta")
    print("="*60)

    np.random.seed(99)
    query_domain = 'cognitivo'
    query_tags = DOMAIN_TAGS[query_domain][:2]
    query_vec = DOMAIN_VECTORS[query_domain] + np.random.normal(0, 0.05, 8)

    correct_with    = 0
    correct_without = 0
    irrelevant_with    = 0
    irrelevant_without = 0

    relevant_domains = {'cognitivo', 'ML'}

    for i in range(n_queries):
        noise = np.random.normal(0, 0.08, 8)
        query = Entity(
            id=f'query_{i}',
            value='query cognitiva',
            vector=np.clip(query_vec + noise, 0, 1),
            tags=query_tags,
        )

        out_with, _, _    = run_pipeline(query, entities, use_tag_filter=True)
        out_without, _, _ = run_pipeline(query, entities, use_tag_filter=False)

        for e, _ in out_with['result']:
            domain = e.id.split('_')[0]
            if domain in relevant_domains:
                correct_with += 1
            else:
                irrelevant_with += 1

        for e, _ in out_without['result']:
            domain = e.id.split('_')[0]
            if domain in relevant_domains:
                correct_without += 1
            else:
                irrelevant_without += 1

    total_with    = correct_with + irrelevant_with
    total_without = correct_without + irrelevant_without

    prec_with    = correct_with / total_with * 100 if total_with else 0
    prec_without = correct_without / total_without * 100 if total_without else 0

    print(f"\n  {'':30s} {'COM filtro':>12} {'SEM filtro':>12}")
    print(f"  {'Resultados relevantes':30s} {correct_with:>12} {correct_without:>12}")
    print(f"  {'Resultados irrelevantes':30s} {irrelevant_with:>12} {irrelevant_without:>12}")
    print(f"  {'Precisão (%)':30s} {prec_with:>11.1f}% {prec_without:>11.1f}%")
    print(f"\n  Ganho de precisão com filtro: +{prec_with - prec_without:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2 — VELOCIDADE por tamanho de dataset
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_speed():
    print("\n" + "="*60)
    print("BENCHMARK 2 — VELOCIDADE por tamanho de dataset")
    print("="*60)

    sizes = [10, 50, 100, 500, 1000]
    query = Entity(
        id='speed_query',
        value='speed test',
        vector=np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        tags=['conceito', 'cognitivo'],
    )

    print(f"\n  {'Entidades':>10} {'Tempo (ms)':>12} {'ms/entidade':>14}")
    print(f"  {'-'*40}")

    for size in sizes:
        entities = generate_dataset(n_per_domain=size // 5 or 1)
        entities = entities[:size]

        runs = 10
        start = time.perf_counter()
        for _ in range(runs):
            run_pipeline(query, entities, use_tag_filter=True)
        elapsed = (time.perf_counter() - start) / runs * 1000

        print(f"  {size:>10} {elapsed:>11.2f}ms {elapsed/size:>13.4f}ms")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3 — IMPACTO DOS PARÂMETROS
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_params(entities):
    print("\n" + "="*60)
    print("BENCHMARK 3 — IMPACTO DOS PARÂMETROS no cluster")
    print("="*60)

    query = Entity(
        id='param_query',
        value='param test',
        vector=np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        tags=['conceito', 'cognitivo'],
    )

    configs = [
        {'threshold': 0.2, 'social_factor': 0.3, 'label': 'threshold=0.2 social=0.3'},
        {'threshold': 0.4, 'social_factor': 0.5, 'label': 'threshold=0.4 social=0.5'},
        {'threshold': 0.6, 'social_factor': 0.7, 'label': 'threshold=0.6 social=0.7'},
        {'threshold': 0.8, 'social_factor': 0.9, 'label': 'threshold=0.8 social=0.9'},
    ]

    print(f"\n  {'Config':35s} {'Candidatos':>12} {'Cluster':>10} {'Top-1':>20}")
    print(f"  {'-'*80}")

    for cfg in configs:
        out, cands, cluster = run_pipeline(
            query, entities,
            threshold=cfg['threshold'],
            social_factor=cfg['social_factor'],
            use_tag_filter=True,
            top_k=1,
        )
        top1 = out['result'][0][0].value if out['result'] else 'none'
        print(f"  {cfg['label']:35s} {len(cands):>12} {len(cluster):>10} {top1:>20}")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4 — RADIUS no propagate: velocidade e impacto no cluster
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_radius():
    print("\n" + "="*60)
    print("BENCHMARK 4 — RADIUS no propagate: velocidade vs cobertura")
    print("="*60)

    sizes       = [100, 500, 1000]
    radius_vals = [None, 1.5, 1.0, 0.5]
    runs        = 5

    query = Entity(
        id='radius_query',
        value='radius test',
        vector=np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        tags=['conceito', 'cognitivo'],
    )

    print(f"\n  {'Entidades':>10} {'Radius':>8} {'Tempo (ms)':>12} {'Candidatos':>12} {'Cluster':>10} {'Speedup':>10}")
    print(f"  {'-'*68}")

    for size in sizes:
        entities = generate_dataset(n_per_domain=size // 5 or 1)[:size]
        baseline_time = None

        for radius in radius_vals:
            start = time.perf_counter()
            for _ in range(runs):
                out, cands, cluster = run_pipeline(
                    query, entities,
                    threshold=0.4,
                    social_factor=0.5,
                    use_tag_filter=True,
                    top_k=3,
                    radius=radius,
                )
            elapsed = (time.perf_counter() - start) / runs * 1000

            if radius is None:
                baseline_time = elapsed
                speedup_str = "baseline"
            else:
                speedup = baseline_time / elapsed if elapsed > 0 else 0
                speedup_str = f"{speedup:.1f}x"

            radius_str = "None" if radius is None else str(radius)
            print(f"  {size:>10} {radius_str:>8} {elapsed:>11.2f}ms {len(cands):>12} {len(cluster):>10} {speedup_str:>10}")

        print(f"  {'-'*68}")


def run_pipeline(query, entities, threshold=0.4, use_tag_filter=True,
                 social_factor=0.5, top_k=5, radius=None):
    ref_map = scan(entities)
    signals_bc = broadcast(query, ref_map, intensity=1.0, falloff='sqrt')
    source_tags = query.tags if use_tag_filter else None
    candidates = candidate(entities, signals_bc, threshold=threshold,
                           source_tags=source_tags)
    signals_prop = propagate(candidates, entities, signals_bc,
                             social_factor=social_factor, radius=radius)
    half_thresh = threshold / 2
    candidate_ids = {e.id for e in candidates}
    propagated_extras = [
        e for e in entities
        if e.id not in candidate_ids
        and signals_prop.get(e.id, 0.0) >= half_thresh
        and (source_tags is None or any(t in source_tags for t in e.tags))
    ]
    cluster = candidates + propagated_extras
    output = compose(cluster, signals_prop, mode='top_k', top_k=top_k)
    return output, candidates, cluster


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4 — RADIUS no propagate: velocidade vs cobertura
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_radius():
    print("\n" + "="*60)
    print("BENCHMARK 4 — RADIUS no propagate: velocidade vs cobertura")
    print("="*60)

    sizes = [100, 500, 1000]
    radii = [None, 1.5, 1.0, 0.5]
    runs  = 5

    query = Entity(
        id='radius_query',
        value='radius test',
        vector=np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        tags=['conceito', 'cognitivo'],
    )

    for size in sizes:
        entities = generate_dataset(n_per_domain=max(size // 5, 1))[:size]
        print(f"\n  Dataset: {size} entidades")
        print(f"  {'Radius':>18} {'Tempo (ms)':>12} {'Speedup':>10} {'Cluster medio':>15}")
        print(f"  {'-'*58}")

        baseline_ms = None

        for radius in radii:
            cluster_sizes = []
            start = time.perf_counter()
            for _ in range(runs):
                ref_map      = scan(entities)
                signals_bc   = broadcast(query, ref_map, intensity=1.0, falloff='sqrt')
                source_tags  = query.tags
                cands        = candidate(entities, signals_bc, threshold=0.4,
                                         source_tags=source_tags)
                signals_prop = propagate(cands, entities, signals_bc,
                                         social_factor=0.5, radius=radius)
                half_thresh  = 0.2
                cand_ids     = {e.id for e in cands}
                extras       = [
                    e for e in entities
                    if e.id not in cand_ids
                    and signals_prop.get(e.id, 0.0) >= half_thresh
                    and any(t in source_tags for t in e.tags)
                ]
                cluster_sizes.append(len(cands) + len(extras))

            elapsed_ms = (time.perf_counter() - start) / runs * 1000

            if baseline_ms is None:
                baseline_ms  = elapsed_ms
                speedup_str  = "baseline"
            else:
                speedup      = baseline_ms / elapsed_ms
                speedup_str  = f"{speedup:.2f}x"

            radius_label = "None (sem limite)" if radius is None else str(radius)
            avg_cluster  = sum(cluster_sizes) / len(cluster_sizes)
            print(f"  {radius_label:>18} {elapsed_ms:>11.2f}ms {speedup_str:>10} {avg_cluster:>15.1f}")

    print(f"\n  Conclusao: radius menor = menos vizinhos = mais rapido.")
    print(f"  Trade-off: cluster menor pode perder contexto periferico relevante.")


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 5 — ESCALA: 1k → 10k entidades
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_scale():
    print("\n" + "="*60)
    print("BENCHMARK 5 — ESCALA: 1k a 10k entidades")
    print("="*60)

    sizes = [1000, 2000, 5000, 10000]
    runs  = 3

    query = Entity(
        id='scale_query',
        value='scale test',
        vector=np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        tags=['conceito', 'cognitivo'],
    )

    print(f"\n  {'Entidades':>10} {'Tempo (ms)':>12} {'ms/entidade':>14} {'Candidatos':>12} {'Cluster':>10} {'Precisao':>10}")
    print(f"  {'-'*72}")

    relevant_domains = {'cognitivo', 'ML'}

    for size in sizes:
        n_per = max(size // len(DOMAIN_TAGS), 1)
        entities = []
        for domain, base_vec in DOMAIN_VECTORS.items():
            tags = DOMAIN_TAGS[domain]
            for i in range(n_per):
                vec = base_vec + np.random.normal(0, 0.15, 8)
                vec = np.clip(vec, 0, 1)
                entities.append(Entity(
                    id=f"{domain}_{i}",
                    value=f"{domain}_token_{i}",
                    vector=vec,
                    tags=[tags[i % len(tags)]],
                ))
        entities = entities[:size]

        total_relevant   = 0
        total_irrelevant = 0
        cluster_total    = 0
        cands_total      = 0

        start = time.perf_counter()
        for _ in range(runs):
            out, cands, cluster = run_pipeline(
                query, entities,
                threshold=0.4,
                use_tag_filter=True,
                top_k=5,
            )
            cands_total  += len(cands)
            cluster_total += len(cluster)
            for e, _ in out['result']:
                domain = e.id.split('_')[0]
                if domain in relevant_domains:
                    total_relevant += 1
                else:
                    total_irrelevant += 1

        elapsed_ms  = (time.perf_counter() - start) / runs * 1000
        avg_cands   = cands_total / runs
        avg_cluster = cluster_total / runs
        total       = total_relevant + total_irrelevant
        precisao    = total_relevant / total * 100 if total else 0

        print(f"  {size:>10} {elapsed_ms:>11.1f}ms {elapsed_ms/size:>13.4f}ms {avg_cands:>12.1f} {avg_cluster:>10.1f} {precisao:>9.1f}%")

    print(f"\n  Observacao: crescimento esperado O(n) no broadcast.")
    print(f"  Precisao deve se manter 100% com filtro de etiqueta ativo.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 6 — broadcast v1 (loop) vs v2 (vetorizado)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_broadcast_v2():
    print("\n" + "="*60)
    print("BENCHMARK 6 — broadcast: loop Python vs numpy vetorizado")
    print("="*60)

    sizes = [100, 500, 1000, 5000, 10000]
    runs  = 5

    query = Entity(
        id='bv2_query',
        value='broadcast v2 test',
        vector=np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        tags=['conceito', 'cognitivo'],
    )

    print(f"\n  {'Entidades':>10} {'v1 loop (ms)':>14} {'v2 numpy (ms)':>15} {'Speedup':>10}")
    print(f"  {'-'*54}")

    for size in sizes:
        n_per    = max(size // len(DOMAIN_TAGS), 1)
        entities = []
        for domain, base_vec in DOMAIN_VECTORS.items():
            tags = DOMAIN_TAGS[domain]
            for i in range(n_per):
                vec = base_vec + np.random.normal(0, 0.15, 8)
                vec = np.clip(vec, 0, 1)
                entities.append(Entity(
                    id=f"{domain}_{i}",
                    value=f"{domain}_token_{i}",
                    vector=vec,
                    tags=[tags[i % len(tags)]],
                ))
        entities = entities[:size]
        ref_map  = scan(entities)

        # v1 — loop
        start = time.perf_counter()
        for _ in range(runs):
            broadcast(query, ref_map)
        t_v1 = (time.perf_counter() - start) / runs * 1000

        # v2 — vetorizado
        start = time.perf_counter()
        for _ in range(runs):
            broadcast_v2(query, ref_map)
        t_v2 = (time.perf_counter() - start) / runs * 1000

        speedup = t_v1 / t_v2 if t_v2 > 0 else 0
        print(f"  {size:>10} {t_v1:>13.2f}ms {t_v2:>14.2f}ms {speedup:>9.1f}x")

    print(f"\n  Conclusao: broadcast_v2 elimina loop Python — ganho cresce com dataset.")

if __name__ == '__main__':
    print("analogOS · Benchmark Suite · v0.2.0")
    print("Gerando dataset sintetico...")
    entities = generate_dataset(n_per_domain=20)
    print(f"Dataset: {len(entities)} entidades | {len(DOMAIN_TAGS)} dominios")

    benchmark_precision(entities, n_queries=50)
    benchmark_speed()
    benchmark_params(entities)
    benchmark_radius()
    benchmark_scale()
    benchmark_broadcast_v2()

    print("\n" + "="*60)
    print("Benchmark concluido.")
    print("="*60)
