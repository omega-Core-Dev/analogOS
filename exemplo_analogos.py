"""
analogOS · exemplo end-to-end
Demonstração completa do pipeline: scan → broadcast → candidate → propagate → compose

Rode direto no Colab:
  !pip install numpy
  # cole este arquivo e execute

Ou localmente:
  python exemplo_analogos.py
"""

import numpy as np
import sys
import os

# ── imports do framework ────────────────────────────────────────────────────
# ajusta path para rodar standalone (sem instalar o pacote)
sys.path.insert(0, os.path.dirname(__file__))

from entity import Entity
from scan import scan
from broadcast import broadcast
from candidate import candidate
from propagate import propagate
from compose import compose


# ══════════════════════════════════════════════════════════════════════════════
# DOMÍNIO: Atenção Semântica (analog-attention)
# Pergunta: "o que é aprendizado?"
# Memória: tokens com vetores semânticos
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("analogOS · Pipeline Completo · Domínio: Atenção Semântica")
print("=" * 60)

# ── 1. Criar espaço de entidades (tokens na memória) ────────────────────────

np.random.seed(42)

# tokens com vetores 4D — simplificado para legibilidade
tokens = [
    Entity(
        id="t1", value="aprendizado",
        vector=np.array([0.9, 0.8, 0.1, 0.2]),
        tags=["conceito", "cognitivo"]
    ),
    Entity(
        id="t2", value="rede neural",
        vector=np.array([0.8, 0.7, 0.2, 0.3]),
        tags=["conceito", "ML"]
    ),
    Entity(
        id="t3", value="gradiente",
        vector=np.array([0.7, 0.5, 0.3, 0.4]),
        tags=["matematica", "ML"]
    ),
    Entity(
        id="t4", value="memória",
        vector=np.array([0.6, 0.9, 0.1, 0.1]),
        tags=["conceito", "cognitivo"]
    ),
    Entity(
        id="t5", value="banana",
        vector=np.array([0.1, 0.1, 0.9, 0.8]),
        tags=["fruta", "alimento"]
    ),
    Entity(
        id="t6", value="fotossíntese",
        vector=np.array([0.0, 0.2, 0.8, 0.9]),
        tags=["biologia", "planta"]
    ),
]

# ── query: entidade fonte que faz a pergunta ─────────────────────────────────
query = Entity(
    id="query",
    value="o que é aprendizado?",
    vector=np.array([0.85, 0.75, 0.15, 0.25]),
    tags=["query"]
)

print(f"\n📡 Query: '{query.value}'")
print(f"   Vetor: {query.vector}")

# ── PASSO 1: SCAN ─────────────────────────────────────────────────────────────
print("\n── SCAN ─────────────────────────────────────────────────────")
ref_map = scan(tokens)
print(f"   Mapa construído: {list(ref_map.keys())} ({len(ref_map)} entidades)")

# ── PASSO 2: BROADCAST ───────────────────────────────────────────────────────
print("\n── BROADCAST ────────────────────────────────────────────────")
signals_bc = broadcast(
    source=query,
    ref_map=ref_map,
    intensity=1.0,
    falloff="sqrt",
    noise=0.0,
)
for eid, sig in sorted(signals_bc.items(), key=lambda x: -x[1]):
    token = ref_map[eid]
    bar = "█" * int(sig * 20)
    print(f"   {eid} ({token.value:15s}) sinal={sig:.3f}  {bar}")

# ── PASSO 3: CANDIDATE ───────────────────────────────────────────────────────
print("\n── CANDIDATE (threshold=0.4) ────────────────────────────────")
candidates = candidate(entities=tokens, signals=signals_bc, threshold=0.4)
if candidates:
    for e in candidates:
        print(f"   ✅ {e.id} — '{e.value}' (sinal={signals_bc[e.id]:.3f})")
else:
    print("   Nenhum candidato acima do threshold.")

# ── PASSO 4: PROPAGATE ───────────────────────────────────────────────────────
print("\n── PROPAGATE (social_factor=0.5) ────────────────────────────")
signals_prop = propagate(
    candidates=candidates,
    all_entities=tokens,
    signals=signals_bc,
    social_factor=0.5,
    radius=None,
)
for eid, sig in sorted(signals_prop.items(), key=lambda x: -x[1]):
    token = ref_map[eid]
    delta = sig - signals_bc.get(eid, 0.0)
    delta_str = f"(+{delta:.3f})" if delta > 0.001 else ""
    print(f"   {eid} ({token.value:15s}) sinal={sig:.3f} {delta_str}")

# ── CLUSTER ──────────────────────────────────────────────────────────────────
half_thresh = 0.4 / 2
candidate_ids = {e.id for e in candidates}
propagated_extras = [
    e for e in tokens
    if e.id not in candidate_ids
    and signals_prop.get(e.id, 0.0) >= half_thresh
]
cluster = candidates + propagated_extras
print(f"\n   Cluster final: {[e.value for e in cluster]}")

# ── PASSO 5: COMPOSE ─────────────────────────────────────────────────────────
print("\n── COMPOSE (modo: top_k=3) ──────────────────────────────────")
output = compose(cluster=cluster, signals=signals_prop, mode="top_k", top_k=3)
print("   Top-3 entidades mais relevantes para a query:\n")
for i, (entity, sig) in enumerate(output["result"], 1):
    print(f"   {i}. '{entity.value}' — sinal={sig:.3f} | tags={entity.tags}")

# ── RESUMO ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Pipeline concluído.")
print(f"Query  : '{query.value}'")
print(f"Resposta (top-3): {[e.value for e, _ in output['result']]}")
print("=" * 60)
