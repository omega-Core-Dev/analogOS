# analogOS
### Programmable Analogy Framework · v0.1.0 · GPL-3.0

> *"Analogy is not a way to explain systems. Analogy **is** the system."*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omega-Core-Dev/analogOS/blob/main/analogOS_demo.ipynb)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## What is analogOS?

**analogOS** is a general-purpose framework that formalizes analogy as a programmable primitive.

It is not a machine learning library.  
It is not a visualization tool.  
It is not a set of didactic metaphors.

It is a **systems algebra** — a minimal set of operators that describes the behavior of any system where information is generated, distributed, filtered, propagated, and composed.

The central proposition:

> Different complex systems — neural networks, financial markets, immune systems, blockchains — are instances of the **same structural pattern**.

Once that pattern is formalized, implementations across any domain become parameterizations of the same five primitives.

---

## The Five Universal Primitives

```
scan → broadcast → candidate → propagate → compose
```

| Primitive | Role | Complexity |
|---|---|---|
| `scan()` | Traverses the entity space and builds a reference map | O(n) — paid once |
| `broadcast()` | Source emits a signal; intensity decays with distance | O(k) |
| `candidate()` | Filters entities that received sufficient signal | O(k) |
| `propagate()` | Candidates relay signal to neighbors — emergent clusters | O(k·r) |
| `compose()` | Aggregates the cluster into a unified output | O(k) |

Every domain supported by analogOS is a mapping of these five operators to specific parameters. **The code is the same — only the parameters change.**

---

## Quick Start

```python
import numpy as np
from entity import Entity
from scan import scan
from broadcast import broadcast
from candidate import candidate
from propagate import propagate
from compose import compose

# Define your entity space
tokens = [
    Entity('t1', 'learning',    np.array([0.9, 0.8, 0.1, 0.2]), tags=['concept']),
    Entity('t2', 'neural net',  np.array([0.8, 0.7, 0.2, 0.3]), tags=['ML']),
    Entity('t3', 'gradient',    np.array([0.7, 0.5, 0.3, 0.4]), tags=['math']),
    Entity('t4', 'banana',      np.array([0.1, 0.1, 0.9, 0.8]), tags=['food']),
]

# Define the query (signal source)
query = Entity('q', 'what is learning?', np.array([0.85, 0.75, 0.15, 0.25]))

# Run the pipeline
ref_map        = scan(tokens)
signals_bc     = broadcast(query, ref_map, intensity=1.0, falloff='sqrt')
candidates     = candidate(tokens, signals_bc, threshold=0.4)
signals_prop   = propagate(candidates, tokens, signals_bc, social_factor=0.5)

cluster = candidates + [
    e for e in tokens
    if e.id not in {c.id for c in candidates}
    and signals_prop.get(e.id, 0.0) >= 0.2
]

output = compose(cluster, signals_prop, mode='top_k', top_k=3)

# Result
print([e.value for e, _ in output['result']])
# → ['learning', 'neural net', 'gradient']
```

▶️ **Run the full interactive demo:** click the **Open in Colab** badge above.

---

## Domain Instances

Each domain below is a parameterization of the five primitives.

| Domain | scan | broadcast | candidate | propagate | compose |
|---|---|---|---|---|---|
| **analog-attention** (ML) | tokens | query vector | keys ≥ threshold | value context | weighted sum |
| **analog-neuro** (Neuroscience) | neurons | action potential | threshold neurons | synaptic relay | firing pattern |
| **analog-market** (Finance) | assets | price signal | eligible assets | market contagion | portfolio |
| **analog-immune** (Immunology) | antigens | receptor signal | activated cells | immune cascade | response |
| **analog-blockchain** (Consensus) | nodes | broadcast tx | validators | peer relay | block commit |

---

## Project Structure

```
analogOS/
├── analog_core/
│   ├── entity.py            ← base unit of the analogy space
│   └── primitives/
│       ├── scan.py
│       ├── broadcast.py
│       ├── candidate.py
│       ├── propagate.py
│       └── compose.py
│
├── pipeline.py              ← universal pipeline interface
│
├── domains/
│   ├── analog_attention/
│   ├── analog_neuro/
│   ├── analog_market/
│   ├── analog_immune/
│   └── analog_blockchain/
│
├── extensions/              ← PyTorch / JAX (v0.4.0)
├── tests/
├── analogOS_demo.ipynb      ← interactive Colab demo
└── analogos.toml
```

---

## Pipeline Interface

```python
from pipeline import Pipeline, PipelineConfig

pipeline = Pipeline(config=PipelineConfig(
    intensity=1.0,
    falloff='sqrt',        # 'sqrt' | 'linear' | 'quadratic'
    threshold=0.4,
    social_factor=0.5,
    mode='top_k',          # 'top_k' | 'weighted_sum' | 'vote' | 'concat'
    top_k=3,
))

result = pipeline.run(source=query, entities=tokens)
print(result.output)
```

---

## Roadmap

- [x] **v0.1.0** — Foundation: five primitives in pure Python + architecture doc
- [ ] **v0.2.0** — Complete core: formal coordinate map, unit tests, pipeline class
- [ ] **v0.3.0** — Domains: analog_blockchain, analog_neuro, analog_market, analog_immune
- [ ] **v0.4.0** — Integrations: PyTorch tensors, JAX, benchmark vs dot-product attention
- [ ] **v1.0.0** — PyPI release + technical paper

---

## Theoretical Foundation

Conventional use of analogy in computing is **didactic**: it explains a hard concept using a familiar domain, then discards the analogy.

**analogOS inverts this flow.**

The central claim is that the structure of the analogy *is* the structure of the system. The immune system is not "similar to" an attention mechanism. Both are instances of the same formal pattern:

```
scan → broadcast → candidate → propagate → compose
```

What does not exist in the literature is the formalization of analogy as a **reusable programmable primitive** — an operator that can be instantiated in any domain without rewriting the core logic. That is what analogOS proposes.

---

## License

GPL-3.0 — use, modify, and redistribute freely. Modified versions must maintain the same license. Source code must remain open.

---

## Author

**Zaqueu Ribeiro** · [github.com/omega-Core-Dev](https://github.com/omega-Core-Dev)

---

> *"The pattern was always there. It just needed a name."*
