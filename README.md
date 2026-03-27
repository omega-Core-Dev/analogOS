# analogOS

**Programmable Analogy Framework**

> Analogy is not a way to explain systems.

> Analogy *is* the system.

---

## ⚡ Core Idea

All complex systems share the same computational structure:

scan → broadcast → candidate → propagate → compose

---

🧠 What is it?

analogOS is a framework that formalizes analogy as a computational primitive.

It's not machine learning.

It's not just abstraction.

It's a systems algebra based on reusable universal operators.

---

⚙️ System Core

The framework is built on 5 primitives:

scan → broadcast → candidate → propagate → compose

Each represents a fundamental step in any system where information flows.

---

🔍 Primitives

Primitive | Function

 scan| percorre entidades e cria mapa
broadcast| emite sinal a partir de uma fonte
candidate| filtra por threshold
propagate| espalha sinal entre vizinhos
compose| gera saída final

---

🌍 Ideia Central

Sistemas diferentes compartilham o mesmo padrão estrutural:

- redes neurais
- mercados financeiros
- sistema imunológico
- blockchain

analogOS unifica todos como instâncias do mesmo pipeline.

---

🔁 Exemplo (Atenção)

scan       → tokens
broadcast  → query
candidate  → keys ≥ threshold
propagate  → values → contexto
compose    → weighted sum

---

🧩 Estrutura do Projeto

analogOS/
├── analog_core/
│   ├── primitives/
│   │   ├── scan.py
│   │   ├── broadcast.py
│   │   ├── candidate.py
│   │   ├── propagate.py
│   │   └── compose.py
│   ├── entity.py
│   └── pipeline.py
│
├── domains/
│   ├── analog_attention/
│   ├── analog_neuro/
│   ├── analog_market/
│   ├── analog_immune/
│   └── analog_blockchain/
│
├── extensions/
├── docs/
├── tests/
└── analogos.toml

---

⚙️ Configuração ("analogos.toml")

[project]
name = "analogOS"
version = "0.1.0"
license = "GPL-3.0"
description = "Framework de Analogia Programável"

[core]
operators = ["scan", "broadcast", "candidate", "propagate", "compose"]

[pipeline]
flow = ["scan", "broadcast", "candidate", "propagate", "compose"]

[primitives.scan]
complexity = "O(n)"

[primitives.broadcast]
falloff_modes = ["linear", "sqrt", "quadratic"]

[primitives.candidate]
parameter = "threshold"

[primitives.propagate]
parameters = ["social_factor", "radius"]

[primitives.compose]
modes = ["weighted_sum", "vote", "top_k", "concat"]

---

🏗️ Arquitetura

analogOS funciona como um framework raiz:

- núcleo genérico ("analog_core")
- instâncias por domínio ("domains")
- extensões ("extensions")

Cada domínio apenas parametriza as mesmas primitivas.

---

🚀 Roadmap

- v0.1.0 — núcleo funcional ✅
- v0.2.0 — pipeline completo
- v0.3.0 — múltiplos domínios
- v0.4.0 — integrações (PyTorch / JAX)
- v1.0.0 — publicação + paper

---

📜 Licença

GPL-3.0

---

⚡ Visão

analogOS propõe que:

«sistemas diferentes são instâncias do mesmo padrão computacional.»

---

🧠 Autor

Zaqueu Ribeiro
