# Biomorphic Memory System

**Brain-inspired memory architecture for AI agents — spreading activation, Q-value decay, and semantic graph retrieval.**

> 🏆 LongMemEval benchmark: **89.8% accuracy** (SOTA #1, surpassing EmergenceMem 86%)

---

## Why This Exists

Most agent memory systems treat memory as a flat vector store — embed, retrieve top-k, done. This misses how biological memory actually works:

- **Spreading activation** — related memories prime each other, surfacing contextually relevant information even without exact keyword matches
- **Utility decay** — memories that aren't accessed lose salience over time (forgetting curve)
- **Graph structure** — entities and experiences form a network, not a list

This system implements these principles as a practical memory layer for AI agents.

## Architecture

```
                    ┌──────────────────┐
 Query ────────────→│ Embedding Search │ (text-embedding-3-large)
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │    Spreading     │ ← Entity Graph
                    │   Activation    │   (763 nodes / 5066 edges)
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Personalized     │ ← Q-values from
                    │  PageRank (PPR)  │   usage feedback
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  Ranked Results  │ → Injected into agent context
                    └──────────────────┘
```

## Components

| Module | Description |
|--------|-------------|
| `src/graph_build.py` | Build entity-relation graph from markdown memory files |
| `src/spreading_activation.py` | Spreading activation over the memory graph |
| `src/q_value.py` | Q-value updates from usage feedback (success/partial/failure) |
| `src/utility.py` | Memory utility scoring and feedback tracking |
| `src/utility_rerank.py` | Re-rank memories by utility score |
| `src/embedding_daemon.py` | Background embedding service (Unix socket) |
| `recall/search.py` | End-to-end recall pipeline (embed → SA → PPR → rank) |
| `recall/extract_keywords.py` | Entity extraction for graph construction |

## Key Design Decisions

1. **Pure semantic pipeline** — No keyword/BM25/TF-IDF fallback. Handler was refactored from 762 → 326 lines by removing all non-semantic paths
2. **Cosine → SA → PPR** — Three-stage ranking combines embedding similarity, graph proximity, and personalized importance
3. **Feedback loop** — Every memory retrieval records outcome (success/partial/failure), updating Q-values for future ranking

## Usage

```python
from src.spreading_activation import spread
from src.graph_build import build_graph

# Build graph from memory files
graph = build_graph("path/to/memory/")

# Spread activation from seed nodes
activated = spread(
    graph=graph,
    seeds=["project_alpha", "agent_safety"],
    decay=0.5,
    max_depth=3
)

# Top activated nodes = contextually relevant memories
for node, score in activated[:10]:
    print(f"{node}: {score:.3f}")
```

## Benchmarks

**LongMemEval** (standardized long-term memory evaluation):
- Baseline (flat vector): 78.6%
- + Spreading Activation: 84.2%
- + PPR + Q-value: 87.1%
- + Model swap (doubao-pro): **89.8%** ← SOTA #1

**Distractor Robustness Evaluation**:
- D1-D2 (mild distractors): 95%+ accuracy
- D3 (adversarial): Exposed embedding source-authority weakness (100% confusion rate) — documented as known limitation

## License

Apache License 2.0 — See [LICENSE](LICENSE).

## Companion Projects

- [**Nous Safety**](https://github.com/dario-github/nous) — Ontology-driven runtime safety engine with Datalog reasoning
- [**Agent Self-Evolution**](https://github.com/dario-github/agent-self-evolution) — Automated evaluation, ablation testing, and improvement loops

## Install via ClawdHub

```bash
openclaw skills install biomorphic-memory
```

