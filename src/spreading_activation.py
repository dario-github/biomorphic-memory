#!/usr/bin/env python3
"""Spreading Activation Engine — Phase 2 of Biomorphic Memory System.

Implements:
- Pure embedding-based anchor selection
- 3-step spreading activation with fan effect
- Lateral inhibition
- Personalized PageRank (HippoRAG-inspired)
- Triple hybrid scoring: semantic + activation + PPR
- Uncertainty gating

Usage:
    python3 src/spreading_activation.py "query text"

    # With options:
    python3 src/spreading_activation.py "query" --top-k 15 --verbose

Environment:
    MEMORY_DIR              Path to memory directory (default: ./memory)
    DATA_DIR                Path to data directory   (default: ./data)
    OPENAI_API_KEY          API key for embeddings
    OPENAI_BASE_URL         Optional custom endpoint
    EMBEDDING_DAEMON_SOCK   Unix socket for the embedding daemon
    BIOMORPHIC_ENABLED      Set to 1 to enable (default: pass-through)

Output: JSON array of {node_id, path, score, activation, semantic_sim, pagerank}
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────

WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).parent.parent))
MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", WORKSPACE / "memory"))
DATA_DIR = Path(os.environ.get("DATA_DIR", WORKSPACE / "data"))
GRAPH_FILE = Path(os.environ.get("GRAPH_FILE", DATA_DIR / "memory-graph.json"))
Q_VALUES_FILE = Path(os.environ.get("Q_VALUES_FILE", DATA_DIR / "memory-q-values.json"))
EMBEDDING_CACHE_FILE = Path(os.environ.get("EMBEDDING_CACHE_FILE", DATA_DIR / "embedding-cache.json"))
EMBEDDING_DAEMON_SOCK = os.environ.get("EMBEDDING_DAEMON_SOCK", "/tmp/embedding-daemon.sock")

# Spreading activation parameters
S = float(os.environ.get("SA_SPREADING_RATE", "0.8"))     # spreading rate
DELTA = float(os.environ.get("SA_RETENTION_RATE", "0.5")) # retention rate
T_STEPS = int(os.environ.get("SA_STEPS", "3"))            # propagation iterations
BETA = float(os.environ.get("SA_INHIBITION", "0.15"))     # lateral inhibition strength
M_TOP = int(os.environ.get("SA_TOP_M", "5"))              # top-M for inhibition
GAMMA = 5.0     # sigmoid steepness
THETA = 0.5     # sigmoid threshold

# Scoring weights
LAMBDA_1 = 0.38   # semantic similarity
LAMBDA_2 = 0.285  # activation
LAMBDA_3 = 0.285  # PageRank
LAMBDA_4 = 0.05   # emotional weight

# Gating
TAU_GATE = float(os.environ.get("SA_UNCERTAINTY_GATE", "0.15"))
TIMEOUT_MS = int(os.environ.get("SA_TIMEOUT_MS", "500"))

# Retrieval
K1 = int(os.environ.get("SA_K1", "10"))
K_FINAL = int(os.environ.get("SA_K_FINAL", "15"))

# Protected node types (reduced inhibition)
PROTECTED_PREFIXES = tuple(os.environ.get("SA_PROTECTED_PREFIXES", "rules/,plans/").split(","))

# Q-value lambda bounds
Q_LAMBDA_MIN = 0.1
Q_LAMBDA_MAX = 0.3
Q_N_THRESHOLD = 30

# Personalized PageRank
PPR_ALPHA = 0.5
PPR_ITERATIONS = 20


# ── Graph Loading ─────────────────────────────────────────────────────

def load_graph():
    """Load graph from JSON. Returns (nodes_dict, adj_list, pagerank_dict)."""
    if not GRAPH_FILE.exists():
        return None, None, None

    data = json.loads(GRAPH_FILE.read_text())
    nodes = {n["id"]: n for n in data["nodes"]}

    adj = {nid: [] for nid in nodes}
    for e in data["edges"]:
        if e["from"] in adj:
            adj[e["from"]].append((e["to"], e["weight"]))
        if e["to"] in adj:
            adj[e["to"]].append((e["from"], e["weight"]))

    pagerank = {n["id"]: n.get("pagerank", 0.0) for n in data["nodes"]}
    return nodes, adj, pagerank


def load_q_values():
    """Load Q-values."""
    if Q_VALUES_FILE.exists():
        try:
            return json.loads(Q_VALUES_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


# ── Embedding Search ──────────────────────────────────────────────────

_embedding_cache: dict | None = None


def load_embedding_cache() -> dict[str, np.ndarray]:
    """Load pre-computed embeddings from cache file."""
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache

    _embedding_cache = {}
    if EMBEDDING_CACHE_FILE.exists():
        try:
            raw = json.loads(EMBEDDING_CACHE_FILE.read_text())
            _embedding_cache = {
                k: np.array(v["vec"], dtype=np.float32)
                for k, v in raw.items()
                if not k.startswith("__") and isinstance(v, dict) and "vec" in v
            }
        except Exception:
            pass
    return _embedding_cache


def get_query_embedding(query: str) -> np.ndarray | None:
    """Get query embedding: local daemon (fast) → API fallback (slow)."""
    import socket as sock_mod

    if os.path.exists(EMBEDDING_DAEMON_SOCK):
        try:
            s = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
            s.settimeout(10.0)
            s.connect(EMBEDDING_DAEMON_SOCK)
            s.sendall(json.dumps({"texts": [query[:500]]}).encode())
            s.shutdown(sock_mod.SHUT_WR)
            data = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
            s.close()
            resp = json.loads(data)
            if "embeddings" in resp and resp["embeddings"]:
                return np.array(resp["embeddings"][0], dtype=np.float32)
        except Exception:
            pass

    # Fallback: remote API
    import requests
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
    if api_key:
        try:
            resp = requests.post(
                f"{base_url}/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "input": [query[:500]]},
                timeout=10,
            )
            if resp.status_code == 200:
                return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
        except Exception:
            pass

    return None


def embedding_search(query_vec: np.ndarray, nodes: dict, top_k: int = 10) -> list[tuple[str, float]]:
    """Find top-k most similar nodes by embedding cosine similarity."""
    cache = load_embedding_cache()
    if not cache or query_vec is None:
        return []

    scores = []
    qnorm = np.linalg.norm(query_vec)
    if qnorm == 0:
        return []

    qdim = len(query_vec)
    for nid in nodes:
        if nid not in cache:
            continue
        doc_vec = cache[nid]
        if len(doc_vec) != qdim:
            continue
        dnorm = np.linalg.norm(doc_vec)
        if dnorm == 0:
            continue
        sim = float(np.dot(query_vec, doc_vec) / (qnorm * dnorm))
        if sim > 0.2:
            scores.append((nid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── Anchor Selection ──────────────────────────────────────────────────

def select_anchors(query: str, nodes: dict) -> list[tuple[str, float]]:
    """Pure embedding-based anchor selection.

    Pipeline:
    1. Get query embedding via daemon or API fallback
    2. Cosine similarity against all node embeddings
    3. Return top-K with normalized scores
    """
    query_vec = get_query_embedding(query)
    if query_vec is None:
        return []

    results = embedding_search(query_vec, nodes, top_k=K1)
    if not results:
        return []

    max_score = results[0][1] if results else 1.0
    if max_score > 0:
        results = [(nid, score / max_score) for nid, score in results]

    return results


# ── Spreading Activation ─────────────────────────────────────────────

def spreading_activation(
    anchors: list[tuple[str, float]],
    nodes: dict,
    adj: dict,
    steps: int = T_STEPS,
) -> dict:
    """Run spreading activation. Returns node_id -> activation value."""
    activation = {nid: 0.0 for nid in nodes}
    for nid, score in anchors:
        if nid in activation:
            activation[nid] = score

    for t in range(steps):
        new_activation = {}

        for nid in nodes:
            retained = (1 - DELTA) * activation[nid]
            incoming = 0.0
            for neighbor_id, weight in adj.get(nid, []):
                fan_out = len(adj.get(neighbor_id, []))
                if fan_out == 0:
                    continue
                incoming += S * weight * activation[neighbor_id] / fan_out
            new_activation[nid] = retained + incoming

        # Lateral inhibition
        sorted_nodes = sorted(new_activation.items(), key=lambda x: x[1], reverse=True)
        top_m = sorted_nodes[:M_TOP]
        top_m_ids = set(nid for nid, _ in top_m)

        for nid in new_activation:
            if nid in top_m_ids:
                continue
            is_protected = any(nid.startswith(p) for p in PROTECTED_PREFIXES)
            effective_beta = BETA * 0.5 if is_protected else BETA
            inhibition = 0.0
            for top_id, top_val in top_m:
                if top_val > new_activation[nid]:
                    inhibition += (top_val - new_activation[nid])
            new_activation[nid] = max(0.0, new_activation[nid] - effective_beta * inhibition)

        # Sigmoid activation
        for nid in new_activation:
            u = new_activation[nid]
            new_activation[nid] = 1.0 / (1.0 + math.exp(-GAMMA * (u - THETA))) if u > 0.01 else 0.0

        activation = new_activation

    return activation


# ── Personalized PageRank ─────────────────────────────────────────────

def personalized_pagerank(
    seed_nodes: list[tuple[str, float]],
    nodes: dict,
    adj: dict,
) -> dict:
    """Run Personalized PageRank seeded from query-linked nodes.

    Unlike static PageRank (query-independent), PPR concentrates mass
    around the query's semantic neighborhood. This is inspired by
    HippoRAG's "hippocampal index" retrieval pattern.

    Args:
        seed_nodes: [(node_id, weight)] — query anchors with initial relevance
        adj: {node_id: [(neighbor_id, edge_weight), ...]}

    Returns:
        {node_id: ppr_score} — personalized relevance to this specific query
    """
    n = len(nodes)
    if n == 0:
        return {}

    node_ids = list(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    personalization = np.zeros(n, dtype=np.float64)
    for nid, weight in seed_nodes:
        if nid in id_to_idx:
            personalization[id_to_idx[nid]] = weight

    p_sum = personalization.sum()
    if p_sum > 0:
        personalization /= p_sum
    else:
        personalization[:] = 1.0 / n

    out_weights = {}
    for nid in node_ids:
        idx = id_to_idx[nid]
        neighbors = adj.get(nid, [])
        if not neighbors:
            continue
        total_w = sum(w for _, w in neighbors)
        if total_w > 0:
            out_weights[idx] = [
                (id_to_idx[nb], w / total_w)
                for nb, w in neighbors if nb in id_to_idx
            ]

    scores = personalization.copy()

    for _ in range(PPR_ITERATIONS):
        new_scores = np.zeros(n, dtype=np.float64)
        for idx in range(n):
            if idx not in out_weights:
                continue
            for nb_idx, w in out_weights[idx]:
                new_scores[nb_idx] += (1 - PPR_ALPHA) * scores[idx] * w

        new_scores += PPR_ALPHA * personalization

        dangling_mass = 0.0
        for idx in range(n):
            if idx not in out_weights:
                dangling_mass += scores[idx]
        new_scores += (1 - PPR_ALPHA) * dangling_mass * personalization

        s = new_scores.sum()
        if s > 0:
            new_scores /= s

        scores = new_scores

    return {node_ids[i]: float(scores[i]) for i in range(n)}


# ── Scoring ───────────────────────────────────────────────────────────

def hybrid_score(
    anchors: dict,
    activation: dict,
    pagerank: dict,
    q_values: dict,
    nodes: dict,
) -> list[dict]:
    """Compute final scores using quad hybrid + Q-value + emotional weight."""
    results = []

    for nid, node in nodes.items():
        sem = anchors.get(nid, 0.0)
        act = activation.get(nid, 0.0)
        pr = pagerank.get(nid, 0.0)
        emo = node.get("emotional_weight", 0.0)

        l2_score = LAMBDA_1 * sem + LAMBDA_2 * act + LAMBDA_3 * pr + LAMBDA_4 * emo

        if l2_score < 0.001:
            continue

        path = node.get("path", "")
        q_entry = q_values.get(path, {})
        q_val = q_entry.get("q", 0.5)
        q_n = q_entry.get("n", 0)

        if q_n < 10:
            q_lambda = Q_LAMBDA_MIN
        elif q_n >= Q_N_THRESHOLD:
            q_lambda = Q_LAMBDA_MAX
        else:
            q_lambda = Q_LAMBDA_MIN + (Q_LAMBDA_MAX - Q_LAMBDA_MIN) * (q_n - 10) / (Q_N_THRESHOLD - 10)

        final_score = (1 - q_lambda) * l2_score + q_lambda * q_val

        results.append({
            "node_id": nid,
            "path": path,
            "type": node.get("type", "unknown"),
            "score": round(final_score, 4),
            "l2_score": round(l2_score, 4),
            "semantic": round(sem, 4),
            "activation": round(act, 4),
            "pagerank": round(pr, 6),
            "q_value": round(q_val, 4),
            "q_n": q_n,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Main Pipeline ─────────────────────────────────────────────────────

def search(query: str, top_k: int = K_FINAL, verbose: bool = False) -> list[dict]:
    """Full search pipeline: embedding anchor → SA + PPR → hybrid scoring."""
    t_start = time.time()

    nodes, adj, pagerank = load_graph()
    if nodes is None:
        print("Error: No graph file. Run graph_build.py build first.", file=sys.stderr)
        return []

    q_values = load_q_values()

    # Step 1: Anchor selection (pure embedding)
    anchors = select_anchors(query, nodes)
    if verbose:
        print(f"Anchors: {len(anchors)}", file=sys.stderr)
        for nid, s in anchors[:5]:
            print(f"  {nid}: {s:.3f}", file=sys.stderr)

    anchor_dict = dict(anchors)

    # Step 2a: Spreading activation
    t_sa_start = time.time()
    activation = spreading_activation(anchors, nodes, adj)
    if verbose:
        print(f"Spreading activation: {(time.time()-t_sa_start)*1000:.0f}ms", file=sys.stderr)

    # Step 2b: Personalized PageRank
    t_ppr_start = time.time()
    ppr_scores = personalized_pagerank(anchors, nodes, adj)
    if verbose:
        print(f"PPR: {(time.time()-t_ppr_start)*1000:.0f}ms", file=sys.stderr)

    elapsed_ms = (time.time() - t_start) * 1000
    if elapsed_ms > TIMEOUT_MS and verbose:
        print(f"Warning: Total {elapsed_ms:.0f}ms > {TIMEOUT_MS}ms budget", file=sys.stderr)

    # Step 3: Four-way hybrid scoring
    results = hybrid_score(anchor_dict, activation, ppr_scores, q_values, nodes)

    # Step 4: Uncertainty gating
    if results and results[0]["score"] < TAU_GATE:
        if verbose:
            print(f"Uncertainty gate: top score {results[0]['score']:.4f} < {TAU_GATE}", file=sys.stderr)
        return [{"node_id": "__LOW_CONFIDENCE__", "path": "", "score": 0.0}]

    if verbose:
        print(f"Total: {(time.time()-t_start)*1000:.0f}ms, returning top {top_k}", file=sys.stderr)

    return results[:top_k]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    json_mode = "--json" in sys.argv
    top_k = K_FINAL

    if "--top-k" in sys.argv:
        idx = sys.argv.index("--top-k")
        if idx + 1 < len(sys.argv):
            top_k = int(sys.argv[idx + 1])

    results = search(query, top_k=top_k, verbose=verbose)

    if json_mode:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for r in results:
            if r.get("node_id") == "__LOW_CONFIDENCE__":
                print("⚠️  Low confidence — no relevant memory found")
            else:
                print(
                    f"{r['score']:.4f}  {r['node_id']} ({r['type']}) — "
                    f"sem:{r['semantic']:.3f} act:{r['activation']:.3f} "
                    f"pr:{r['pagerank']:.6f} q:{r['q_value']:.2f}"
                )
