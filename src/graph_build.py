#!/usr/bin/env python3
"""Memory Graph Builder — Phase 1 of Biomorphic Memory System.

Scans the memory directory, builds an Episodic-Semantic Graph with:
- Nodes: episodic (daily notes), semantic (entities), procedural (experiences/rules)
- Edges: temporal, abstraction, association, causal
- PageRank computation

Usage:
    python3 src/graph_build.py build          # Full rebuild
    python3 src/graph_build.py incremental    # Add new files only
    python3 src/graph_build.py stats          # Graph statistics
    python3 src/graph_build.py query <term>   # Find nodes matching term

Environment:
    MEMORY_DIR     Path to memory directory (default: ./memory)
    GRAPH_FILE     Output graph JSON file (default: ./data/memory-graph.json)
    OPENAI_API_KEY OpenAI API key for embeddings
    OPENAI_BASE_URL Optional custom OpenAI-compatible base URL

Output: data/memory-graph.json
"""

import json
import os
import re
import sys
import math
import hashlib
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx

# ── Constants ──────────────────────────────────────────────────────────

WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).parent.parent))
MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", WORKSPACE / "memory"))
DATA_DIR = Path(os.environ.get("DATA_DIR", WORKSPACE / "data"))
GRAPH_FILE = Path(os.environ.get("GRAPH_FILE", DATA_DIR / "memory-graph.json"))
Q_VALUES_FILE = Path(os.environ.get("Q_VALUES_FILE", DATA_DIR / "memory-q-values.json"))

MAX_ACTIVE_NODES = int(os.environ.get("MAX_ACTIVE_NODES", "2000"))
ASSOC_THRESHOLD = float(os.environ.get("ASSOC_THRESHOLD", "0.55"))
TEMPORAL_RHO = 0.02         # time decay rate (per day)
ABSTRACTION_W = 0.8         # default abstraction edge weight
MAX_EDGES_PER_NODE = 10     # graph density constraint
MAX_CONTENT_CHARS = 2000    # chars to read per file for embedding/matching
DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")

# ── Decay Parameters (Memory Decay) ──────────────────────────────────
DECAY_HALF_LIFE = {
    "episodic":   14,    # daily notes — 2 week half-life
    "semantic":   60,    # entities/research — 2 month half-life
    "procedural": 180,   # rules/experiences — 6 month half-life
}
ARCHIVE_DECAY = 0.05       # hard floor for archive/dead-files

# ── Emotional Markers (Emotional Tagging) ────────────────────────────
EMOTIONAL_MARKERS = [
    "[important]", "[lesson]", "[critical]", "🔥", "critical",
    "breakthrough", "[CRITICAL]", "⚠️", "key insight",
]

# Files that should never be dormant/deleted
PROTECTED_PATHS = {
    "memory/rules/",
}


# ── Node Type Classification ──────────────────────────────────────────

def classify_node(rel_path: str) -> str:
    """Classify a memory file into node type."""
    if DATE_PATTERN.match(Path(rel_path).name):
        return "episodic"
    if "/entities/" in rel_path:
        return "semantic"
    if "/experiences/" in rel_path or "/rules/" in rel_path:
        return "procedural"
    if "/research/" in rel_path:
        return "semantic"
    if "/plans/" in rel_path:
        return "semantic"
    return "semantic"


def extract_date(rel_path: str):
    """Extract date from daily note filename, or file mtime."""
    match = DATE_PATTERN.match(Path(rel_path).name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").isoformat()
        except ValueError:
            pass
    full_path = MEMORY_DIR / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
    if full_path.exists():
        mtime = os.path.getmtime(full_path)
        return datetime.fromtimestamp(mtime).isoformat()
    return datetime.now().isoformat()


def node_id_from_path(rel_path: str) -> str:
    """Create a stable node ID from relative path."""
    nid = rel_path
    if nid.startswith("memory/"):
        nid = nid[7:]
    if nid.endswith(".md"):
        nid = nid[:-3]
    return nid


# ── File Scanning ─────────────────────────────────────────────────────

def scan_memory_files():
    """Scan memory/ for all .md files, return list of (rel_path, content_preview)."""
    files = []
    for root, dirs, filenames in os.walk(MEMORY_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in filenames:
            if not fname.endswith(".md"):
                continue
            full_path = Path(root) / fname
            try:
                rel_path = str(full_path.relative_to(MEMORY_DIR.parent))
            except ValueError:
                rel_path = str(full_path)
            try:
                content = full_path.read_text(encoding="utf-8")[:MAX_CONTENT_CHARS]
            except (UnicodeDecodeError, PermissionError):
                content = ""
            files.append((rel_path, content))
    return files


# ── Embedding Support ────────────────────────────────────────────────

EMBEDDING_CACHE = DATA_DIR / "embedding-cache.json"
EMBEDDING_DAEMON_SOCK = os.environ.get("EMBEDDING_DAEMON_SOCK", "/tmp/embedding-daemon.sock")
EMBEDDING_BATCH_SIZE = 4

_inline_model = None


def _query_daemon(texts: list[str]) -> list[list[float]]:
    """Query the local embedding daemon via Unix socket."""
    import socket as sock_mod
    s = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
    s.settimeout(120.0)
    s.connect(EMBEDDING_DAEMON_SOCK)
    s.sendall(json.dumps({"texts": texts}).encode())
    s.shutdown(sock_mod.SHUT_WR)
    data = b""
    while True:
        chunk = s.recv(1024 * 1024)
        if not chunk:
            break
        data += chunk
    s.close()
    resp = json.loads(data)
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["embeddings"]


def _embed_inline(texts: list[str]) -> list[list[float]]:
    """Embed directly using fastembed (no daemon). Slower cold start, no socket issues."""
    global _inline_model
    if _inline_model is None:
        from fastembed import TextEmbedding
        _inline_model = TextEmbedding("jinaai/jina-embeddings-v3")
    return [e.tolist() for e in _inline_model.embed(texts)]


def _query_api(texts: list[str]) -> list[list[float]]:
    """Fallback: query remote OpenAI-compatible embedding API."""
    import requests
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
    resp = requests.post(
        f"{base_url}/embeddings",
        headers=headers,
        json={"model": model, "input": texts},
        timeout=30,
    )
    if resp.status_code == 200:
        return [item["embedding"] for item in resp.json()["data"]]
    raise RuntimeError(f"API error {resp.status_code}: {resp.text[:100]}")


def build_embeddings(nodes: list[dict], file_contents: dict) -> dict[str, np.ndarray]:
    """Build embeddings for all nodes. Prefers local daemon, falls back to API."""
    import hashlib as hl

    if os.path.exists(EMBEDDING_DAEMON_SOCK):
        try:
            _query_daemon(["test"])
            embed_fn = _query_daemon
            source_tag = "api"
            print(f"  Using daemon socket (text-embedding-3-large)", file=sys.stderr)
        except Exception as e:
            embed_fn = _query_api
            source_tag = "api"
            print(f"  Daemon failed ({e}), using direct API", file=sys.stderr)
    else:
        try:
            _query_api(["test"])
            embed_fn = _query_api
            source_tag = "api"
            print(f"  Using direct API", file=sys.stderr)
        except Exception:
            embed_fn = _embed_inline
            source_tag = "inline"
            print(f"  API unavailable, falling back to inline fastembed", file=sys.stderr)

    cache = {}
    if EMBEDDING_CACHE.exists():
        try:
            raw = json.loads(EMBEDDING_CACHE.read_text())
            cache_source = raw.get("__meta__", {}).get("source", "api")
            if cache_source == source_tag:
                for k, v in raw.items():
                    if k.startswith("__"):
                        continue
                    cache[k] = {"hash": v["hash"], "vec": np.array(v["vec"], dtype=np.float32)}
            else:
                print(f"  Cache source mismatch ({cache_source} vs {source_tag}), rebuilding", file=sys.stderr)
        except Exception:
            pass

    to_embed = []
    for n in nodes:
        text = file_contents.get(n["path"], "")[:MAX_CONTENT_CHARS]
        content_hash = hl.md5(text.encode()).hexdigest()[:12]
        if n["id"] in cache and cache[n["id"]]["hash"] == content_hash:
            continue
        to_embed.append((n["id"], text, content_hash))

    if to_embed:
        print(f"  Embedding {len(to_embed)} files via {source_tag} ({len(nodes) - len(to_embed)} cached)...", file=sys.stderr)
        for batch_start in range(0, len(to_embed), EMBEDDING_BATCH_SIZE):
            batch = to_embed[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
            texts = [t[1] if t[1].strip() else "empty document" for t in batch]
            try:
                embs = embed_fn(texts)
                for emb, (nid, _, chash) in zip(embs, batch):
                    vec = np.array(emb, dtype=np.float32)
                    cache[nid] = {"hash": chash, "vec": vec}
            except Exception as e:
                print(f"  ⚠️ Embedding batch failed: {e}", file=sys.stderr)

        serializable = {"__meta__": {"source": source_tag, "updated": datetime.now().isoformat()}}
        serializable.update({k: {"hash": v["hash"], "vec": v["vec"].tolist()} for k, v in cache.items()})
        EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EMBEDDING_CACHE.write_text(json.dumps(serializable))
        print(f"  Cache updated: {len(cache)} entries ({source_tag})", file=sys.stderr)
    else:
        print(f"  All {len(nodes)} embeddings cached, skip", file=sys.stderr)

    result = {}
    for n in nodes:
        if n["id"] in cache:
            result[n["id"]] = cache[n["id"]]["vec"]
    return result


def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


# ── Edge Construction ─────────────────────────────────────────────────

def build_temporal_edges(nodes: list[dict]) -> list[dict]:
    """Build temporal edges between consecutive daily notes."""
    edges = []
    episodic = [(n, n.get("timestamp", "")) for n in nodes if n["type"] == "episodic"]
    episodic.sort(key=lambda x: x[1])

    for i in range(len(episodic) - 1):
        n1, t1 = episodic[i]
        n2, t2 = episodic[i + 1]
        try:
            d1 = datetime.fromisoformat(t1)
            d2 = datetime.fromisoformat(t2)
            delta_days = (d2 - d1).days
        except (ValueError, TypeError):
            delta_days = 1

        weight = math.exp(-TEMPORAL_RHO * max(delta_days, 0))
        edges.append({
            "from": n1["id"], "to": n2["id"],
            "type": "temporal", "weight": round(weight, 4),
            "last_activation": datetime.now().isoformat(),
        })
    return edges


def build_abstraction_edges(nodes: list[dict], file_contents: dict) -> list[dict]:
    """Build abstraction edges: episodic node mentions semantic entity."""
    edges = []
    entity_names = {}
    for n in nodes:
        if n["type"] in ("semantic", "procedural"):
            name = n["id"].split("/")[-1].replace("-", " ").lower()
            if len(name) >= 2:
                entity_names[name] = n["id"]

    for n in nodes:
        if n["type"] != "episodic":
            continue
        content = file_contents.get(n["path"], "").lower()
        if not content:
            continue
        mention_count = 0
        for entity_name, entity_id in entity_names.items():
            if entity_name in content and mention_count < MAX_EDGES_PER_NODE:
                edges.append({
                    "from": n["id"], "to": entity_id,
                    "type": "abstraction", "weight": ABSTRACTION_W,
                    "last_activation": datetime.now().isoformat(),
                })
                mention_count += 1
    return edges


def build_association_edges(nodes: list[dict], embeddings: dict[str, np.ndarray], node_index: dict) -> list[dict]:
    """Build association edges using vectorized cosine similarity."""
    edges = []
    if not embeddings:
        return edges

    semantic_nodes = [n for n in nodes if n["type"] in ("semantic", "procedural") and n["id"] in embeddings]
    if len(semantic_nodes) < 2:
        return edges

    ids = [n["id"] for n in semantic_nodes]
    vecs = np.array([embeddings[nid] for nid in ids], dtype=np.float32)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    vecs_normed = vecs / norms
    sim_matrix = vecs_normed @ vecs_normed.T
    np.fill_diagonal(sim_matrix, 0)
    sim_matrix = np.triu(sim_matrix)

    for i in range(len(ids)):
        row = sim_matrix[i]
        candidates = np.where(row > ASSOC_THRESHOLD)[0]
        if len(candidates) == 0:
            continue
        top_idx = candidates[np.argsort(-row[candidates])[:MAX_EDGES_PER_NODE]]
        for j in top_idx:
            edges.append({
                "from": ids[i], "to": ids[int(j)],
                "type": "association", "weight": round(float(row[j]), 4),
                "last_activation": datetime.now().isoformat(),
            })
    return edges


def build_causal_edges(nodes: list[dict], file_contents: dict) -> list[dict]:
    """Build causal edges from explicit references."""
    edges = []
    path_to_id = {n["path"]: n["id"] for n in nodes}

    ref_patterns = [
        re.compile(r"→\s*`?(memory/[^`\s]+\.md)`?"),
        re.compile(r"see\s+`?(memory/[^`\s]+\.md)`?", re.IGNORECASE),
        re.compile(r"ref:\s*`?(memory/[^`\s]+\.md)`?", re.IGNORECASE),
    ]

    for n in nodes:
        content = file_contents.get(n["path"], "")
        if not content:
            continue
        for pattern in ref_patterns:
            for match in pattern.finditer(content):
                ref_path = match.group(1)
                target_id = path_to_id.get(ref_path)
                if target_id and target_id != n["id"]:
                    edges.append({
                        "from": n["id"], "to": target_id,
                        "type": "causal", "weight": 1.0,
                        "last_activation": datetime.now().isoformat(),
                    })
    return edges


# ── PageRank ──────────────────────────────────────────────────────────

def compute_decay_factor(rel_path: str, node_type: str) -> float:
    """Compute time-based decay factor for a node.

    Exponential decay: exp(-ln(2) / half_life * age_days)
    Archive/dead-files: hard-capped at ARCHIVE_DECAY.
    """
    if "/archive/" in rel_path or "/dead-files/" in rel_path:
        return ARCHIVE_DECAY

    half_life = DECAY_HALF_LIFE.get(node_type, 60)

    try:
        full_path = MEMORY_DIR.parent / rel_path
        if full_path.exists():
            mtime = os.path.getmtime(full_path)
            age_days = (datetime.now().timestamp() - mtime) / 86400
            return math.exp(-math.log(2) / half_life * age_days)
    except (OSError, ValueError):
        pass

    return 0.5


def extract_emotional_weight(content: str) -> float:
    """Extract emotional weight from file content."""
    content_lower = content.lower()
    count = 0
    for marker in EMOTIONAL_MARKERS:
        count += content_lower.count(marker.lower())
    return min(1.0, 0.2 * count)


def compute_pagerank(nodes: list[dict], edges: list[dict]) -> dict:
    """Compute PageRank for all nodes. Returns node_id -> pagerank."""
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        G.add_edge(e["from"], e["to"], weight=e["weight"])

    try:
        pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=100)
    except nx.PowerIterationFailedConvergence:
        pr = {n["id"]: 1.0 / len(nodes) for n in nodes}

    return {k: round(v, 6) for k, v in pr.items()}


# ── Q-value Integration ──────────────────────────────────────────────

def load_q_values() -> dict:
    """Load Q-values from JSON file."""
    if Q_VALUES_FILE.exists():
        try:
            return json.loads(Q_VALUES_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def init_q_values(nodes: list[dict]):
    """Initialize Q-values for all nodes. Preserve existing values, prune deleted nodes."""
    existing = load_q_values()
    meta = existing.pop("_meta", {})
    current_paths = {n["path"] for n in nodes}
    pruned = {k: v for k, v in existing.items() if k in current_paths}

    for n in nodes:
        path = n["path"]
        if path not in pruned:
            pruned[path] = {"q": 0.5, "n": 0, "last_update": date.today().isoformat()}

    pruned["_meta"] = meta
    Q_VALUES_FILE.parent.mkdir(parents=True, exist_ok=True)
    Q_VALUES_FILE.write_text(json.dumps(pruned, indent=2, ensure_ascii=False))
    return pruned


# ── Graph Assembly ────────────────────────────────────────────────────

def build_graph():
    """Full graph build."""
    print("Scanning memory files...", file=sys.stderr)
    files = scan_memory_files()
    print(f"  Found {len(files)} .md files", file=sys.stderr)

    nodes = []
    file_contents = {}
    node_index = {}

    for i, (rel_path, content) in enumerate(files):
        nid = node_id_from_path(rel_path)
        ntype = classify_node(rel_path)
        timestamp = extract_date(rel_path)

        node = {
            "id": nid,
            "type": ntype,
            "path": rel_path,
            "timestamp": timestamp,
        }
        nodes.append(node)
        file_contents[rel_path] = content
        node_index[nid] = i

    if len(nodes) > MAX_ACTIVE_NODES:
        print(f"  Warning: {len(nodes)} nodes exceeds limit {MAX_ACTIVE_NODES}, trimming...", file=sys.stderr)
        nodes.sort(key=lambda n: n["timestamp"], reverse=True)
        nodes = nodes[:MAX_ACTIVE_NODES]
        node_index = {n["id"]: i for i, n in enumerate(nodes)}

    print("Building embeddings...", file=sys.stderr)
    embeddings = build_embeddings(nodes, file_contents)

    print("Building edges...", file=sys.stderr)
    edges = []

    temporal = build_temporal_edges(nodes)
    print(f"  Temporal edges: {len(temporal)}", file=sys.stderr)
    edges.extend(temporal)

    abstraction = build_abstraction_edges(nodes, file_contents)
    print(f"  Abstraction edges: {len(abstraction)}", file=sys.stderr)
    edges.extend(abstraction)

    association = build_association_edges(nodes, embeddings, node_index)
    print(f"  Association edges: {len(association)}", file=sys.stderr)
    edges.extend(association)

    causal = build_causal_edges(nodes, file_contents)
    print(f"  Causal edges: {len(causal)}", file=sys.stderr)
    edges.extend(causal)

    edge_key = {}
    for e in edges:
        key = (e["from"], e["to"])
        if key not in edge_key or e["weight"] > edge_key[key]["weight"]:
            edge_key[key] = e
    edges = list(edge_key.values())
    print(f"  Total edges (deduplicated): {len(edges)}", file=sys.stderr)

    print("Computing PageRank...", file=sys.stderr)
    pagerank = compute_pagerank(nodes, edges)

    print("Applying decay factors and emotional weights...", file=sys.stderr)
    for n in nodes:
        raw_pr = pagerank.get(n["id"], 0.0)
        decay = compute_decay_factor(n["path"], n["type"])
        emotional = extract_emotional_weight(file_contents.get(n["path"], ""))
        n["raw_pagerank"] = raw_pr
        n["decay_factor"] = round(decay, 4)
        n["pagerank"] = round(raw_pr * decay, 6)
        n["emotional_weight"] = round(emotional, 2)

    print("Initializing Q-values...", file=sys.stderr)
    init_q_values(nodes)

    graph = {
        "version": "1.0",
        "built_at": datetime.now().isoformat(),
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "type_counts": {},
            "edge_type_counts": {},
        },
        "nodes": nodes,
        "edges": edges,
    }

    for n in nodes:
        t = n["type"]
        graph["stats"]["type_counts"][t] = graph["stats"]["type_counts"].get(t, 0) + 1
    for e in edges:
        t = e["type"]
        graph["stats"]["edge_type_counts"][t] = graph["stats"]["edge_type_counts"].get(t, 0) + 1

    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_FILE.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    print(f"\nGraph saved to {GRAPH_FILE}", file=sys.stderr)
    print(f"  Nodes: {len(nodes)} | Edges: {len(edges)}", file=sys.stderr)
    print(f"  Types: {graph['stats']['type_counts']}", file=sys.stderr)
    print(f"  Edge types: {graph['stats']['edge_type_counts']}", file=sys.stderr)

    return graph


def cmd_stats():
    """Print graph statistics."""
    if not GRAPH_FILE.exists():
        print("No graph file found. Run 'build' first.")
        sys.exit(1)

    graph = json.loads(GRAPH_FILE.read_text())
    s = graph["stats"]
    print(f"Built at: {graph['built_at']}")
    print(f"Nodes: {s['node_count']} | Edges: {s['edge_count']}")
    print(f"Node types: {json.dumps(s['type_counts'])}")
    print(f"Edge types: {json.dumps(s['edge_type_counts'])}")

    degree = defaultdict(int)
    for e in graph["edges"]:
        degree[e["from"]] += 1
        degree[e["to"]] += 1

    if degree:
        vals = list(degree.values())
        print(f"Degree: mean={np.mean(vals):.1f}, max={max(vals)}, min={min(vals)}")

    nodes_sorted = sorted(graph["nodes"], key=lambda n: n.get("pagerank", 0), reverse=True)
    print("\nTop 10 PageRank nodes:")
    for n in nodes_sorted[:10]:
        print(f"  {n['id']}: {n.get('pagerank', 0):.6f} ({n['type']})")


def cmd_query(term: str):
    """Find nodes matching a search term."""
    if not GRAPH_FILE.exists():
        print("No graph file found. Run 'build' first.")
        sys.exit(1)

    graph = json.loads(GRAPH_FILE.read_text())
    term_lower = term.lower()
    matches = [n for n in graph["nodes"] if term_lower in n["id"].lower()]

    if not matches:
        print(f"No nodes matching '{term}'")
        return

    print(f"Found {len(matches)} nodes matching '{term}':")
    for n in matches:
        edge_count = sum(1 for e in graph["edges"] if e["from"] == n["id"] or e["to"] == n["id"])
        print(f"  {n['id']} ({n['type']}) — {edge_count} edges, PR={n.get('pagerank', 0):.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        build_graph()
    elif cmd == "incremental":
        build_graph()
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "query" and len(sys.argv) >= 3:
        cmd_query(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
