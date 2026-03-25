#!/usr/bin/env python3
"""Q-value Manager for Biomorphic Memory System.

Non-parametric RL Q-value updates for memory retrieval utility.

Usage:
    python3 src/q_value.py reward <path> <signal>    # signal: hit|miss|neutral|mention
    python3 src/q_value.py query <path>              # check Q-value
    python3 src/q_value.py stats                     # distribution stats
    python3 src/q_value.py report                    # daily report
    python3 src/q_value.py decay                     # apply edge time decay
    python3 src/q_value.py forget-scan               # scan for dormancy candidates
"""

import json
import sys
import os
from datetime import datetime, date
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).parent.parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", WORKSPACE / "data"))
Q_VALUES_FILE = Path(os.environ.get("Q_VALUES_FILE", DATA_DIR / "memory-q-values.json"))
GRAPH_FILE = Path(os.environ.get("GRAPH_FILE", DATA_DIR / "memory-graph.json"))

# Q-value parameters
ALPHA = float(os.environ.get("Q_ALPHA", "0.1"))
Q_INIT = float(os.environ.get("Q_INIT", "0.5"))
EDGE_DECAY_RHO = float(os.environ.get("Q_EDGE_DECAY_RHO", "0.02"))

# Forgetting thresholds
DEMOTION_Q = float(os.environ.get("Q_DEMOTION", "0.3"))
DORMANCY_Q = float(os.environ.get("Q_DORMANCY", "0.2"))
DORMANCY_DAYS = int(os.environ.get("Q_DORMANCY_DAYS", "30"))

PROTECTED_PREFIXES = tuple(os.environ.get("Q_PROTECTED_PREFIXES", "memory/rules/").split(","))

# Reward mapping
REWARD_MAP = {
    "hit":     1.0,    # cited and correct
    "miss":   -1.0,    # cited but corrected
    "neutral": 0.0,    # retrieved but not cited
    "mention": 0.5,    # user mentioned entity
}


def load_q_values() -> dict:
    if Q_VALUES_FILE.exists():
        try:
            return json.loads(Q_VALUES_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def save_q_values(data: dict):
    Q_VALUES_FILE.parent.mkdir(parents=True, exist_ok=True)
    Q_VALUES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_reward(path: str, signal: str):
    """Update Q-value for a memory path."""
    if signal not in REWARD_MAP:
        print(f"Invalid signal: {signal}. Use: {', '.join(REWARD_MAP.keys())}")
        sys.exit(1)
    
    r = REWARD_MAP[signal]
    data = load_q_values()
    
    entry = data.get(path, {"q": Q_INIT, "n": 0, "last_update": None})
    
    q_old = entry["q"]
    q_new = q_old + ALPHA * (r - q_old)
    q_new = max(0.0, min(1.0, q_new))
    
    entry["q"] = round(q_new, 4)
    entry["n"] = entry.get("n", 0) + 1
    entry["last_update"] = date.today().isoformat()
    
    data[path] = entry
    save_q_values(data)
    
    print(f"{path}: Q {q_old:.4f} → {q_new:.4f} (r={r}, n={entry['n']})")


def cmd_query(path: str):
    data = load_q_values()
    entry = data.get(path)
    if entry:
        print(json.dumps({path: entry}, indent=2, ensure_ascii=False))
    else:
        print(f"No Q-value for: {path} (would init at Q={Q_INIT})")


def cmd_stats():
    data = load_q_values()
    if not data:
        print("No Q-values found.")
        return
    
    q_vals = [e["q"] for e in data.values()]
    n_vals = [e.get("n", 0) for e in data.values()]
    never_updated = sum(1 for e in data.values() if e.get("n", 0) == 0)
    below_03 = sum(1 for q in q_vals if q < 0.3)
    above_09 = sum(1 for q in q_vals if q > 0.9)
    
    print(f"Total entries: {len(data)}")
    print(f"Q distribution: mean={sum(q_vals)/len(q_vals):.3f}, min={min(q_vals):.3f}, max={max(q_vals):.3f}")
    print(f"Observations: mean={sum(n_vals)/len(n_vals):.1f}, max={max(n_vals)}")
    print(f"Never updated (n=0): {never_updated}")
    print(f"Below 0.3 (demotion): {below_03}")
    print(f"Above 0.9 (high util): {above_09}")


def cmd_report():
    data = load_q_values()
    if not data:
        print("No Q-values.")
        return
    
    q_vals = [e["q"] for e in data.values()]
    below_02 = [(p, e) for p, e in data.items() if e["q"] < 0.2]
    above_07 = [(p, e) for p, e in data.items() if e["q"] > 0.7 and e.get("n", 0) >= 5]
    
    report = []
    report.append(f"## Q-value Daily Report ({date.today()})")
    report.append(f"- Entries: {len(data)}")
    report.append(f"- Mean Q: {sum(q_vals)/len(q_vals):.3f}")
    
    if below_02:
        report.append(f"\n### Dormancy Candidates ({len(below_02)})")
        for p, e in sorted(below_02, key=lambda x: x[1]["q"])[:10]:
            report.append(f"- Q={e['q']:.2f} n={e.get('n',0)} {p}")
            
    if above_07:
        report.append(f"\n### High Utility ({len(above_07)})")
        for p, e in sorted(above_07, key=lambda x: x[1]["q"], reverse=True)[:10]:
            report.append(f"- Q={e['q']:.2f} n={e.get('n',0)} {p}")
            
    print("\n".join(report))


def cmd_decay():
    if not GRAPH_FILE.exists():
        print("No graph file.")
        sys.exit(1)
        
    graph = json.loads(GRAPH_FILE.read_text())
    today = datetime.now()
    decayed = 0
    
    for edge in graph["edges"]:
        last_act = edge.get("last_activation")
        if not last_act:
            continue
            
        try:
            last_dt = datetime.fromisoformat(last_act)
            days = (today - last_dt).days
        except (ValueError, TypeError):
            continue
            
        if days > 0:
            decay_factor = pow(2.718281828, -EDGE_DECAY_RHO * days)
            old_w = edge["weight"]
            edge["weight"] = round(old_w * decay_factor, 4)
            if edge["weight"] != old_w:
                decayed += 1
                
    GRAPH_FILE.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    print(f"Decayed {decayed} edges ({len(graph['edges'])} total)")


def _is_protected(path: str) -> bool:
    for prefix in PROTECTED_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def cmd_forget_scan():
    data = load_q_values()
    today = date.today()
    
    tier1 = []
    tier2 = []
    
    for path, entry in data.items():
        if _is_protected(path):
            continue
            
        q = entry["q"]
        last_update = entry.get("last_update")
        
        if last_update:
            try:
                days_since = (today - date.fromisoformat(last_update)).days
            except ValueError:
                days_since = 0
        else:
            days_since = 999
            
        if q < DORMANCY_Q and days_since >= DORMANCY_DAYS:
            tier2.append((path, q, days_since))
        elif q < DEMOTION_Q:
            tier1.append((path, q, days_since))
            
    print(f"## Forgetting Scan ({today})")
    print(f"\n### Tier 1 — Demotion (Q < {DEMOTION_Q}): {len(tier1)}")
    for path, q, days in sorted(tier1, key=lambda x: x[1])[:20]:
        print(f"  Q={q:.3f} {days}d {path}")
        
    print(f"\n### Tier 2 — Dormancy Candidates (Q < {DORMANCY_Q} + {DORMANCY_DAYS}d): {len(tier2)}")
    for path, q, days in sorted(tier2, key=lambda x: x[1])[:20]:
        print(f"  Q={q:.3f} {days}d {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
        
    cmd = sys.argv[1]
    if cmd == "reward" and len(sys.argv) == 4:
        cmd_reward(sys.argv[2], sys.argv[3])
    elif cmd == "query" and len(sys.argv) == 3:
        cmd_query(sys.argv[2])
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "report":
        cmd_report()
    elif cmd == "decay":
        cmd_decay()
    elif cmd == "forget-scan":
        cmd_forget_scan()
    else:
        print(__doc__)
        sys.exit(1)
