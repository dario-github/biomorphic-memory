#!/usr/bin/env python3
"""Memory Utility Scoring System

Bayesian utility scoring + Thompson Sampling re-ranking.
Combines both the update logic and the re-ranker from the original design.

Usage:
    python3 src/utility.py query <path>
    python3 src/utility.py feedback <path> <outcome>   # outcome: success|partial|failure
    python3 src/utility.py stats                       # summary statistics
    python3 src/utility.py rerank                      # Re-ranks search results from stdin
"""

import json
import sys
import os
import random
import re
from datetime import date
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).parent.parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", WORKSPACE / "data"))
SCORES_FILE = Path(os.environ.get("UTILITY_SCORES_FILE", DATA_DIR / "utility-scores.jsonl"))

# Bayesian update parameters
NOISE_VAR = float(os.environ.get("UTILITY_NOISE_VAR", "0.1"))
DEFAULT_MU = float(os.environ.get("UTILITY_DEFAULT_MU", "0.5"))
DEFAULT_SIGMA_SQ = float(os.environ.get("UTILITY_DEFAULT_SIGMA_SQ", "0.25"))

# Re-ranker parameters
LAMBDA = float(os.environ.get("UTILITY_RERANK_LAMBDA", "0.1"))

R_ADV_MAP = {"success": 0.5, "partial": 0.0, "failure": -0.5}


def _load_scores():
    entries = []
    if not SCORES_FILE.exists():
        return entries
    for line in SCORES_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _save_scores(entries):
    header = [
        "# Memory Utility Scores",
        '# Schema: {"path": str, "mu": float, "sigma_sq": float, "hits": int, "misses": int, "last_used": str|null, "created": str}',
    ]
    lines = header + [json.dumps(e, ensure_ascii=False) for e in entries]
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.write_text("\n".join(lines) + "\n")


def _find_entry(entries, path):
    for e in entries:
        if e["path"] == path:
            return e
    return None


def _make_entry(path, mu=DEFAULT_MU, sigma_sq=DEFAULT_SIGMA_SQ):
    return {
        "path": path,
        "mu": mu,
        "sigma_sq": sigma_sq,
        "hits": 0,
        "misses": 0,
        "last_used": None,
        "created": date.today().isoformat(),
    }


def cmd_query(path):
    entries = _load_scores()
    entry = _find_entry(entries, path)
    if entry:
        print(json.dumps(entry, indent=2, ensure_ascii=False))
    else:
        print(f"No entry found for: {path}")


def cmd_feedback(path, outcome):
    if outcome not in R_ADV_MAP:
        print(f"Invalid outcome: {outcome}. Must be: success|partial|failure")
        sys.exit(1)

    r_adv = R_ADV_MAP[outcome]
    entries = _load_scores()
    entry = _find_entry(entries, path)

    if not entry:
        entry = _make_entry(path)
        entries.append(entry)

    mu_prior = entry["mu"]
    sigma_sq = entry["sigma_sq"]

    mu_post = (sigma_sq * r_adv + NOISE_VAR * mu_prior) / (sigma_sq + NOISE_VAR)
    sigma_sq_post = (sigma_sq * NOISE_VAR) / (sigma_sq + NOISE_VAR)

    mu_post = max(0.0, min(1.0, mu_post))

    entry["mu"] = round(mu_post, 4)
    entry["sigma_sq"] = round(sigma_sq_post, 4)
    entry["last_used"] = date.today().isoformat()

    if outcome == "success":
        entry["hits"] += 1
    elif outcome == "failure":
        entry["misses"] += 1

    _save_scores(entries)
    print(f"Updated {path}: mu={entry['mu']}, σ²={entry['sigma_sq']}, hits={entry['hits']}, misses={entry['misses']}")


def cmd_stats():
    entries = _load_scores()
    if not entries:
        print("No entries.")
        return

    mus = [e["mu"] for e in entries]
    total_hits = sum(e["hits"] for e in entries)
    total_misses = sum(e["misses"] for e in entries)

    print(f"Total entries: {len(entries)}")
    print(f"Mean μ: {sum(mus)/len(mus):.3f} (min={min(mus):.3f}, max={max(mus):.3f})")
    print(f"Total hits: {total_hits}, misses: {total_misses}")


# ── Re-ranker logic ───────────────────────────────────────────────────

def thompson_sample(mu, sigma_sq):
    if sigma_sq <= 0:
        return mu
    return random.gauss(mu, sigma_sq ** 0.5)


def parse_vml_output(lines):
    results = []
    current = None
    in_files_section = False
    
    for line in lines:
        if line.strip().startswith("## File Hits") or line.strip().startswith("## 文件命中"):
            in_files_section = True
            results.append({"passthrough": line})
            continue
        if line.strip().startswith("## Directory Hits") or line.strip().startswith("## 目录命中"):
            in_files_section = False
            results.append({"passthrough": line})
            continue
        
        if not in_files_section:
            if current:
                results.append(current)
                current = None
            results.append({"passthrough": line})
            continue
        
        file_match = re.match(r'^(\d+)\.\s+([\w/\-.]+\.md)(?::L\d+)?\s*\(score:\s*([\d.]+)\)', line)
        if file_match:
            if current:
                results.append(current)
            current = {
                "rank": int(file_match.group(1)),
                "semantic_score": float(file_match.group(3)),
                "path": file_match.group(2),
                "raw_lines": [line],
            }
            continue
        
        num_match = re.match(r'^(\d+)\.\s+([\w/\-.]+\.md)', line)
        if num_match:
            if current:
                results.append(current)
            current = {
                "rank": int(num_match.group(1)),
                "semantic_score": 1.0,
                "path": num_match.group(2),
                "raw_lines": [line],
            }
            continue
        
        if current:
            current["raw_lines"].append(line)
        else:
            results.append({"passthrough": line})
    
    if current:
        results.append(current)
    return results


def rerank(results, scores):
    reranked = []
    scores_dict = {e["path"]: e for e in scores}
    
    for r in results:
        if "passthrough" in r:
            reranked.append(r)
            continue
        
        path = r["path"]
        entry = scores_dict.get(path)
        
        if entry:
            mu = entry["mu"]
            sigma_sq = entry["sigma_sq"]
        else:
            mu = DEFAULT_MU
            sigma_sq = DEFAULT_SIGMA_SQ
        
        sampled_utility = max(0.0, min(1.0, thompson_sample(mu, sigma_sq)))
        hybrid_score = (1 - LAMBDA) * r["semantic_score"] + LAMBDA * sampled_utility
        
        r["hybrid_score"] = hybrid_score
        reranked.append(r)
    
    scored = [r for r in reranked if "hybrid_score" in r]
    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    output = []
    scored_iter = iter(scored)
    for r in reranked:
        if "passthrough" in r:
            output.append(r)
        else:
            output.append(next(scored_iter))
    
    return output


def format_output(results):
    lines = []
    file_rank = 0
    for r in results:
        if "passthrough" in r:
            lines.append(r["passthrough"])
        else:
            file_rank += 1
            for i, raw_line in enumerate(r["raw_lines"]):
                if i == 0:
                    renumbered = re.sub(r'^\d+\.', f'{file_rank}.', raw_line)
                    lines.append(renumbered)
                else:
                    lines.append(raw_line)
    return "\n".join(lines)


def cmd_rerank():
    input_lines = sys.stdin.read().strip().split("\n")
    if not input_lines or input_lines == [""]:
        return
    
    scores = _load_scores()
    results = parse_vml_output(input_lines)
    reranked = rerank(results, scores)
    print(format_output(reranked))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "query" and len(sys.argv) == 3:
        cmd_query(sys.argv[2])
    elif cmd == "feedback" and len(sys.argv) == 4:
        cmd_feedback(sys.argv[2], sys.argv[3])
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "rerank":
        cmd_rerank()
    else:
        print(__doc__)
        sys.exit(1)
