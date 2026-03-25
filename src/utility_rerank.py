#!/usr/bin/env python3
"""Memory Utility Re-ranker for vml-search output.

Reads vml-search results from stdin, re-ranks using Thompson Sampling
on utility scores, outputs re-ranked results.

Usage:
    vml-search.sh "query" | python3 scripts/memory-utility-rerank.py
    
Environment:
    MEMORY_UTILITY_LAMBDA=0.1   # weight for utility (default 0.1)
    MEMORY_UTILITY_ENABLED=1    # set to enable (default: disabled)
"""

import sys
import os
import json
import random
import re
from pathlib import Path

SCORES_FILE = os.path.join(os.path.dirname(__file__), "..", "memory", "index", "utility-scores.jsonl")
LAMBDA = float(os.environ.get("MEMORY_UTILITY_LAMBDA", "0.1"))
DEFAULT_MU = 0.5
DEFAULT_SIGMA_SQ = 0.25


def load_scores():
    """Load utility scores into a dict keyed by path."""
    scores = {}
    path = Path(SCORES_FILE)
    if not path.exists():
        return scores
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
            scores[entry["path"]] = entry
        except (json.JSONDecodeError, KeyError):
            continue
    return scores


def thompson_sample(mu, sigma_sq):
    """Sample from N(mu, sigma_sq) for Thompson Sampling."""
    if sigma_sq <= 0:
        return mu
    return random.gauss(mu, sigma_sq ** 0.5)


def parse_vml_output(lines):
    """Parse vml-search output into structured results.
    
    vml-search.sh outputs:
      [VML-TRACE] query="..." phase1_dirs=N phase2_files=N
      
      ## 目录命中
      1. memory/dir/ (score: 0.85) — description
      
      ## 文件命中
      1. memory/file.md:L42 (score: 0.92)
         > excerpt text
    """
    results = []
    current = None
    in_files_section = False
    
    for line in lines:
        # Detect file section
        if line.strip().startswith("## 文件命中"):
            in_files_section = True
            results.append({"passthrough": line})
            continue
        if line.strip().startswith("## 目录命中"):
            in_files_section = False
            results.append({"passthrough": line})
            continue
        
        if not in_files_section:
            # Pass through non-file-section lines (trace, dir hits, headers)
            if current:
                results.append(current)
                current = None
            results.append({"passthrough": line})
            continue
        
        # In file section: match "1. memory/file.md:L42 (score: 0.92)"
        file_match = re.match(r'^(\d+)\.\s+(memory/\S+\.md)(?::L\d+)?\s*\(score:\s*([\d.]+)\)', line)
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
        
        # Fallback: any numbered line with a memory path
        num_match = re.match(r'^(\d+)\.\s+(memory/\S+\.md)', line)
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
        
        # Continuation line (excerpt with >)
        if current:
            current["raw_lines"].append(line)
        else:
            results.append({"passthrough": line})
    
    if current:
        results.append(current)
    
    return results


def rerank(results, scores):
    """Re-rank results using hybrid score = (1-λ)*semantic + λ*thompson_sample(utility)."""
    reranked = []
    
    for r in results:
        if "passthrough" in r:
            reranked.append(r)
            continue
        
        path = r["path"]
        entry = scores.get(path)
        
        if entry:
            mu = entry["mu"]
            sigma_sq = entry["sigma_sq"]
        else:
            mu = DEFAULT_MU
            sigma_sq = DEFAULT_SIGMA_SQ
        
        sampled_utility = thompson_sample(mu, sigma_sq)
        # Clamp to [0, 1]
        sampled_utility = max(0.0, min(1.0, sampled_utility))
        
        hybrid_score = (1 - LAMBDA) * r["semantic_score"] + LAMBDA * sampled_utility
        
        r["hybrid_score"] = hybrid_score
        r["utility_mu"] = mu
        r["utility_sample"] = round(sampled_utility, 4)
        reranked.append(r)
    
    # Sort scored results by hybrid_score descending, keep passthroughs in place
    scored = [r for r in reranked if "hybrid_score" in r]
    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # Reconstruct: passthroughs stay, scored results fill in order
    output = []
    scored_iter = iter(scored)
    for r in reranked:
        if "passthrough" in r:
            output.append(r)
        else:
            output.append(next(scored_iter))
    
    return output


def format_output(results):
    """Format re-ranked results for output, re-numbering file hits."""
    lines = []
    file_rank = 0
    for r in results:
        if "passthrough" in r:
            lines.append(r["passthrough"])
        else:
            file_rank += 1
            # Re-number and optionally annotate with utility
            for i, raw_line in enumerate(r["raw_lines"]):
                if i == 0:
                    # Replace original rank number with new rank
                    renumbered = re.sub(r'^\d+\.', f'{file_rank}.', raw_line)
                    lines.append(renumbered)
                else:
                    lines.append(raw_line)
    return "\n".join(lines)


def main():
    if not os.environ.get("MEMORY_UTILITY_ENABLED"):
        # Pass through unchanged
        sys.stdout.write(sys.stdin.read())
        return
    
    input_lines = sys.stdin.read().strip().split("\n")
    if not input_lines or input_lines == [""]:
        return
    
    scores = load_scores()
    results = parse_vml_output(input_lines)
    reranked = rerank(results, scores)
    print(format_output(reranked))


if __name__ == "__main__":
    main()
