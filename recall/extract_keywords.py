#!/usr/bin/env python3
"""
Phase 4.1 — Auto-extract keywords from memory/entities/ into keywords.json.

Scans entities/people/ and entities/projects/ for .md files.
For each entity:
  1. Filename (sans .md) → primary keyword
  2. YAML frontmatter `related:` → alias keywords
  3. First `# Title` line → extract name variants (e.g., "John Doe (johnd / jdoe)")
  4. YAML `aliases:` field if present

Merges with existing keywords.json: entity entries are updated,
non-entity entries (manual additions like "论文", "投资") are preserved.

Usage:
  python3 hooks/auto-recall/extract-keywords.py [--dry-run]
"""

import json
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "."))
ENTITIES_DIR = WORKSPACE / "memory" / "entities"
KEYWORDS_JSON = WORKSPACE / "hooks" / "auto-recall" / "keywords.json"
CONCEPTS_DIR = ENTITIES_DIR / "concepts"

SKIP_FILES = {"_abstract.md", "_overview.md", "entity-map.json"}


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter as a simple dict."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_text = content[3:end].strip()
    result = {}
    for line in fm_text.split("\n"):
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        # Handle YAML list: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
            result[key] = items
        else:
            result[key] = val
    return result


def extract_title_variants(content: str) -> list[str]:
    """Extract name variants from first # heading.
    
    e.g., '# John Doe (johnd / jdoe)' → ['John Doe', 'User', 'jdoe']
    e.g., '# MemBrain' → ['MemBrain']
    e.g., '# 智能评估体系' → ['智能评估体系']
    """
    variants = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            # Extract parenthesized alternatives
            paren_match = re.search(r'\(([^)]+)\)', title)
            if paren_match:
                main = title[:paren_match.start()].strip()
                alts_str = paren_match.group(1)
                # Split by / or ,
                alts = [a.strip() for a in re.split(r'[/,]', alts_str) if a.strip()]
                if main:
                    variants.append(main)
                variants.extend(alts)
            else:
                if title:
                    variants.append(title)
            break  # Only first heading
    return variants


def scan_entities() -> dict[str, list[str]]:
    """Scan entities/ and extract keywords for each entity."""
    entity_keywords = {}
    
    for subdir in ["people", "projects", "concepts"]:
        dirpath = ENTITIES_DIR / subdir
        if not dirpath.exists():
            continue
        for f in sorted(dirpath.iterdir()):
            if f.name in SKIP_FILES or not f.name.endswith(".md"):
                continue
            entity_name = f.stem  # filename without .md
            
            content = f.read_text(encoding="utf-8")
            fm = parse_frontmatter(content)
            
            keywords = set()
            
            # 1. Related entities from frontmatter
            related = fm.get("related", [])
            if isinstance(related, list):
                for r in related:
                    if r and r != entity_name:
                        keywords.add(r)
            
            # 2. Aliases from frontmatter
            aliases = fm.get("aliases", [])
            if isinstance(aliases, list):
                for a in aliases:
                    if a:
                        keywords.add(a)
            
            # 3. Title variants
            for v in extract_title_variants(content):
                # Don't add if it's the same as entity_name (case-insensitive)
                if v.lower() != entity_name.lower():
                    keywords.add(v)
            
            entity_keywords[entity_name] = sorted(keywords)
    
    return entity_keywords


def merge_keywords(existing: dict, extracted: dict) -> dict:
    """Merge extracted entity keywords with existing manual entries.
    
    Strategy:
    - For entity entries (found in entities/): use extracted as base, 
      add any manual keywords that aren't already present
    - For non-entity entries (manual additions): preserve as-is
    """
    result = {}
    entity_names = set(extracted.keys())
    
    # First, add all extracted entities (with merged manual additions)
    for name, kws in extracted.items():
        merged_kws = set(kws)
        if name in existing:
            for manual_kw in existing[name]:
                merged_kws.add(manual_kw)
        result[name] = sorted(merged_kws)
    
    # Then, preserve non-entity manual entries
    for name, kws in existing.items():
        if name not in entity_names and not name.startswith("_"):
            result[name] = kws
    
    return result


def main():
    dry_run = "--dry-run" in sys.argv
    
    # Load existing
    existing = {}
    if KEYWORDS_JSON.exists():
        existing = json.loads(KEYWORDS_JSON.read_text(encoding="utf-8"))
    
    # Extract from entities/
    extracted = scan_entities()
    
    # Merge
    merged = merge_keywords(existing, extracted)
    
    # Stats
    new_entities = set(extracted.keys()) - set(existing.keys())
    updated = 0
    for name in extracted:
        old = set(existing.get(name, []))
        new = set(merged.get(name, []))
        if new != old:
            updated += 1
    
    print(f"Scanned: {len(extracted)} entities")
    print(f"New entities: {len(new_entities)} ({', '.join(sorted(new_entities)) if new_entities else 'none'})")
    print(f"Updated: {updated}")
    print(f"Manual entries preserved: {len(merged) - len(extracted)}")
    print(f"Total entries: {len(merged)}")
    
    if dry_run:
        print("\n--- DRY RUN (no write) ---")
        print(json.dumps(merged, ensure_ascii=False, indent=2))
    else:
        KEYWORDS_JSON.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"\nWritten to {KEYWORDS_JSON}")


if __name__ == "__main__":
    main()
