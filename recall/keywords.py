#!/usr/bin/env python3
"""
Auto-extract keywords from memory/entities/ into keywords.json.

Scans entities/people/ and entities/projects/ for .md files.
For each entity:
  1. Filename (sans .md) → primary keyword
  2. YAML frontmatter `related:` → alias keywords
  3. First `# Title` line → extract name variants 
  4. YAML `aliases:` field if present

Merges with existing keywords.json.

Usage:
  python3 recall/keywords.py [--dry-run]
"""

import json
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", os.getcwd()))
ENTITIES_DIR = WORKSPACE / "memory" / "entities"
DATA_DIR = Path(os.environ.get("DATA_DIR", WORKSPACE / "data"))
KEYWORDS_JSON = DATA_DIR / "keywords.json"

SKIP_FILES = {"_abstract.md", "_overview.md", "entity-map.json"}


def parse_frontmatter(content: str) -> dict:
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
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
            result[key] = items
        else:
            result[key] = val
    return result


def extract_title_variants(content: str) -> list[str]:
    variants = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            paren_match = re.search(r'\(([^)]+)\)', title)
            if paren_match:
                main = title[:paren_match.start()].strip()
                alts_str = paren_match.group(1)
                alts = [a.strip() for a in re.split(r'[/,]', alts_str) if a.strip()]
                if main:
                    variants.append(main)
                variants.extend(alts)
            else:
                if title:
                    variants.append(title)
            break
    return variants


def scan_entities() -> dict[str, list[str]]:
    entity_keywords = {}
    
    for subdir in ["people", "projects", "concepts"]:
        dirpath = ENTITIES_DIR / subdir
        if not dirpath.exists():
            continue
        for f in sorted(dirpath.iterdir()):
            if f.name in SKIP_FILES or not f.name.endswith(".md"):
                continue
            entity_name = f.stem
            
            content = f.read_text(encoding="utf-8")
            fm = parse_frontmatter(content)
            
            keywords = set()
            
            related = fm.get("related", [])
            if isinstance(related, list):
                for r in related:
                    if r and r != entity_name:
                        keywords.add(r)
            
            aliases = fm.get("aliases", [])
            if isinstance(aliases, list):
                for a in aliases:
                    if a:
                        keywords.add(a)
            
            for v in extract_title_variants(content):
                if v.lower() != entity_name.lower():
                    keywords.add(v)
            
            entity_keywords[entity_name] = sorted(keywords)
    
    return entity_keywords


def merge_keywords(existing: dict, extracted: dict) -> dict:
    result = {}
    entity_names = set(extracted.keys())
    
    for name, kws in extracted.items():
        merged_kws = set(kws)
        if name in existing:
            for manual_kw in existing[name]:
                merged_kws.add(manual_kw)
        result[name] = sorted(merged_kws)
    
    for name, kws in existing.items():
        if name not in entity_names and not name.startswith("_"):
            result[name] = kws
    
    return result


def main():
    dry_run = "--dry-run" in sys.argv
    
    existing = {}
    if KEYWORDS_JSON.exists():
        existing = json.loads(KEYWORDS_JSON.read_text(encoding="utf-8"))
    
    extracted = scan_entities()
    merged = merge_keywords(existing, extracted)
    
    new_entities = set(extracted.keys()) - set(existing.keys())
    updated = 0
    for name in extracted:
        old = set(existing.get(name, []))
        new = set(merged.get(name, []))
        if new != old:
            updated += 1
    
    print(f"Scanned: {len(extracted)} entities")
    print(f"New entities: {len(new_entities)}")
    print(f"Updated: {updated}")
    print(f"Manual entries preserved: {len(merged) - len(extracted)}")
    print(f"Total entries: {len(merged)}")
    
    if dry_run:
        print("\n--- DRY RUN (no write) ---")
        print(json.dumps(merged, ensure_ascii=False, indent=2))
    else:
        KEYWORDS_JSON.parent.mkdir(parents=True, exist_ok=True)
        KEYWORDS_JSON.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"\nWritten to {KEYWORDS_JSON}")


if __name__ == "__main__":
    main()
