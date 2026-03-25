#!/usr/bin/env python3
"""
recall-search.py — Dual-path search: FTS5 (BM25) + Vec (cosine similarity)
Called by auto-recall handler via execSync
Input: user message (argv[1])
Output: JSON { bm25: [...], vec: [...] }
"""
import sys
import json
import sqlite3
import struct
import os

DB_PATH = os.environ.get("RECALL_DB_PATH", os.path.expanduser("~/.memory/main.sqlite"))
VEC_SO = os.environ.get("SQLITE_VEC_SO", "/usr/local/lib/sqlite-vec/vec0.so")
MAX_RESULTS = 5
MAX_SNIPPET = 400

def fts5_search(conn, query, limit=MAX_RESULTS):
    """BM25 full-text search"""
    terms = []
    for char in query:
        if '\u4e00' <= char <= '\u9fff':
            terms.append(char)
        elif char.isalnum():
            if terms and terms[-1].isascii() and terms[-1].isalnum():
                terms[-1] += char
            else:
                terms.append(char)

    if not terms:
        return []

    results = []
    seen_paths = set()

    for term in terms:
        if len(term) < 2 and term.isascii():
            continue
        try:
            safe_term = term.replace('"', '""')
            rows = conn.execute("""
                SELECT c.path, c.start_line, c.end_line, substr(c.text, 1, ?),
                       rank
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (MAX_SNIPPET, f'"{safe_term}"', limit * 2)).fetchall()

            for path, sl, el, text, rank in rows:
                key = f"{path}:{sl}"
                if key not in seen_paths:
                    seen_paths.add(key)
                    results.append({
                        "path": path,
                        "startLine": sl,
                        "endLine": el,
                        "snippet": text[:MAX_SNIPPET],
                        "score": -rank,  # BM25 rank is negative (lower = better)
                        "source": "bm25",
                        "term": term
                    })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    final = []
    seen = set()
    for r in results:
        if r["path"] not in seen and len(final) < limit:
            seen.add(r["path"])
            final.append(r)
    return final


def vec_search(conn, query_text, limit=MAX_RESULTS):
    """Vector cosine similarity search."""
    try:
        conn.enable_load_extension(True)
        conn.load_extension(VEC_SO)
    except Exception:
        return []

    try:
        terms = [c for c in query_text if '\u4e00' <= c <= '\u9fff']
        if not terms:
            terms = query_text.split()[:3]

        best_chunk = None
        for term in terms[:5]:
            safe = term.replace('"', '""')
            try:
                row = conn.execute("""
                    SELECT c.id, c.embedding FROM chunks_fts f
                    JOIN chunks c ON c.id = f.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank LIMIT 1
                """, (f'"{safe}"',)).fetchone()
                if row and row[1] and len(row[1]) > 10:
                    best_chunk = row
                    break
            except:
                continue

        if not best_chunk:
            return []

        emb = json.loads(best_chunk[1])
        blob = struct.pack('<' + 'f' * len(emb), *emb)

        rows = conn.execute("""
            SELECT c.path, c.start_line, c.end_line, substr(c.text, 1, ?),
                   vec_distance_cosine(v.embedding, ?) as dist
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.id
            WHERE c.id != ?
            ORDER BY dist ASC
            LIMIT ?
        """, (MAX_SNIPPET, blob, best_chunk[0], limit)).fetchall()

        return [{
            "path": path,
            "startLine": sl,
            "endLine": el,
            "snippet": text[:MAX_SNIPPET],
            "score": round(1.0 - dist, 4),
            "source": "vec"
        } for path, sl, el, text, dist in rows]

    except Exception as e:
        return []


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"bm25": [], "vec": [], "error": "no query"}))
        return

    query = sys.argv[1][:500]

    if not os.path.exists(DB_PATH):
        print(json.dumps({"bm25": [], "vec": [], "error": "no db"}))
        return

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    bm25_results = fts5_search(conn, query)
    vec_results = vec_search(conn, query)

    conn.close()

    print(json.dumps({
        "bm25": bm25_results,
        "vec": vec_results
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
