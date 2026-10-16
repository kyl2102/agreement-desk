"""Step 5: build the search indexes.

  1. CHUNK:  split each contract by its STRUCTURE (sections and numbered clauses), not every N characters.
             Every chunk keeps its heading path ("Distributor Agreement > 10. Limitation of Liability")
             and its pages, so search results can cite where they came from.
  2. KEYWORD INDEX: SQLite FTS5 (built-in full-text search, ranked with BM25).
  3. VECTOR INDEX:  one 768-number embedding per chunk from nomic-embed-text, stored in SQLite.

Usage:
  uv run python scripts/05_index.py             # chunk + keyword index + embeddings
  uv run python scripts/05_index.py --no-embed  # chunk + keyword index only (fast, for checking chunks)
"""
import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

PARSED = Path("data/parsed")
DB = Path("data/legal.db")
EMBED_MODEL = "nomic-embed-text"
CHUNKER_VERSION = "c2"
MAX_CHARS = 1200          # target chunk size: about one clause
SKIP_LABELS = {"page_header", "page_footer"}   # running headers/footers: noise for search

# A numbered clause start: "10.", "10.1", "Section 12.3", "ARTICLE 5" followed by a capitalized word.
NUMBERED = re.compile(r"^\s*(?:(?:section|article)\s+)?(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?=[A-Z(\"])", re.I)
# The same pattern appearing INSIDE a giant blob, right after the end of a sentence.
INLINE_NUMBERED = re.compile(r"(?<=[.:;)]\s)(?=(?:\d{1,2}\.){1,3}\d{0,2}\s+[A-Z])")

# A clause number followed by ALL-CAPS title words; the last word may end with a period ("FOR CAUSE.").
CAPS_TITLE = re.compile(r"^\s*((?:(?:Section|SECTION|Article|ARTICLE)\s+)?[\d.]+\s+(?:[A-Z][A-Z/&,'\-]*\.?\s?)+)(?=\s|$)")

def heading_title(text):
    """'10. LIMITATION OF LIABILITY. Except for...' -> '10. LIMITATION OF LIABILITY'
       '1. SCOPE This Agreement establishes...'  -> '1. SCOPE'"""
    m = CAPS_TITLE.match(text)
    if m and re.search(r"[A-Z]{2}", m.group(1)):
        return m.group(1).strip(" .,")
    first = re.split(r"(?<=[a-z\])])\.\s|:\s", text, maxsplit=1)[0]
    return first[:60].strip()

def trusted_header(item):
    """Docling's heading label, sanity-checked: we saw it call a stray 'COMPANY' a heading."""
    t = item["text"].strip()
    return item["label"] in ("section_header", "title") and len(t) <= 120 and (len(t.split()) >= 2 or NUMBERED.match(t))

def page_at(item, offset):
    """Which page a character offset inside a (possibly multi-page) item falls on, using Docling's charspans."""
    for prov in item["prov"]:
        start, end = prov.get("charspan", [0, 10**9])
        if start <= offset < end:
            return prov["page_no"]
    return item["prov"][-1]["page_no"]

def pieces(item):
    """Yield (text, page) pieces of one item: split giant blobs at inline clause numbers, then by size."""
    text = item["text"]
    if len(text) <= MAX_CHARS:
        yield text, page_at(item, 0)
        return
    starts = [0] + [m.start() for m in INLINE_NUMBERED.finditer(text)]
    bounds = list(zip(starts, starts[1:] + [len(text)]))
    for a, b in bounds:
        s = a
        while s < b:                                 # a single clause can still be long: cut by size,
            e = min(b, s + MAX_CHARS)                # preferably at the end of a sentence
            if e < b:
                cut = text.rfind(". ", s + MAX_CHARS // 2, e)
                e = cut + 1 if cut != -1 else e
            yield text[s:e], page_at(item, s)
            s = e

def chunk_document(doc_id):
    doc = json.load(open(PARSED / f"{doc_id}.json"))
    items = [t for t in doc["texts"] if t["label"] not in SKIP_LABELS and t["text"].strip()]
    # Document title: an early heading that names the agreement, else the agreement type from the filename.
    early = [t["text"].strip() for t in items[:15] if re.search(r"agreement|amendment", t["text"][:100], re.I)]
    filename = json.load(open(PARSED / f"{doc_id}.meta.json"))["filename"]
    from_file = re.split(r"[-_]", Path(filename).stem)[-1].strip()
    title = next((t[:60] for t in early if len(t) <= 100), from_file or doc_id)
    top, sub = "", ""                                # current section and sub-section headings
    chunks, buf, pages = [], [], []

    def flush():
        if buf and chunks and sum(len(b) for b in buf) < 150:   # tiny scrap: attach to the previous chunk
            chunks[-1]["text"] += " " + " ".join(buf)
            chunks[-1]["page_end"] = max(chunks[-1]["page_end"], *pages)
            buf.clear(); pages.clear()
        elif buf:
            path = " > ".join(p for p in (title, top, sub) if p)
            chunks.append({"heading_path": path, "text": " ".join(buf), "page_start": min(pages), "page_end": max(pages)})
            buf.clear(); pages.clear()

    for item in items:
        for text, page in pieces(item):
            m = NUMBERED.match(text)
            if trusted_header(item) and text == item["text"]:
                flush(); top, sub = heading_title(text), ""
                continue                             # a heading line on its own: becomes context, not body
            if m:                                    # a numbered clause starts a new chunk
                flush()
                if "." in m.group(1):                # "10.1" -> sub-section under the current section
                    sub = heading_title(text)
                else:                                # "10." -> new top-level section
                    top, sub = heading_title(text), ""
            elif sum(len(b) for b in buf) + len(text) > MAX_CHARS:
                flush()
            buf.append(text.strip()); pages.append(page)
    flush()
    return chunks

SCHEMA = """
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS chunks_fts;
CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, seq INTEGER, heading_path TEXT,
                     text TEXT, page_start INTEGER, page_end INTEGER, chars INTEGER);
-- FTS5 = SQLite's full-text search. 'porter' stemming lets "terminate" match "termination".
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, heading_path, text, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS embeddings (embed_key TEXT PRIMARY KEY, model TEXT, dim INTEGER, vector BLOB);
CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT);
"""

def embed_text(c):
    # nomic-embed-text expects a task prefix: "search_document:" for stored text, "search_query:" for queries.
    # The heading path is included so a chunk carries its context ("contextual retrieval").
    return f"search_document: {c['heading_path']}\n{c['text']}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)

    doc_ids = [r[0] for r in con.execute("SELECT doc_id FROM documents ORDER BY doc_id")]
    all_chunks = []
    for doc_id in doc_ids:
        for seq, c in enumerate(chunk_document(doc_id)):
            c.update(chunk_id=f"{doc_id}-{seq:04d}", doc_id=doc_id, seq=seq)
            all_chunks.append(c)
    con.executemany("INSERT INTO chunks VALUES (:chunk_id,:doc_id,:seq,:heading_path,:text,:page_start,:page_end,:chars)",
                    [dict(c, chars=len(c["text"])) for c in all_chunks])
    con.executemany("INSERT INTO chunks_fts (chunk_id, heading_path, text) VALUES (?,?,?)",
                    [(c["chunk_id"], c["heading_path"], c["text"]) for c in all_chunks])
    sizes = [len(c["text"]) for c in all_chunks]
    print(f"Chunked {len(doc_ids)} contracts into {len(all_chunks)} chunks  |  "
          f"avg {int(np.mean(sizes))} chars, max {max(sizes)}, chunks under 100 chars: {sum(s < 100 for s in sizes)}")

    if not args.no_embed:
        import ollama
        # Cache by content: a chunk whose text didn't change keeps its old embedding (no re-work on re-runs).
        todo = []
        for c in all_chunks:
            c["embed_key"] = hashlib.sha256((EMBED_MODEL + embed_text(c)).encode()).hexdigest()
            if not con.execute("SELECT 1 FROM embeddings WHERE embed_key=?", (c["embed_key"],)).fetchone():
                todo.append(c)
        start = time.time()
        for i in range(0, len(todo), 32):            # send 32 chunks per call (batching = fewer round trips)
            batch = todo[i:i + 32]
            vectors = ollama.embed(model=EMBED_MODEL, input=[embed_text(c) for c in batch])["embeddings"]
            con.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                            [(c["embed_key"], EMBED_MODEL, len(v), np.asarray(v, dtype=np.float32).tobytes())
                             for c, v in zip(batch, vectors)])
            con.commit()
            print(f"  embedded {min(i + 32, len(todo))}/{len(todo)}", end="\r")
        con.execute("ALTER TABLE chunks ADD COLUMN embed_key TEXT")
        con.executemany("UPDATE chunks SET embed_key=? WHERE chunk_id=?", [(c["embed_key"], c["chunk_id"]) for c in all_chunks])
        print(f"\nEmbedded {len(todo)} new chunks in {round(time.time() - start, 1)}s "
              f"({len(all_chunks) - len(todo)} reused from cache)")

    con.executemany("INSERT OR REPLACE INTO index_meta VALUES (?,?)",
                    [("chunker_version", CHUNKER_VERSION), ("embed_model", EMBED_MODEL),
                     ("built_at", time.strftime("%Y-%m-%d %H:%M:%S"))])
    con.commit()
    print(f"Indexes written to {DB}")

if __name__ == "__main__":
    main()
