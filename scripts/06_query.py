"""Step 6: answer questions. Every answer cites its source.

Part 6a: HYBRID SEARCH for "find examples" questions.
Part 6b: SQL TOOLS for "find all / how many" questions.
  1. Keyword search (FTS5, BM25 ranking): good at exact terms.
  2. Vector search (embeddings, cosine similarity): good at meaning.
  3. Reciprocal Rank Fusion (RRF): merge the two ranked lists into one.

Usage:
  uv run python scripts/06_query.py search "termination for convenience"
  uv run python scripts/06_query.py search "how much notice to end the contract" --show-lists
  uv run python scripts/06_query.py all "which contracts have no liability cap?"
"""
import argparse
import re
import sqlite3

import numpy as np

DB = "data/legal.db"
EMBED_MODEL = "nomic-embed-text"
RRF_K = 60   # standard RRF constant: dampens the gap between rank 1 and rank 2

con = sqlite3.connect(DB)

# ============================================================
# 1. Keyword search (BM25)
# ============================================================
STOPWORDS = set("""a an the of to in on for and or is are was be by with as at from that this these those it its
what which who how do does any all show me find examples example clauses clause contract contracts agreement
agreements about where when""".split())

def keyword_search(query, k=20):
    """Words joined with OR, so a chunk needn't contain every word; BM25 ranks chunks with more,
    and rarer, matching words higher. Each word is quoted so FTS5 never reads it as a command."""
    words = [w for w in re.findall(r"[a-z0-9$.,]+", query.lower()) if w not in STOPWORDS]
    words = [w.strip(".,") for w in words if w.strip(".,")]
    if not words:
        return []
    fts_query = " OR ".join(f'"{w}"' for w in words)
    rows = con.execute(
        "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (fts_query, k)).fetchall()
    return [r[0] for r in rows]

# ============================================================
# 2. Vector search (cosine similarity, brute force with numpy)
# ============================================================
_matrix, _ids = None, None

def load_vectors():
    """Load every chunk's vector into one matrix (2,398 x 768). At this size, comparing a query
    against ALL of them takes milliseconds, so no special vector database is needed."""
    global _matrix, _ids
    if _matrix is None:
        rows = con.execute("SELECT c.chunk_id, e.vector FROM chunks c JOIN embeddings e ON e.embed_key = c.embed_key "
                           "WHERE e.model = ?", (EMBED_MODEL,)).fetchall()
        _ids = [r[0] for r in rows]
        m = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        _matrix = m / np.linalg.norm(m, axis=1, keepdims=True)   # normalize once, so dot product = cosine
    return _matrix, _ids

def vector_search(query, k=20):
    import ollama
    matrix, ids = load_vectors()
    # nomic expects "search_query:" on questions (and "search_document:" on stored text, set in Step 5)
    q = np.asarray(ollama.embed(model=EMBED_MODEL, input=f"search_query: {query}")["embeddings"][0], dtype=np.float32)
    scores = matrix @ (q / np.linalg.norm(q))
    top = np.argsort(-scores)[:k]
    return [ids[i] for i in top]

# ============================================================
# 3. Reciprocal Rank Fusion
# ============================================================
def rrf(*ranked_lists, k=RRF_K):
    """Each list votes 1 / (k + rank) for each chunk. A chunk ranked high in BOTH lists wins.
    RRF uses only ranks, not raw scores, because BM25 scores and cosine scores aren't comparable."""
    scores = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def hybrid_search(query, top_n=5):
    kw, vec = keyword_search(query), vector_search(query)
    fused = rrf(kw, vec)[:top_n]
    return fused, kw, vec

def describe(chunk_id):
    doc_id, heading, page_start, page_end, text = con.execute(
        "SELECT doc_id, heading_path, page_start, page_end, text FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
    pages = f"p.{page_start}" if page_start == page_end else f"pp.{page_start}-{page_end}"
    return doc_id, heading, pages, text

def print_results(query, fused, kw, vec, show_lists=False):
    print(f'\nSearch: "{query}"   (hybrid: keyword + meaning, merged with RRF)\n')
    for n, cid in enumerate(fused, start=1):
        doc_id, heading, pages, text = describe(cid)
        kw_rank = kw.index(cid) + 1 if cid in kw else "-"
        vec_rank = vec.index(cid) + 1 if cid in vec else "-"
        print(f"{n}. [{doc_id}, {pages}] {heading}")
        print(f"   keyword rank {kw_rank} | meaning rank {vec_rank}")
        print(f"   \"{text[:220]}...\"\n")
    if show_lists:
        print("Top 5 keyword only:", ", ".join(kw[:5]) or "none")
        print("Top 5 meaning only:", ", ".join(vec[:5]))

# ============================================================
# Part 6b: SQL TOOLS for "find all / how many" questions
# The AI never writes SQL. It only picks a tool and fills its blanks (as JSON).
# Code builds the SQL from fixed templates, so every query is one we know is correct.
# ============================================================
import json

# How each user-facing field maps to the extractions table
FIELD_SQL = {
    "governing_law":      ("governing_law", "e.value_norm"),
    "effective_date":     ("effective_date", "e.value_norm"),
    "parties":            ("parties", "e.value_norm"),
    "termination_notice": ("termination_notice", "CAST(e.value_norm AS INTEGER)"),   # days
    "cap_type":           ("liability_cap", "json_extract(e.value_norm, '$.cap_type')"),
    "cap_amount":         ("liability_cap", "json_extract(e.value_norm, '$.cap_amount')"),
}
NUMERIC = {"termination_notice", "cap_amount"}
OPS = {"<", "<=", ">", ">=", "="}

def _rows(sql, params):
    return con.execute(sql, params).fetchall()

def count_by(field):
    src, expr = FIELD_SQL[field]
    sql = (f"SELECT COALESCE(CASE WHEN e.found = 1 THEN {expr} END, '(not found)') AS value, COUNT(*) AS contracts, "
           f"SUM(e.needs_review) AS flagged FROM extractions e WHERE e.field = ? GROUP BY value ORDER BY contracts DESC")
    return sql, [src], _rows(sql, [src])

def list_where(field, value):
    src, expr = FIELD_SQL[field]
    if str(value).lower() in ("none", "not found", "missing", "no"):
        # "no cap" means: cap_type is 'none' OR the model found no cap clause at all
        cond = f"(e.found = 0 OR {expr} IS NULL OR LOWER({expr}) = 'none')"
        params = [src]
    else:
        cond = f"e.found = 1 AND LOWER({expr}) LIKE ?"
        params = [src, f"%{str(value).lower()}%"]
    sql = (f"SELECT e.doc_id, {expr} AS value, e.page, e.quote, e.needs_review FROM extractions e "
           f"WHERE e.field = ? AND {cond} ORDER BY e.doc_id")
    return sql, params, _rows(sql, params)

def compare(field, op, number):
    if field not in NUMERIC or op not in OPS:
        raise ValueError(f"compare needs a numeric field {sorted(NUMERIC)} and an operator {sorted(OPS)}")
    src, expr = FIELD_SQL[field]
    sql = (f"SELECT e.doc_id, {expr} AS value, e.page, e.quote, e.needs_review FROM extractions e "
           f"WHERE e.field = ? AND e.found = 1 AND {expr} {op} ? ORDER BY value DESC")
    return sql, [src, number], _rows(sql, [src, number])

def terms_in_force(doc_id):
    """A master plus its amendments, in date order: a value stated in a later amendment overrides the earlier one.
    Simplification: we treat any stated value as a full replacement (real amendments can change part of a clause)."""
    root = con.execute("SELECT COALESCE(master_id, doc_id) FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if not root:
        raise ValueError(f"unknown document {doc_id}")
    return family_terms(root[0])

def family_roots():
    """One root per agreement relationship: each master, plus each orphan (an amendment whose master isn't here)."""
    return [r[0] for r in con.execute("SELECT doc_id FROM documents WHERE master_id IS NULL ORDER BY doc_id")]

def family_terms(root_id):
    root = (root_id,)
    family = _rows("SELECT d.doc_id, d.doc_type, (SELECT value_norm FROM extractions WHERE doc_id = d.doc_id "
                   "AND field = 'effective_date') AS eff FROM documents d WHERE d.doc_id = ? OR d.master_id = ?",
                   (root[0], root[0]))
    # master first, then amendments by effective date
    family.sort(key=lambda r: (r[1] != "master", r[2] or "9999"))
    terms = {}
    for doc, doc_type, eff in family:
        if doc_type == "ancillary":
            continue                      # e.g. a power of attorney: related, but it doesn't change terms
        for field, (src, expr) in FIELD_SQL.items():
            if field in ("effective_date", "parties") and doc_type != "master":
                continue                  # the agreement's start date and parties come from the master;
                                          # an amendment's date is when ITS changes took effect
            row = con.execute(f"SELECT {expr}, e.page, e.quote FROM extractions e WHERE e.doc_id = ? AND e.field = ? "
                              f"AND e.found = 1", (doc, src)).fetchone()
            if row and row[0] not in (None, "none"):
                terms[field] = (row[0], doc, row[1], row[2])
    return family, terms

# ---------- Family level: the unit is the AGREEMENT, not the PDF ----------
def family_answer(tool, field, value=None, op=None, number=None):
    """Same questions as the SQL tools, asked of each family's terms in force.
    Amendments roll up into their master, so a cap stated only in the master still counts."""
    rows = []
    for root in family_roots():
        family, terms = family_terms(root)
        doc_type = next(t for d, t, _ in family if d == root)
        hit = terms.get(field)                       # (value, from_doc, page, quote) or None
        v = hit[0] if hit else None
        note = f"{len(family)} docs" + ("; master not in sample" if doc_type != "master" else "")
        if tool == "list_where":
            wants_none = str(value).lower() in ("none", "not found", "missing", "no")
            keep = (v is None) if wants_none else (v is not None and str(value).lower() in str(v).lower())
        elif tool == "compare":
            keep = v is not None and eval_op(float(v), op, float(number))
        else:
            keep = True
        if keep:
            rows.append((root, v, hit, note))
    return rows

def eval_op(a, op, b):
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "=": a == b}[op]

def accuracy_note(field):
    """The measured accuracy for this field, from the latest eval runs (Step 7)."""
    try:
        rows = con.execute("SELECT eval_set, precision, recall, n, prompt_version FROM eval_results WHERE field = ?",
                           (field,)).fetchall()
    except sqlite3.OperationalError:
        return "  Accuracy: not measured yet (run scripts/07_evaluate.py)."
    if not rows:
        return f"  Accuracy for {field}: not measured."
    return "\n".join(f"  Measured accuracy for {field} on {s} ({n} contracts, prompt {pv}): "
                     f"precision {p:.0%}, recall {r:.0%}" for s, p, r, n, pv in rows)

# ---------- The AI's job: question -> tool + blanks ----------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["count_by", "list_where", "compare", "terms_in_force"]},
        "field": {"type": "string", "enum": list(FIELD_SQL)},
        "value": {"type": ["string", "null"]},
        "op": {"type": ["string", "null"], "enum": sorted(OPS) + [None]},
        "number": {"type": ["number", "null"]},
        "doc_id": {"type": ["string", "null"]},
    },
    "required": ["tool", "field", "value", "op", "number", "doc_id"],
}
PLAN_PROMPT = """You translate a question about a contract database into ONE tool call. Do not answer the question.
Tools:
- count_by(field): how many contracts per value. "How many contracts per governing law?"
- list_where(field, value): which contracts have a value. value "none" means missing/no such clause.
  "Which contracts have no liability cap?" -> field cap_type, value none
- compare(field, op, number): numeric fields only (termination_notice in DAYS, cap_amount in dollars).
  "notice periods longer than 60 days" -> termination_notice > 60
- terms_in_force(doc_id): current terms of one agreement family, e.g. "terms in force for D001".
Fields: governing_law, effective_date, parties, termination_notice (days), cap_type
(fixed_amount, fees_based, exclusion_only, none), cap_amount.
Set unused blanks to null."""

def plan(question):
    import ollama
    r = ollama.chat(model="qwen2.5:14b", format=PLAN_SCHEMA, options={"temperature": 0},
                    messages=[{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": question}])
    return json.loads(r["message"]["content"])

def run_plan(p):
    """Code checks the AI's plan before running anything: an invalid plan is refused, never guessed at."""
    tool, field = p["tool"], p["field"]
    if tool == "count_by":
        return count_by(field)
    if tool == "list_where":
        if p["value"] is None:
            raise ValueError("list_where needs a value")
        return list_where(field, p["value"])
    if tool == "compare":
        return compare(field, p["op"], p["number"])
    if tool == "terms_in_force":
        if not p["doc_id"]:
            raise ValueError("terms_in_force needs a doc_id")
        return terms_in_force(p["doc_id"].upper())

def print_all(question, p, result):
    print(f'\nQuestion: "{question}"   (exhaustive: answered from the extraction table with SQL)')
    print(f"AI's plan: {json.dumps({k: v for k, v in p.items() if v is not None})}")
    if p["tool"] == "terms_in_force":
        family, terms = result
        print(f"Family: " + ", ".join(f"{d} ({t})" for d, t, _ in family) + "\n")
        for field, (value, doc, page, quote) in terms.items():
            print(f"  {field:18} {str(value)[:40]:40} from {doc}, p.{page}: \"{(quote or '')[:90]}\"")
        return
    sql, params, rows = result
    print(f"SQL (built by code): {sql}\n   params: {params}\n")
    if p["tool"] == "count_by":
        for value, n, flagged in rows:
            print(f"  {str(value)[:50]:50} {n:3} contracts" + (f"  ({flagged} flagged for review)" if flagged else ""))
        return
    print(f"{len(rows)} contracts:")
    for doc, value, page, quote, review in rows:
        cite = f'p.{page}: "{(quote or "")[:90]}"' if quote else "(no clause found: unverified, could be a miss)"
        print(f"  {doc}  {str(value)[:30]:30} {cite}" + ("  [needs review]" if review else ""))
    print("\nCounted per DOCUMENT: amendments are listed on their own. Use --level family to count agreements.")
    print(accuracy_note(p["field"]))

def print_family(question, p, rows):
    print(f'\nQuestion: "{question}"   (exhaustive, per AGREEMENT: each master merged with its amendments)')
    print(f"AI's plan: {json.dumps({k: v for k, v in p.items() if v is not None})}\n")
    if p["tool"] == "count_by":
        counts = {}
        for root, v, hit, note in rows:
            counts.setdefault(v if v is not None else "(not found)", []).append(root)
        for v, roots in sorted(counts.items(), key=lambda kv: -len(kv[1])):
            print(f"  {str(v)[:45]:45} {len(roots):3} agreements  ({', '.join(roots)})")
    else:
        print(f"{len(rows)} agreements:")
        for root, v, hit, note in rows:
            if hit:
                print(f"  {root}  {str(v)[:28]:28} from {hit[1]}, p.{hit[2]}: \"{(hit[3] or '')[:70]}\"  [{note}]")
            else:
                print(f"  {root}  {'-':28} no clause found in any family document: unverified, could be a miss  [{note}]")
    print("\n" + accuracy_note(p["field"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="find examples (hybrid search)")
    s.add_argument("query")
    s.add_argument("--show-lists", action="store_true", help="also show the keyword-only and meaning-only rankings")
    a = sub.add_parser("all", help="find all / how many (SQL tools)")
    a.add_argument("question")
    a.add_argument("--level", choices=["family", "document"], default="family",
                   help="count agreements (master + amendments merged, default) or individual PDFs")
    args = ap.parse_args()
    if args.cmd == "all":
        p = plan(args.question)
        try:
            if args.level == "family" and p["tool"] in ("count_by", "list_where", "compare"):
                print_family(args.question, p, family_answer(p["tool"], p["field"], p["value"], p["op"], p["number"]))
            else:
                print_all(args.question, p, run_plan(p))
        except ValueError as e:
            print(f"Can't run that plan ({e}). AI's plan was: {p}. Try rephrasing.")
    if args.cmd == "search":
        fused, kw, vec = hybrid_search(args.query)
        print_results(args.query, fused, kw, vec, args.show_lists)
