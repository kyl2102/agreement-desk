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

# Human review (Step 8). Model output in `extractions` is NEVER overwritten: reviews live in their own
# table, with who and when (an audit trail), and the view below layers them on top. Queries read the view,
# so a correction changes every answer, while evals keep measuring the model's own output.
con.executescript("""
CREATE TABLE IF NOT EXISTS reviews (
    doc_id TEXT, field TEXT,
    status TEXT,              -- accepted / corrected / absent
    value TEXT, quote TEXT, page INTEGER, note TEXT,
    reviewer TEXT, reviewed_at TEXT,
    PRIMARY KEY (doc_id, field));
DROP VIEW IF EXISTS current_extractions;   -- recreated each run, so the definition is always current
CREATE VIEW current_extractions AS
SELECT e.doc_id, e.field,
       CASE WHEN r.status = 'absent' THEN 0 WHEN r.status = 'corrected' THEN 1 ELSE e.found END AS found,
       e.value_raw,
       CASE WHEN r.status = 'absent' THEN NULL WHEN r.status = 'corrected' THEN r.value ELSE e.value_norm END AS value_norm,
       COALESCE(r.quote, e.quote) AS quote, COALESCE(r.page, e.page) AS page,
       e.quote_check, e.model_confidence, e.confidence,
       CASE WHEN r.status IS NOT NULL THEN 0 ELSE e.needs_review END AS needs_review,
       e.model, e.prompt_version, r.status AS review_status, r.reviewer
FROM extractions e LEFT JOIN reviews r ON r.doc_id = e.doc_id AND r.field = e.field;
""")

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

# ============================================================
# 3b. RERANKER (Step 9b): a second, slower look at the top candidates
# ============================================================
# Keyword and meaning search score the question and each chunk SEPARATELY, then compare.
# A cross-encoder reads the question and the chunk TOGETHER, so it can tell "notice to terminate"
# from "notice of a price change". Too slow to run on all 2,398 chunks, so it only reorders the top 20.
# Turned on only after 06e_search_eval.py showed it beat plain hybrid: the test decided, not the hype.
# Two candidates, compared in 06e_search_eval.py (a bake-off on OUR questions, not a public leaderboard):
RERANK_MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",   # ~90 MB, fast baseline; trained on web search only
    "bge":    "BAAI/bge-reranker-v2-m3",                 # ~2.3 GB, newer and stronger on general benchmarks
}
RERANK_MODEL = "bge"      # won the bake-off: precision@5 52% vs 44% hybrid, 46% MiniLM; hit@5 90% vs 80%
USE_RERANK = True
RERANK_POOL = 20
_rerankers = {}

def rerank(query, chunk_ids, top_n=5, model=None):
    model = model or RERANK_MODEL
    if not chunk_ids:
        return []
    if model not in _rerankers:
        from sentence_transformers import CrossEncoder   # imported here so the rest of the tool works without it
        _rerankers[model] = CrossEncoder(RERANK_MODELS[model])
    _reranker = _rerankers[model]
    rows = dict(((c, (h, t)) for c, h, t in con.execute(
        f"SELECT chunk_id, heading_path, text FROM chunks WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})", chunk_ids)))
    pairs = [(query, f"{rows[c][0]}\n{rows[c][1]}") for c in chunk_ids]   # heading included: it carries meaning
    scores = _reranker.predict(pairs)
    order = sorted(range(len(chunk_ids)), key=lambda i: -scores[i])
    return [chunk_ids[i] for i in order[:top_n]]

def hybrid_search(query, top_n=5, use_rerank=None):
    kw, vec = keyword_search(query), vector_search(query)
    fused = rrf(kw, vec)
    if USE_RERANK if use_rerank is None else use_rerank:
        return rerank(query, fused[:RERANK_POOL], top_n), kw, vec
    return fused[:top_n], kw, vec

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
           f"SUM(e.needs_review) AS flagged FROM current_extractions e WHERE e.field = ? GROUP BY value ORDER BY contracts DESC")
    return sql, [src], _rows(sql, [src])

def list_where(field, value):
    src, expr = FIELD_SQL[field]
    if str(value).lower() in ("none", "not found", "missing", "no"):
        # "no cap" means: cap_type is 'none' OR the model found no cap clause at all
        cond = f"(e.found = 0 OR {expr} IS NULL OR LOWER({expr}) = 'none')"
        params = [src]
    else:
        term = party_term(value) if field == "parties" else str(value).lower()
        cond = f"e.found = 1 AND LOWER({expr}) LIKE ?"
        params = [src, f"%{term}%"]
    sql = (f"SELECT e.doc_id, {expr} AS value, e.page, e.quote, e.needs_review FROM current_extractions e "
           f"WHERE e.field = ? AND {cond} ORDER BY e.doc_id")
    return sql, params, _rows(sql, params)

def compare(field, op, number):
    if field not in NUMERIC or op not in OPS:
        raise ValueError(f"compare needs a numeric field {sorted(NUMERIC)} and an operator {sorted(OPS)}")
    src, expr = FIELD_SQL[field]
    sql = (f"SELECT e.doc_id, {expr} AS value, e.page, e.quote, e.needs_review FROM current_extractions e "
           f"WHERE e.field = ? AND e.found = 1 AND {expr} {op} ? ORDER BY value DESC")
    return sql, [src, number], _rows(sql, [src, number])

def date_between(field, start, end):
    """Dates are stored as YYYY-MM-DD text, so a plain BETWEEN compares them correctly."""
    src, expr = FIELD_SQL[field]
    sql = (f"SELECT e.doc_id, {expr} AS value, e.page, e.quote, e.needs_review FROM current_extractions e "
           f"WHERE e.field = ? AND e.found = 1 AND {expr} BETWEEN ? AND ? ORDER BY value")
    return sql, [src, start, end], _rows(sql, [src, start, end])

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
    family = _rows("SELECT d.doc_id, d.doc_type, (SELECT value_norm FROM current_extractions WHERE doc_id = d.doc_id "
                   "AND field = 'effective_date') AS eff FROM documents d WHERE d.doc_id = ? OR d.master_id = ?",
                   (root[0], root[0]))
    # master first, then amendments by effective date
    family.sort(key=lambda r: (r[1] != "master", r[2] or "9999"))
    terms = {}
    for doc, doc_type, eff in family:
        if doc_type == "ancillary":
            continue                      # e.g. a power of attorney: related, but it doesn't change terms
        for field, (src, expr) in FIELD_SQL.items():
            if field in ("effective_date", "parties") and doc_type != "master" and doc != root_id:
                continue                  # the agreement's start date and parties come from the master;
                                          # an amendment's date is when ITS changes took effect
                                          # (an orphan amendment with no master here is its own root, so it keeps its own)
            row = con.execute(f"SELECT {expr}, e.page, e.quote FROM current_extractions e WHERE e.doc_id = ? AND e.field = ? "
                              f"AND e.found = 1", (doc, src)).fetchone()
            if row and row[0] not in (None, "none"):
                terms[field] = (row[0], doc, row[1], row[2])
    return family, terms

# ---------- Family level: the unit is the AGREEMENT, not the PDF ----------
def family_answer(tool, field, value=None, op=None, number=None, start=None, end=None):
    """Same questions as the SQL tools, asked of each family's terms in force.
    Amendments roll up into their master, so a cap stated only in the master still counts."""
    rows = []
    scope = tool != "date_between" and (start or end)   # "after 2000" etc.: limit WHICH agreements are counted
    for root in family_roots():
        family, terms = family_terms(root)
        if scope:
            eff = terms.get("effective_date", (None,))[0]
            if not eff or not ((start or "0000") <= eff <= (end or "9999")):
                continue                              # outside the date range, or date unknown (see scope_unknown)
        doc_type = next(t for d, t, _ in family if d == root)
        hit = terms.get(field)                       # (value, from_doc, page, quote) or None
        v = hit[0] if hit else None
        note = f"{len(family)} docs" + ("; master not in sample" if doc_type != "master" else "")
        if tool == "list_where":
            wants_none = str(value).lower() in ("none", "not found", "missing", "no")
            want = party_term(value) if field == "parties" else norm_value(value)
            keep = (v is None) if wants_none else (v is not None and want in str(v).lower())
        elif tool == "compare":
            keep = v is not None and eval_op(float(v), op, float(number))
        elif tool == "date_between":
            keep = v is not None and start <= str(v) <= end
        else:
            keep = True
        if keep:
            rows.append((root, v, hit, note))
    return rows

# A second, independent check for "not found" answers: does the agreement's own text contain
# language about this topic? If yes, extraction may have missed it. Keyword search vs. LLM extraction.
ABSENCE_CHECK = {
    "cap_type": '("liability" OR "liable") AND ("exceed" OR "limit" OR "limited" OR "limitation" '
                'OR "consequential" OR "maximum")',
}

def confirmed_absent(root, field):
    """Did a reviewer confirm this agreement really has none? (any family document marked 'absent')"""
    src = FIELD_SQL[field][0]
    return con.execute("SELECT 1 FROM reviews r JOIN documents d ON d.doc_id = r.doc_id WHERE r.field = ? "
                       "AND r.status = 'absent' AND (d.doc_id = ? OR d.master_id = ?)", (src, root, root)).fetchone() is not None

def absence_evidence(root, field):
    """Return (number of matching chunks, best chunk's heading, page) within this agreement's family."""
    query = ABSENCE_CHECK.get(field)
    if not query:
        return None
    docs = [r[0] for r in con.execute("SELECT doc_id FROM documents WHERE doc_id = ? OR master_id = ?", (root, root))]
    marks = ",".join("?" * len(docs))
    hits = con.execute(f"SELECT c.heading_path, c.page_start FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.chunk_id "
                       f"WHERE chunks_fts MATCH ? AND c.doc_id IN ({marks}) "
                       # a chunk whose HEADING is about liability is the best place to start reading
                       f"ORDER BY (LOWER(c.heading_path) LIKE '%liab%') DESC, bm25(chunks_fts)",
                       [query] + docs).fetchall()
    return (len(hits), hits[0][0].split(" > ")[-1][:50], hits[0][1]) if hits else (0, None, None)

def norm_value(v):
    """'New York law' -> 'new york': drop filler words the AI copies from the question."""
    v = re.sub(r"(?i)\b(law|laws|the|state of|commonwealth of)\b", " ", str(v))
    return re.sub(r"\s+", " ", v).strip().lower()

# Company suffixes people type but the stored name may not have: "Ingram Micro Inc" -> "ingram micro".
# Only the search term is trimmed. Matching is "contains", so "netgear" still finds "netgear, inc.".
SUFFIX = re.compile(r"(?i)[,\s]+(inc|incorporated|corp|corporation|co|company|llc|ltd|limited|plc|lp|llp|gmbh|ag|sa)\.?$")

def party_term(v):
    v = norm_value(v)
    while SUFFIX.search(v):          # loop handles stacked suffixes like "Co., Ltd."
        v = SUFFIX.sub("", v).strip()
    return v

def party_mentions(v):
    """A second check for an EMPTY party search: the name may be in the text but not in the parties field
    (a parent company, an affiliate, a former name). Returns one row per document: (doc_id, passages, heading, page, snippet)."""
    words = re.findall(r"[a-z0-9]+", party_term(v))
    if not words:
        return []
    phrase = '"' + " ".join(words) + '"'              # FTS5 phrase: the words together, in order
    hits = con.execute("SELECT c.doc_id, c.heading_path, c.page_start, c.text FROM chunks_fts f "
                       "JOIN chunks c ON c.chunk_id = f.chunk_id WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)",
                       (phrase,)).fetchall()
    by_doc = {}
    for doc, heading, page, text in hits:
        if doc not in by_doc:
            m = re.search(re.escape(words[0]), text, re.I)
            start = max(0, (m.start() if m else 0) - 40)
            by_doc[doc] = [doc, 0, heading, page, text[start:start + 240].strip()]
        by_doc[doc][1] += 1
    return [tuple(r) for r in sorted(by_doc.values(), key=lambda r: -r[1])]

def scope_unknown():
    """Agreements with no known effective date: a date-scoped answer can't include them, so say how many."""
    return [r for r in family_roots() if not family_terms(r)[1].get("effective_date")]

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
        "tool": {"type": "string", "enum": ["count_by", "list_where", "compare", "date_between", "terms_in_force"]},
        "field": {"type": "string", "enum": list(FIELD_SQL)},
        "value": {"type": ["string", "null"]},
        "op": {"type": ["string", "null"], "enum": sorted(OPS) + [None]},
        "number": {"type": ["number", "null"]},
        "doc_id": {"type": ["string", "null"]},
        "start": {"type": ["string", "null"]},
        "end": {"type": ["string", "null"]},
    },
    "required": ["tool", "field", "value", "op", "number", "doc_id", "start", "end"],
}
PLAN_PROMPT = """You translate a question about a contract database into ONE tool call. Do not answer the question.
Tools:
- count_by(field): how many contracts per value. "How many contracts per governing law?"
- list_where(field, value): which contracts have a value. value "none" means missing/no such clause.
  "Which contracts have no liability cap?" -> field cap_type, value none
- compare(field, op, number): numeric fields only (termination_notice in DAYS, cap_amount in dollars).
  "notice periods longer than 60 days" -> termination_notice > 60
- date_between(field, start, end): dates in a range, as YYYY-MM-DD. Only for effective_date.
  "agreements signed between 2000 and 2004" -> effective_date, start 2000-01-01, end 2004-12-31
  "contracts effective before 2005" -> start 1900-01-01, end 2004-12-31
- terms_in_force(doc_id): current terms of one agreement family, e.g. "terms in force for D001".
  If the question names a company instead of an ID ("notice period for Ingram"), set doc_id null,
  value = the company name, field = the term asked about (or parties if no term is named).
Any of count_by, list_where and compare can ALSO be limited to agreements effective in a date range: set start/end
(YYYY-MM-DD). "notice periods for agreements after 2000" -> count_by termination_notice, start 2000-01-01, end null.
Never drop part of the question: if it mentions a year or date range, set start/end.
Fields: governing_law, effective_date, parties, termination_notice (days), cap_type
(fixed_amount, fees_based, exclusion_only, none), cap_amount.
Set unused blanks to null."""

def plan(question):
    import ollama
    r = ollama.chat(model="qwen2.5:14b", format=PLAN_SCHEMA, options={"temperature": 0},
                    messages=[{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": question}])
    return json.loads(r["message"]["content"])

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def agreements_with(party):
    """Agreement roots whose parties include this name (suffixes ignored): 'ingram' -> ['D006']."""
    return [root for root, v, hit, note in family_answer("list_where", "parties", party)]

def repair_plan(p):
    """Fix predictable AI plan shapes instead of refusing them (each repair is a known model habit).
    Returns (plan, note) where note says what was changed, so the repair is visible to the user."""
    p = dict(p)
    if p.get("tool") == "terms_in_force" and not re.match(r"^D\d{3}$", str(p.get("doc_id") or "").upper()):
        # People name agreements by counterparty, not by our IDs: "termination notice for Ingram".
        # The model puts the name in doc_id or value. Code looks it up; it never guesses between several.
        name = p.get("doc_id") or p.get("value")
        roots = agreements_with(name) if name else []
        if len(roots) == 1:
            p.update(doc_id=roots[0], value=None)
            return p, f"found the agreement by party name: {name!r} is a party to {roots[0]}"
        p["party"] = name                              # validate_plan explains the 0 or many case
    if p.get("field") == "effective_date" and p.get("tool") in ("list_where", "count_by", "compare") \
            and (p.get("start") or p.get("end")):
        # "agreements executed between 2000 and 2004" planned as list_where + a date scope -> date_between
        p.update(tool="date_between", start=p.get("start") or "0001-01-01", end=p.get("end") or "9999-12-31",
                 value=None, op=None, number=None)
        return p, "turned a date filter on the wrong query type into a date-range query"
    return p, None

def validate_plan(p):
    """Code checks the AI's plan BEFORE running it. A plan that can't be right is refused, not run."""
    tool, field = p.get("tool"), p.get("field")
    if field not in FIELD_SQL:
        raise ValueError(f"unknown field {field!r}")
    if tool == "list_where":
        v = str(p.get("value") or "")
        if not v:
            raise ValueError("list_where needs a value")
        if len(re.findall(r"\d{4}", v)) >= 2:
            raise ValueError(f"value {v!r} looks like a date range, which needs a date-range query")
        if field == "effective_date":
            raise ValueError("dates can't be matched as text; ask for a date range instead (e.g. 'between 2000 and 2004')")
    if tool == "compare":
        if field not in NUMERIC or p.get("op") not in OPS or p.get("number") is None:
            raise ValueError("compare needs a numeric field, an operator and a number")
    if tool == "date_between":
        if field != "effective_date":
            raise ValueError("date_between only works on effective_date")
        if not (ISO.match(str(p.get("start"))) and ISO.match(str(p.get("end")))) or p["start"] > p["end"]:
            raise ValueError(f"date_between needs start <= end as YYYY-MM-DD, got {p.get('start')}..{p.get('end')}")
    for k in ("start", "end"):
        if tool != "date_between" and p.get(k) and not ISO.match(str(p[k])):
            raise ValueError(f"date scope {k}={p[k]!r} must be YYYY-MM-DD")
    if tool == "terms_in_force" and not re.match(r"^D\d{3}$", str(p.get("doc_id") or "").upper()):
        name = p.get("party")
        if not name:
            raise ValueError("name the agreement: a party such as 'Ingram Micro', or an ID like D006")
        roots = agreements_with(name)
        if roots:
            raise ValueError(f"{name!r} is a party to {len(roots)} agreements ({', '.join(roots)}); say which one")
        raise ValueError(f"no agreement lists {name!r} as a party")

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
    if tool == "date_between":
        return date_between(field, p["start"], p["end"])
    if tool == "terms_in_force":
        if not p["doc_id"]:
            raise ValueError("terms_in_force needs a doc_id")
        return terms_in_force(p["doc_id"].upper())

def print_mentions(value):
    """Never answer an empty party search with a bare zero: say where else the name appears."""
    m = party_mentions(value)
    if not m:
        print(f"  No agreement lists {value!r} as a party, and the name doesn't appear in any agreement's text.")
        return
    print(f"  No agreement lists {value!r} as a party, but {sum(r[1] for r in m)} passages in {len(m)} documents "
          f"mention it (a parent, affiliate or former name?):")
    for doc, n, heading, page, snippet in m:
        print(f"    {doc}  {n} passages; start at {heading.split(' > ')[-1][:40]}, p.{page}: \"{snippet[:130]}\"")

ZERO_WARNING = ("  Zero results can mean \"none exist\" OR \"the question didn't fit the tools.\"\n"
                "  Check the AI's plan above. Does it match what you asked? If not, rephrase or use --mode examples.")

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
    if not rows and p["field"] == "parties":
        print_mentions(p["value"])
    elif not rows:
        print(ZERO_WARNING)
    for doc, value, page, quote, review in rows:
        cite = f'p.{page}: "{(quote or "")[:90]}"' if quote else "(no clause found: unverified, could be a miss)"
        print(f"  {doc}  {str(value)[:30]:30} {cite}" + ("  [needs review]" if review else ""))
    print("\nCounted per DOCUMENT: amendments are listed on their own. Use --level family to count agreements.")
    print(accuracy_note(p["field"]))

def print_family(question, p, rows):
    print(f'\nQuestion: "{question}"   (exhaustive, per AGREEMENT: each master merged with its amendments)')
    print(f"AI's plan: {json.dumps({k: v for k, v in p.items() if v is not None})}\n")
    if p["tool"] == "count_by" and p.get("value"):
        # The AI asked for a count but also named a value ("how many ... New York?"): count_by can't use a value,
        # so code answers the question actually asked instead of dumping every state.
        want = norm_value(p["value"])
        match = [root for root, v, hit, note in rows if v is not None and want in str(v).lower()]
        print(f"{len(match)} of {len(rows)} agreements have {p['field']} = {p['value']}: {', '.join(match) or '-'}")
        print("(The AI picked count_by with a value; code filtered to that value.)")
    elif p["tool"] == "count_by":
        counts = {}
        for root, v, hit, note in rows:
            counts.setdefault(v if v is not None else "(not found)", []).append(root)
        for v, roots in sorted(counts.items(), key=lambda kv: -len(kv[1])):
            print(f"  {str(v)[:45]:45} {len(roots):3} agreements  ({', '.join(roots)})")
    elif p["tool"] == "list_where" and all(hit is None for _, _, hit, _ in rows) and p["field"] in ABSENCE_CHECK:
        # An absence question: sort the "not found" agreements by how likely the absence is real.
        check, likely, unknown = [], [], []
        for root, v, hit, note in rows:
            n, heading, page = absence_evidence(root, p["field"])
            if n:
                check.append((n, root, f"{n} chunks; start at {heading}, p.{page}", note))
            elif "master not in sample" in note:
                unknown.append(root)
            else:
                likely.append(root)
        print(f"{len(rows)} agreements with no {p['field']} extracted. Second check: the keyword index searched each "
              f"agreement's own text for limitation language.\n")
        print(f"{len(check)} to CHECK (limitation language found, but nothing extracted: possible miss):")
        for n, root, where, note in sorted(check, reverse=True):
            print(f"  {root}  {where}  [{note}]")
        print(f"\n{len(likely)} LIKELY NONE (no limitation language in any family document): {', '.join(likely) or '-'}")
        if unknown:
            print(f"{len(unknown)} UNKNOWN (amendment only; the cap may be in the master, which isn't in the sample): "
                  f"{', '.join(unknown)}")
    else:
        print(f"{len(rows)} agreements:")
        if not rows and p["field"] == "parties":
            print_mentions(p["value"])
        elif not rows:
            print(ZERO_WARNING)
        for root, v, hit, note in rows:
            if hit:
                print(f"  {root}  {str(v)[:28]:28} from {hit[1]}, p.{hit[2]}: \"{(hit[3] or '')[:70]}\"  [{note}]")
            else:
                print(f"  {root}  {'-':28} no clause found in any family document: unverified, could be a miss  [{note}]")
    print("\n" + accuracy_note(p["field"]))

# ============================================================
# Part 6c: ROUTER. Which desk answers this question?
#   "find all / how many"  -> SQL tools over the extraction table (exact, complete)
#   "find examples"        -> hybrid search over the text (best matches, NOT complete)
# ============================================================
# (label shown to the user, pattern)
EXHAUSTIVE = [("how many", r"\bhow many\b"), ("count", r"\bcount\b"), ("number of", r"\bnumber of\b"),
              ("all", r"\ball\b"), ("every", r"\bevery\b"), ("list", r"\blist\b"),
              ("which contracts", r"\bwhich (contracts|agreements)\b"),
              ("find contracts", r"\bfind (the )?(contracts|agreements)\b"), ("any contract", r"\bany (contracts?|agreements?)\b"),
              ("terms in force", r"\bterms in force\b"),
              ("more/less than", r"\b(longer|shorter|more|less|greater|fewer) than\b"),
              ("over/under a number", r"\b(over|under|above|below) \$?\d"), ("no cap", r"\bno (liability )?cap\b"),
              ("per state", r"\bper (state|governing law)\b")]
EXAMPLES = [("example", r"\bexamples?\b"), ("show me", r"\bshow me\b"), ("find a/some", r"\bfind (me )?(a|an|some)\b"),
            ("like", r"\blike\b"), ("similar", r"\bsimilar\b"), ("sample", r"\bsample\b"),
            ("what does ... say", r"\bwhat does .* say\b"), ("how do we", r"\bhow do (we|they)\b"),
            ("wording", r"\bwording\b"), ("language", r"\blanguage\b")]

# ---------- Coverage: can the SQL desk actually answer this? ----------
# Topics we extracted into fields (Step 4). SQL can list or count these completely.
EXTRACTED_TOPICS = [r"\bpart(y|ies)\b", r"counterpart", r"\bdate|effective|executed|signed|\b(19|20)\d\d\b",
                    r"govern|jurisdiction|\blaw\b|state", r"notice|terminat|renewal", r"\bcap\b|liability|limit",
                    r"amendment|master|family|in force"]
# Common clause topics we did NOT extract: (pattern, name shown to the user).
UNEXTRACTED_TOPICS = [("indemnif", "indemnification"), ("warrant", "warranties"), ("confidential", "confidentiality"),
                      ("non-?compete", "non-compete"), ("exclusiv", "exclusivity"), ("assign", "assignment"),
                      ("force majeure", "force majeure"), ("audit", "audit rights"), ("insurance", "insurance"),
                      ("intellectual property|\\bip\\b", "intellectual property"), ("non-?solicit", "non-solicitation"),
                      ("most favou?red", "most favored nation"), ("change of control", "change of control"),
                      ("royalt", "royalties"), ("rebate", "rebates"), ("arbitrat|dispute", "dispute resolution"),
                      ("licen[cs]e", "license grants"), ("payment terms", "payment terms"),
                      ("service level", "service levels"), ("subcontract", "subcontracting"),
                      ("data protection|privacy", "data protection"), ("trademark", "trademarks")]

def coverage_gap(question):
    """If a find-all question is about a topic we never extracted (and no extracted topic), name that topic."""
    q = question.lower()
    gap = next((name for pat, name in UNEXTRACTED_TOPICS if re.search(pat, q)), None)
    if gap and not any(re.search(t, q) for t in EXTRACTED_TOPICS):
        return gap
    return None

def route(question, mode="auto"):
    """Returns (desk, reason). Order: (1) explicit mode always wins, (2) rules on the wording,
    (3) if no rule fires, the trained model when it's at least 65% sure, (4) otherwise run BOTH desks.
    If both kinds of signal appear, the exhaustive desk wins: in legal work a missed contract
    (search returning only the top few) costs more than a slower, complete answer."""
    if mode == "all":
        return "sql", "you chose --mode all"
    if mode == "examples":
        return "search", "you chose --mode examples"
    q = question.lower()
    ex = [label for label, pat in EXHAUSTIVE if re.search(pat, q)]
    eg = [label for label, pat in EXAMPLES if re.search(pat, q)]
    clean = lambda labels: ", ".join(f"'{l}'" for l in labels[:2])
    if ex:
        gap = coverage_gap(question)
        if gap:
            return "search", (f"wording sounds exhaustive ({clean(ex)}), BUT '{gap}' isn't an extracted field, so a "
                              f"complete list isn't available. Showing best matches instead. (To support find-all "
                              f"questions about it, add it to the extraction schema. --mode all forces SQL.)")
        return "sql", f"wording sounds exhaustive ({clean(ex)})" + (" (also had example wording; exhaustive wins)" if eg else "")
    if eg:
        return "search", f"wording asks for examples ({clean(eg)})"
    # No rule fired: ask the trained model (Step 6d), but only trust it when it's fairly sure.
    model = _router_model()
    if model is not None:
        probs = dict(zip(model.classes_, model.predict_proba([question])[0]))
        best = max(probs, key=probs.get)
        if probs[best] >= 0.65:
            return best, f"no rule matched; trained model says {best} ({probs[best]:.0%} sure)"
        return "both", (f"no rule matched and the trained model is unsure (sql {probs.get('sql', 0):.0%}); "
                        "running BOTH desks so nothing is silently missed")
    return "search", "no clear signal, defaulted to examples. If you need EVERY match, re-run with --mode all"

_model_cache = []
def _router_model():
    """Load the trained router once, if it exists. The tool still works without it (rules only)."""
    if not _model_cache:
        try:
            import joblib
            _model_cache.append(joblib.load("data/router/router_model.joblib"))
        except Exception:
            _model_cache.append(None)
    return _model_cache[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="find examples (hybrid search)")
    s.add_argument("query")
    s.add_argument("--show-lists", action="store_true", help="also show the keyword-only and meaning-only rankings")
    k = sub.add_parser("ask", help="ask anything: the router picks search or SQL")
    k.add_argument("question")
    k.add_argument("--mode", choices=["auto", "examples", "all"], default="auto")
    k.add_argument("--level", choices=["family", "document"], default="family")
    a = sub.add_parser("all", help="find all / how many (SQL tools)")
    a.add_argument("question")
    a.add_argument("--level", choices=["family", "document"], default="family",
                   help="count agreements (master + amendments merged, default) or individual PDFs")
    args = ap.parse_args()
    if args.cmd == "ask":
        desk, reason = route(args.question, args.mode)
        label = {"sql": "SQL (find all)", "search": "SEARCH (find examples)", "both": "BOTH desks"}[desk]
        print(f"Router: -> {label}  because {reason}")
        args.query, args.show_lists = args.question, False
        if desk == "both":
            fused, kw, vec = hybrid_search(args.question)
            print_results(args.question, fused, kw, vec)
            args.cmd = "all"
        else:
            args.cmd = "all" if desk == "sql" else "search"
    if args.cmd == "all":
        p = plan(args.question)
        try:
            validate_plan(p)
            if args.level == "family" and p["tool"] in ("count_by", "list_where", "compare", "date_between"):
                print_family(args.question, p, family_answer(p["tool"], p["field"], p["value"], p["op"], p["number"],
                                                             p.get("start"), p.get("end")))
            else:
                print_all(args.question, p, run_plan(p))
        except ValueError as e:
            print(f"\nRefused to run the AI's plan: {e}.\nPlan was: {json.dumps(p)}\nTry rephrasing, or use --mode examples.")
    if args.cmd == "search":
        fused, kw, vec = hybrid_search(args.query)
        print_results(args.query, fused, kw, vec, args.show_lists)
