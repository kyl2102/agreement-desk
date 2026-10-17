"""Step 8d: a JSON API over the pipeline, for the front end built from the Claude Design canvas (web/).

Same logic as the Streamlit app: both call scripts/06_query.py, so answers are identical. The helpers below are
COPIED from app.py on purpose (Streamlit stays untouched before the interview); merging them into one shared module
is a clean follow-up.

Run:  uv run uvicorn api:app --port 8000      then open http://localhost:8000
"""
import csv
import datetime
import functools
import importlib.util
import io
import json
import re
import sqlite3
import sys
import threading

import pypdfium2 as pdfium
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import ImageDraw
from pydantic import BaseModel

# ---------- the pipeline ----------
spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
Q = importlib.util.module_from_spec(spec)
sys.argv = ["api"]
spec.loader.exec_module(Q)
Q.con = sqlite3.connect(Q.DB, check_same_thread=False)    # requests run in worker threads
LOCK = threading.Lock()                                    # one request touches the database at a time
SAMPLE = {r["doc_id"]: r for r in csv.DictReader(open("data/sample.csv", encoding="utf-8"))}

app = FastAPI(title="Agreement Desk API")

@functools.lru_cache(maxsize=256)
def plan_question(question):
    return Q.plan(question)

# ---------- display helpers (copied from app.py) ----------
FIELD_NAMES = {"parties": "Parties", "effective_date": "Effective date", "governing_law": "Governing law",
               "termination_notice": "Termination notice", "cap_type": "Liability cap",
               "cap_amount": "Liability cap amount", "liability_cap": "Liability cap"}
CAP_NAMES = {"fixed_amount": "Fixed dollar amount", "fees_based": "Tied to fees paid",
             "exclusion_only": "Excludes damage types only", "none": "No cap"}

def nice_value(field, v):
    if v is None:
        return ""
    if field == "cap_type":
        return CAP_NAMES.get(v, v)
    if field == "cap_amount":
        return f"${float(v):,.0f}"
    if field == "termination_notice":
        return f"{int(float(v))} days"
    if field == "effective_date":
        return nice_date(v)
    if field == "liability_cap" and str(v).startswith("{"):
        cap = json.loads(v)
        amount = f", ${cap['cap_amount']:,.0f}" if cap.get("cap_amount") else ""
        return CAP_NAMES.get(cap.get("cap_type"), cap.get("cap_type")) + amount
    return str(v)

SMALL = {"and", "of", "the", "to", "for", "a"}
SKIP = {"agmt", "undr", "serv", "agree", "to", "the", "ex", "between", "new"}

def _tc(s, shouting):
    out = []
    for i, w in enumerate(s.split()):
        if i and w.lower() in SMALL:
            out.append(w.lower())
        elif (not shouting and w.isupper() and len(w) <= 3) or any(ch.isdigit() for ch in w):
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)

def _kind(filename):
    words = re.findall(r"[A-Za-z]+", filename.rsplit(".", 1)[0])
    low = [w.lower() for w in words]
    if "powerof" in low or ("power" in low and "attorney" in low):
        return "Power of Attorney"
    idx = max((i for i, w in enumerate(low) if w in ("agreement", "addendum")), default=None)
    if idx is None:
        return "Agreement"
    pre = []
    for w in reversed(words[max(0, idx - 3):idx]):
        if w.lower() in SKIP or re.fullmatch(r"[IVXL]+", w):
            break
        pre.insert(0, w)
    while pre and pre[0].lower() in SMALL:
        pre.pop(0)
    return _tc(" ".join(pre + [words[idx]]), True)

def _short_party(p):
    p = re.sub(r",?\s*\b(inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|company|co|s\.a\.s|s\.a|gmbh|plc|n\.a)\.?$",
               "", p.strip(), flags=re.I).strip(" ,")
    return _tc(" ".join(p.split()[:3]), p.isupper())

@functools.lru_cache(maxsize=1)
def names():
    types = dict(Q.con.execute("SELECT doc_id, doc_type FROM documents").fetchall())
    parties = dict(Q.con.execute("SELECT doc_id, value_norm FROM extractions WHERE field='parties' AND found=1").fetchall())
    out = {}
    for d, r in SAMPLE.items():
        kind = _kind(r["filename"])
        if types.get(d) == "amendment" and "amend" not in kind.lower():
            kind = "Amendment to " + kind
        ps = [_short_party(x) for x in (parties.get(d) or "").split(";") if x.strip()][:2]
        out[d] = {"kind": kind, "parties": " and ".join(ps)}
    return out

def name(doc_id):
    n = names().get(doc_id)
    return f"{n['kind']}, {n['parties']}" if n and n["parties"] else (n["kind"] if n else doc_id)

def short_name(doc_id):
    """'Netgear · Distributor Agreement': counterparty first, the way people recall deals."""
    n = names().get(doc_id)
    if not n:
        return doc_id
    return f"{n['parties'].split(' and ')[-1]} · {n['kind']}" if n["parties"] else n["kind"]

def _squash(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def page_png(doc_id, page_no, snippet):
    doc = json.load(open(f"data/parsed/{doc_id}.json"))
    height = doc["pages"][str(page_no)]["size"]["height"]
    img = pdfium.PdfDocument(SAMPLE[doc_id]["pdf_path"])[page_no - 1].render(scale=2).to_pil().convert("RGB")
    draw, target = ImageDraw.Draw(img), _squash(snippet)[:120]
    for item in doc["texts"]:
        text = _squash(item["text"])
        if target and text and (target[:60] in text or text[:60] in target):
            for prov in item["prov"]:
                if prov["page_no"] == page_no:
                    b = prov["bbox"]
                    draw.rectangle((b["l"] * 2, (height - b["t"]) * 2, b["r"] * 2, (height - b["b"]) * 2),
                                   outline="#C8102E", width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@functools.lru_cache(maxsize=128)
def page_count(doc_id):
    return len(json.load(open(f"data/parsed/{doc_id}.json"))["pages"])

def family_docs(root):
    return [r[0] for r in Q.con.execute("SELECT doc_id FROM documents WHERE doc_id = ? OR master_id = ? "
                                        "ORDER BY doc_type != 'master', doc_id", (root, root))]

def root_of(doc_id):
    row = Q.con.execute("SELECT COALESCE(master_id, doc_id) FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return row[0] if row else doc_id

def search_within(docs, fts_query, limit=50):
    marks = ",".join("?" * len(docs))
    return Q.con.execute(
        f"SELECT c.doc_id, c.heading_path, c.page_start, c.text FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.chunk_id "
        f"WHERE chunks_fts MATCH ? AND c.doc_id IN ({marks}) ORDER BY c.doc_id, c.seq LIMIT ?",
        [fts_query] + docs + [limit]).fetchall()

def section_of(heading_path):
    return (heading_path or "").split(" > ")[-1][:80]

def accuracy_line(field):
    rows = Q.con.execute("SELECT eval_set, precision, recall, n FROM eval_results WHERE field = ?", (field,)).fetchall()
    if not rows:
        return "Not yet measured: spot-check against the source before relying on this."
    worst = min(rows, key=lambda r: r[2])
    misses = round((1 - worst[2]) * 10)
    measured = "; ".join(f"{p:.0%} precision, {r:.0%} recall on {s}" for s, p, r, n in rows)
    advice = ("Extraction was reliable on the contracts tested; open two or three sources for anything high-stakes."
              if worst[2] >= 0.9 else f"Spot-check before relying on this: the AI missed about {misses} in 10 on {worst[0]}.")
    return f"{advice} (Measured: {measured}.)"

@functools.lru_cache(maxsize=8)
def market_values(field):
    return tuple(v for r, v, h, n in Q.family_answer("count_by", field) if v is not None)

def market_note(field, v):
    if v is None or field not in ("termination_notice", "cap_type", "governing_law"):
        return ""
    others = market_values(field)
    if len(others) < 2:
        return ""
    if field == "termination_notice":
        days = sorted(float(x) for x in others)
        shorter = sum(d < float(v) for d in days)
        median = days[len(days) // 2]
        side = "above" if float(v) > median else "below" if float(v) < median else "at"
        return f"Longer than {shorter} of {len(days)}; {side} the median ({median:.0f} days)"
    same = sum(str(x) == str(v) for x in others)
    return f"Same as {same} of {len(others)} agreements"

def doc_label(doc_id, doc_type):
    if doc_type == "master":
        return "Master agreement"
    if doc_type == "amendment":
        row = Q.con.execute("SELECT filename FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        n = Q.amendment_number(row[0] if row else "")
        return "Amendment" + (f" #{n}" if n > 1 else "")
    return "Related document"

def nice_date(iso):
    try:
        d = datetime.date.fromisoformat(iso)
        return f"{d.strftime('%b')} {d.day}, {d.year}"
    except (TypeError, ValueError):
        return "date unknown"

def term_history(family, field, root):
    src, expr = Q.FIELD_SQL[field]
    steps, prev = [], None
    for n, (doc, doc_type, eff) in enumerate(family, 1):
        if doc_type == "ancillary" or (field in ("effective_date", "parties") and doc != root):
            continue
        row = Q.con.execute(f"SELECT {expr}, e.page FROM current_extractions e WHERE e.doc_id = ? AND e.field = ? "
                            f"AND e.found = 1", (doc, src)).fetchone()
        if not row or row[0] in (None, "none"):
            continue
        v = nice_value(field, row[0])
        what = "Set" if prev is None else ("Restated, no change" if v == prev else f"Changed from {prev}")
        steps.append({"n": n, "what": what, "doc": doc, "page": row[1]})
        prev = v
    return steps

# ---------- review queue (copied from app.py) ----------
CHECK_QUESTION = {
    "termination_notice": "Does the clause on page {p} give {v} notice to terminate?",
    "governing_law": "Does the clause on page {p} make {v} law govern the agreement?",
    "effective_date": "Does page {p} show the agreement taking effect on {v}?",
    "parties": "Are the parties {v}, as page {p} says?",
    "liability_cap": "Does the clause on page {p} cap liability as: {v}?",
}
FIELD_HINT = {"parties": "Legal names separated by ;", "effective_date": "YYYY-MM-DD",
              "governing_law": "e.g. New York", "termination_notice": "Number of days, e.g. 90"}

def review_queue():
    items = []
    for doc, field, value, quote, page, conf, check in Q.con.execute(
            "SELECT doc_id, field, value_norm, quote, page, confidence, quote_check FROM current_extractions "
            "WHERE needs_review = 1 ORDER BY doc_id"):
        why = ("The quote only partly matched the page text." if check == "partial" else
               "The AI's answer failed one of the automatic checks.")
        items.append({"doc": doc, "field": field, "why": why, "value": value, "quote": quote, "page": page})
    for root, v, hit, note in Q.family_answer("list_where", "cap_type", "none"):
        if Q.confirmed_absent(root, "cap_type"):
            continue
        n, heading, page = Q.absence_evidence(root, "cap_type")
        reviewed = Q.con.execute("SELECT 1 FROM reviews WHERE doc_id = ? AND field = 'liability_cap'", (root,)).fetchone()
        if n and not reviewed:
            docs = family_docs(root)
            chunk = Q.con.execute("SELECT text FROM chunks WHERE doc_id IN (%s) AND heading_path LIKE ? ORDER BY seq LIMIT 1"
                                  % ",".join("?" * len(docs)), docs + [f"%{heading}%"]).fetchone()
            items.append({"doc": root, "field": "liability_cap",
                          "why": f"The AI found no cap, but {n} passages mention limiting liability.",
                          "value": None, "quote": chunk[0] if chunk else "", "page": page})
    return items

def task_question(it):
    v = nice_value(it["field"], it["value"])
    if not v and it["field"] == "liability_cap":
        return "The AI found no liability cap. Do any of these passages limit liability?"
    if not v:
        return f"The AI found no {FIELD_NAMES.get(it['field'], it['field']).lower()}. Does the agreement state one?"
    return CHECK_QUESTION.get(it["field"], "Is {v} correct, per page {p}?").format(v=v, p=it["page"] or "?")

def answer_labels(it):
    """The answer buttons are the question's own answers, not generic Accept/Reject."""
    v = nice_value(it["field"], it["value"])
    term = FIELD_NAMES.get(it["field"], it["field"]).lower()
    if v:
        return {"yes": f"Yes, {v} is correct", "diff": "No, it says something else:",
                "none": "No, this agreement has no such clause"}
    if it["field"] == "liability_cap":
        return {"yes": None, "diff": "Yes, there is a cap:", "none": "No, there's no cap in this agreement"}
    return {"yes": None, "diff": f"Yes, the {term} is:", "none": f"No, it doesn't state a {term}"}

# ---------- answer builders: the same decisions as the Streamlit Ask page, returned as data ----------
def row(agreement_id, cells, doc, page, quote):
    return {"agreement": short_name(agreement_id), "id": agreement_id, "cells": cells,
            "doc": doc, "page": page or 1, "quote": quote or "", "pages": page_count(doc), "doc_name": short_name(doc)}

def mentions_answer(value):
    m = Q.party_mentions(value)
    if not m:
        return {"headline": f"No agreement lists {value} as a party, and the name doesn't appear in any agreement's text.",
                "columns": [], "rows": []}
    return {"headline": f"No agreement lists {value} as a party, but {sum(r[1] for r in m)} passages in {len(m)} "
                        f"documents mention it. It may be a parent company, affiliate or former name.",
            "columns": ["Agreement", "Mentions", "Where"],
            "rows": [row(d, [str(n), f"{section_of(h)}, p.{pg}"], d, pg, snip) for d, n, h, pg, snip in m]}

def complete_answer(question, level, drop_scope, how):
    p = plan_question(question)
    p, repaired = Q.repair_plan(p)
    how["plan"] = {k: v for k, v in p.items() if v is not None}
    if repaired:
        how["repaired"] = repaired
    out = {"badge": "complete", "filters": [], "notes": [], "columns": [], "rows": []}
    try:
        Q.validate_plan(p)
    except ValueError as e:
        if p.get("party") and not Q.agreements_with(p["party"]):
            return {**out, **mentions_answer(p["party"])}
        return {**out, "headline": "This question doesn't fit the available queries",
                "notes": [{"kind": "error", "text": f"{e}. Try rephrasing, or answer as Examples."}]}
    field = p["field"]
    label = FIELD_NAMES.get(field, field)
    years = re.findall(r"\b(?:19|20)\d\d\b", question)
    if years and not (p.get("start") or p.get("end")) and p["tool"] != "date_between":
        out["notes"].append({"kind": "warn", "text": f"Your question mentions {', '.join(years)}, but this answer is "
                             "not filtered by date. Try e.g. \"agreements effective after 2000\"."})
    if drop_scope:
        p = dict(p, start=None, end=None)
    out["filters"].append(label)
    if p.get("value"):
        out["filters"].append(f"= {p['value']}")
    if p.get("op") and p.get("number") is not None:
        out["filters"].append(f"{p['op']} {p['number']:g}")
    scoped = p["tool"] != "date_between" and (p.get("start") or p.get("end"))
    if p.get("start") or p.get("end"):
        out["filters"].append(f"effective {p.get('start') or 'any date'} to {p.get('end') or 'today'}")
    out["scoped"] = bool(scoped)
    if scoped:
        unknown = Q.scope_unknown()
        out["notes"].append({"kind": "info", "text": f"{len(unknown)} agreements have no known effective date and are "
                             f"left out: {', '.join(unknown)}."})
    out["filters"].append("per agreement" if level == "agreements" else "per document")
    out["accuracy"] = accuracy_line(field)

    if p["tool"] == "terms_in_force":
        family, terms = Q.terms_in_force(p["doc_id"].upper())
        root = next((d for d, t, _ in family if t == "master"), family[0][0] if family else p["doc_id"].upper())
        out["agreement_id"] = root
        if field in terms and field != "parties":
            v, d, pg, q = terms[field]
            note = market_note(field, v)
            out["headline"] = f"{label} for {short_name(root)}: {nice_value(field, v)}"
            out["subhead"] = note
            rows = [(field, terms[field])]
        else:
            out["headline"] = (f"No {label.lower()} was extracted for {short_name(root)}. Its other terms in force:"
                               if field not in terms and field != "parties" else f"Terms in force for {short_name(root)}")
            rows = list(terms.items())
        out["columns"] = ["Agreement", "Term", "Value", "Source"]
        out["rows"] = [row(root, [FIELD_NAMES.get(f, f), nice_value(f, v), f"p.{pg}"], d, pg, q) for f, (v, d, pg, q) in rows]
        return out

    if level == "documents" and p["tool"] != "count_by":
        sql, params, rows = Q.run_plan(p)
        how["sql"] = sql + f"\n-- params: {params}"
        if not rows and field == "parties":
            return {**out, **mentions_answer(p["value"])}
        out["headline"] = f"{len(rows)} documents match"
        out["subhead"] = "Amendments counted on their own."
        out["columns"] = ["Document", label, "Source"]
        out["rows"] = [row(d, [nice_value(field, v), f"p.{pg}"], d, pg, q) for d, v, pg, q, r in rows]
        return out

    rows = Q.family_answer(p["tool"], field, p["value"], p["op"], p["number"], p.get("start"), p.get("end"))
    total = len(Q.family_roots())
    if p["tool"] == "count_by":
        if p.get("value"):
            want = Q.norm_value(p["value"])
            match = [(r, v, h) for r, v, h, n in rows if v is not None and want in str(v).lower()]
            out["headline"] = f"{len(match)} of {len(rows)} agreements have {label.lower()} {p['value']}"
            out["columns"] = ["Agreement", label, "Source"]
            out["rows"] = [row(r, [nice_value(field, v), f"p.{h[2]}"], h[1], h[2], h[3]) for r, v, h in match]
            return out
        missing = f"No {label.lower()} found"
        counts = {}
        for r, v, h, n in rows:
            counts.setdefault(nice_value(field, v) or missing, []).append(r)
        ordered = sorted(counts.items(), key=lambda kv: (kv[0] == missing, -len(kv[1])))
        out["headline"] = f"{label} across {len(rows)} agreements"
        out["columns"] = [label, "Agreements", "Which"]
        out["groups"] = [{"value": v, "count": len(r), "agreements": [{"id": x, "name": short_name(x)} for x in r]}
                         for v, r in ordered]
        return out

    absence = p["tool"] == "list_where" and rows and all(h is None for _, _, h, _ in rows) and field in Q.ABSENCE_CHECK
    if absence:
        table = []
        for r, v, h, note in rows:
            n, heading, page = Q.absence_evidence(r, field)
            if Q.confirmed_absent(r, field):
                status, order = "Confirmed none by a reviewer", 3
            elif n:
                status, order = f"Check: {n} passage{'s' if n != 1 else ''} mention{'' if n != 1 else 's'} limiting liability", 0
            elif "master not in sample" in note:
                status, order = "Unknown: the master agreement isn't here", 2
            else:
                status, order = "Likely none: no limitation language found", 1
            table.append((order, row(r, [status], r, page if n else 1, "")))
        table.sort(key=lambda t: t[0])
        to_check = sum(o == 0 for o, _ in table)
        out["headline"] = f"{len(rows)} agreements have no {label.lower()} extracted"
        out["subhead"] = (f"{to_check} of them mention limiting liability somewhere, so those may be misses worth "
                          "checking. A second, independent check (keyword search of each agreement's text) sorted them.")
        out["columns"] = ["Agreement", "Status"]
        out["rows"] = [r for _, r in table]
        return out

    if not rows and field == "parties":
        return {**out, **mentions_answer(p["value"])}
    out["headline"] = f"{len(rows)} of {total} agreements match"
    if not rows:
        out["notes"].append({"kind": "warn", "text": "No agreements match. That can mean none exist, or that the "
                             "question didn't map to the data the way you meant. Check \"How this was answered\"."})
    out["columns"] = ["Agreement", label, "Source"]
    out["rows"] = [row(r, [nice_value(field, v), f"p.{h[2]}" if h else "none found"],
                       h[1] if h else r, h[2] if h else 1, (h[3] or "") if h else "") for r, v, h, n in rows]
    return out

def examples_answer(question, how):
    fused, kw, vec = Q.hybrid_search(question)
    how["ranks"] = [{"result": n, "agreement": Q.con.execute("SELECT doc_id FROM chunks WHERE chunk_id=?", (c,)).fetchone()[0],
                     "keyword rank": kw.index(c) + 1 if c in kw else None,
                     "meaning rank": vec.index(c) + 1 if c in vec else None} for n, c in enumerate(fused, 1)]
    results = []
    for cid in fused:
        doc_id, heading, pages, text = Q.describe(cid)
        page = int(re.search(r"\d+", pages).group())
        results.append({"agreement": short_name(doc_id), "doc": doc_id, "section": section_of(heading),
                        "doc_pages": page_count(doc_id),
                        "pages": pages.replace("pp.", "pages ").replace("p.", "page "), "page": page, "text": text[:600]})
    return {"badge": "examples", "results": results}

# ---------- endpoints ----------
class AskIn(BaseModel):
    question: str
    mode: str = "auto"            # auto | all | examples
    level: str = "agreements"     # agreements | documents
    drop_scope: bool = False

@app.post("/api/ask")
def ask(body: AskIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    with LOCK:
        desk, reason = Q.route(q, body.mode)
        how = {"routing": reason}
        out = {"question": q, "desk": desk, "how": how, "answers": []}
        m = re.search(r"'([^']+)' isn't an extracted field", reason)
        if m:
            out["coverage"] = (f"A complete list isn't available: {m.group(1)} isn't one of the fields this system "
                               "extracts. Here are the best-matching clauses instead.")
        # Each desk fails on its own: a planner hiccup must not take the search results down with it.
        if desk in ("sql", "both"):
            try:
                out["answers"].append(complete_answer(q, body.level, body.drop_scope, how))
            except Exception as e:
                out["answers"].append({"badge": "complete", "error": True, "headline": "The complete-list desk couldn't answer this",
                                       "filters": [], "rows": [], "columns": [],
                                       "notes": [{"kind": "error", "text": f"{type(e).__name__}: {e}. Is Ollama running?"}]})
        if desk in ("search", "both"):
            try:
                out["answers"].append(examples_answer(q, how))
            except Exception as e:
                out["answers"].append({"badge": "complete", "error": True, "headline": "Search couldn't answer this", "filters": [],
                                       "rows": [], "columns": [], "notes": [{"kind": "error", "text": f"{type(e).__name__}: {e}"}]})
        return out

@app.get("/api/meta")
def meta():
    with LOCK:
        return {"agreements": len(Q.family_roots()),
                "documents": Q.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "review_waiting": len(review_queue())}

@app.get("/api/agreements")
def agreements():
    with LOCK:
        return [{"id": r, "name": short_name(r)} for r in sorted(Q.family_roots(), key=short_name)]

@app.get("/api/agreement/{root}")
def agreement(root: str):
    with LOCK:
        root = root_of(root.upper())
        family, terms = Q.family_terms(root)
        if not family:
            raise HTTPException(404, "unknown agreement")
        master = next((d for d, t, _ in family if t == "master"), root)
        n = names().get(master, {"kind": master, "parties": ""})
        return {"id": root, "title": n["kind"], "parties": n["parties"],
                "orphan": not any(t == "master" for _, t, _ in family),
                "documents": [{"n": i, "id": d, "label": doc_label(d, t), "date": nice_date(e), "pages": page_count(d)}
                              for i, (d, t, e) in enumerate(family, 1)],
                "terms": [{"field": f, "term": FIELD_NAMES.get(f, f), "value": nice_value(f, v),
                           "history": term_history(family, f, root), "market": market_note(f, v),
                           "doc": d, "page": pg, "quote": q or ""} for f, (v, d, pg, q) in terms.items()]}

@app.get("/api/review")
def review():
    with LOCK:
        items = review_queue()
        done = Q.con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        return {"done": done, "items": [{**it, "agreement": short_name(root_of(it["doc"])), "term": FIELD_NAMES.get(it["field"]),
                                         "display_value": nice_value(it["field"], it["value"]),
                                         "question": task_question(it), "labels": answer_labels(it),
                                         "hint": (FIELD_HINT.get(it["field"], "") if it["field"] != "liability_cap"
                                                  else "fixed_amount, fees_based or exclusion_only"),
                                         "pages": page_count(it["doc"])} for it in items]}

class ReviewIn(BaseModel):
    doc: str
    field: str
    status: str                   # accepted | corrected | absent
    value: str | None = None
    note: str = ""
    page: int | None = None
    quote: str | None = None

@app.post("/api/review")
def save_review(body: ReviewIn):
    if body.status not in ("accepted", "corrected", "absent"):
        raise HTTPException(400, "status must be accepted, corrected or absent")
    with LOCK:
        Q.con.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
                      (body.doc, body.field, body.status, body.value if body.status == "corrected" else None,
                       (body.quote or "")[:300] or None if body.status == "corrected" else None,
                       body.page if body.status == "corrected" else None, body.note, "Ken"))
        Q.con.commit()
        return {"ok": True}

@app.get("/api/page/{doc_id}/{page_no}")
def page(doc_id: str, page_no: int, q: str = ""):
    if doc_id not in SAMPLE:
        raise HTTPException(404, "unknown document")
    total = page_count(doc_id)
    if not 1 <= page_no <= total:
        raise HTTPException(404, f"page must be 1 to {total}")
    return Response(page_png(doc_id, page_no, q), media_type="image/png")

@app.get("/api/pdf/{doc_id}")
def pdf(doc_id: str):
    if doc_id not in SAMPLE:
        raise HTTPException(404, "unknown document")
    return FileResponse(SAMPLE[doc_id]["pdf_path"], media_type="application/pdf")

@app.get("/api/search/{root}")
def search_agreement(root: str, q: str):
    """Search every document in one agreement (the reviewer's 'search the whole agreement')."""
    with LOCK:
        words = " OR ".join(f'"{w}"' for w in re.findall(r"[a-z0-9$]+", q.lower()))
        if not words:
            return []
        return [{"doc": d, "section": section_of(h), "page": pg, "text": t[:400]}
                for d, h, pg, t in search_within(family_docs(root_of(root.upper())), words)]

app.mount("/", StaticFiles(directory="web", html=True), name="web")
