"""Step 8: demo UI. A thin display layer over the existing pipeline.

All logic (routing, search, SQL tools, family terms, citations) lives in scripts/06_query.py.
This file only DISPLAYS it, so a more polished front end can replace this file later
without touching anything underneath.

Run:  uv run streamlit run app.py
"""
import csv
import importlib.util
import json
import re
import sqlite3
import sys

import pypdfium2 as pdfium
import streamlit as st
from PIL import ImageDraw

st.set_page_config(page_title="Legal Doc Tool", page_icon="⚖️", layout="wide")

# ---------- Load the pipeline once ----------
@st.cache_resource
def load_pipeline():
    spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
    q = importlib.util.module_from_spec(spec)
    sys.argv = ["app"]
    spec.loader.exec_module(q)
    # Streamlit runs the page in different threads; let the shared database connection be used from any of them.
    q.con = sqlite3.connect(q.DB, check_same_thread=False)
    return q

Q = load_pipeline()
SAMPLE = {r["doc_id"]: r for r in csv.DictReader(open("data/sample.csv", encoding="utf-8"))}

@st.cache_data(show_spinner=False)
def plan_question(question):
    return Q.plan(question)          # cached: asking the same question twice doesn't re-run the model

# ---------- Citations: show the PDF page with the quoted text boxed ----------
def squash(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def page_image(doc_id, page_no, snippet):
    """Render one PDF page and draw a box around every Docling text item on it that overlaps the snippet."""
    doc = json.load(open(f"data/parsed/{doc_id}.json"))
    height = doc["pages"][str(page_no)]["size"]["height"]
    img = pdfium.PdfDocument(SAMPLE[doc_id]["pdf_path"])[page_no - 1].render(scale=2).to_pil().convert("RGB")
    draw, target = ImageDraw.Draw(img), squash(snippet)[:120]
    for item in doc["texts"]:
        text = squash(item["text"])
        if not target or not text or not (target[:60] in text or text[:60] in target):
            continue
        for prov in item["prov"]:
            if prov["page_no"] == page_no:
                b = prov["bbox"]
                draw.rectangle((b["l"] * 2, (height - b["t"]) * 2, b["r"] * 2, (height - b["b"]) * 2),
                               outline="#d62728", width=4)
    return img

def show_source(doc_id, page_no, snippet, key):
    if page_no and st.toggle("Show source page", key=key):
        st.image(page_image(doc_id, int(page_no), snippet),
                 caption=f"{doc_id}, page {page_no}: {SAMPLE[doc_id]['filename'][:80]}")


# ---------- Whole-document access: a reviewer can't confirm something is absent from a fragment ----------
def family_docs(root):
    return [r[0] for r in Q.con.execute(
        "SELECT doc_id FROM documents WHERE doc_id = ? OR master_id = ? ORDER BY doc_type != 'master', doc_id", (root, root))]

def page_count(doc_id):
    return len(json.load(open(f"data/parsed/{doc_id}.json"))["pages"])

def search_within(docs, fts_query, limit=50):
    marks = ",".join("?" * len(docs))
    return Q.con.execute(
        f"SELECT c.doc_id, c.heading_path, c.page_start, c.text FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.chunk_id "
        f"WHERE chunks_fts MATCH ? AND c.doc_id IN ({marks}) ORDER BY c.doc_id, c.seq LIMIT ?",
        [fts_query] + docs + [limit]).fetchall()

def words_to_fts(text):
    words = [w for w in re.findall(r"[a-z0-9$.,]+", text.lower()) if w.strip(".,")]
    return " OR ".join(f'"{w.strip(".,")}"' for w in words)

def show_matches(matches, key):
    for n, (doc, heading, page, text) in enumerate(matches):
        st.markdown(f"**{doc}, p.{page}** · {heading.split(' > ')[-1][:70]}")
        st.caption(text[:400] + ("..." if len(text) > 400 else ""))

def full_document(root, key, highlight=None):
    """Search box + page-by-page viewer over every document in the agreement's family."""
    docs = family_docs(root)
    term = st.text_input("Search within this agreement (all family documents)", key=f"{key}-q",
                         placeholder="e.g. liable, exceed, terminate")
    if term:
        found = search_within(docs, words_to_fts(term))
        st.write(f"{len(found)} passages match" + (" (showing the first 50)" if len(found) == 50 else ""))
        show_matches(found, key)
    c1, c2, c3 = st.columns([2, 1, 1])
    doc = c1.selectbox("Document", docs, key=f"{key}-doc",
                       format_func=lambda d: f"{d}: {SAMPLE[d]['filename'][:60]}")
    total = page_count(doc)
    start = highlight[1] if highlight and highlight[0] == doc and highlight[1] else 1
    page = c2.number_input(f"Page (of {total})", min_value=1, max_value=total, value=min(start, total), key=f"{key}-pg")
    if c3.button("Open full PDF", key=f"{key}-open", help="Opens the original in Preview on this Mac"):
        import subprocess
        subprocess.run(["open", SAMPLE[doc]["pdf_path"]])
    snippet = highlight[2] if highlight and highlight[0] == doc and highlight[1] == page else ""
    st.image(page_image(doc, int(page), snippet), caption=f"{doc}, page {page} of {total}")

# ---------- Display helpers ----------
def show_search(question):
    fused, kw, vec = Q.hybrid_search(question)
    st.subheader("Examples (hybrid search)")
    st.caption("Best matches, not a complete list. Keyword and meaning search merged with Reciprocal Rank Fusion.")
    for n, cid in enumerate(fused, start=1):
        doc_id, heading, pages, text = Q.describe(cid)
        kw_rank = kw.index(cid) + 1 if cid in kw else "-"
        vec_rank = vec.index(cid) + 1 if cid in vec else "-"
        with st.expander(f"{n}. {doc_id}, {pages}: {heading}", expanded=n <= 2):
            st.write(text)
            st.caption(f"keyword rank {kw_rank} | meaning rank {vec_rank}")
            page = int(re.search(r"\d+", pages).group())
            show_source(doc_id, page, text, key=f"src-{cid}")

def show_sql(question, level):
    st.subheader("Complete answer (SQL over extracted data)")
    with st.spinner("The local model is planning the query..."):
        p = plan_question(question)
    with st.expander("AI's plan (the model only fills these blanks; code builds the SQL)"):
        st.json({k: v for k, v in p.items() if v is not None})
    try:
        Q.validate_plan(p)
    except ValueError as e:
        st.error(f"Refused to run the AI's plan: {e}. Try rephrasing, or ask for examples instead.")
        return
    field = p["field"]
    if p["tool"] == "terms_in_force":
        st.info("Terms in force are shown on the Agreement page.")
        return
    if level == "Agreements" and p["tool"] in ("count_by", "list_where", "compare", "date_between"):
        rows = Q.family_answer(p["tool"], field, p["value"], p["op"], p["number"], p.get("start"), p.get("end"))
        show_family_rows(p, rows)
    else:
        sql, params, rows = Q.run_plan(p)
        with st.expander("SQL that ran"):
            st.code(sql + f"\n-- params: {params}", language="sql")
        if p["tool"] == "count_by":
            st.dataframe([{"value": v, "documents": n} for v, n, _ in rows], hide_index=True)
        else:
            st.write(f"**{len(rows)} documents** (amendments counted on their own)")
            st.dataframe([{"doc": d, "value": v, "page": pg, "quote": (qt or "")[:150],
                           "needs review": "yes" if r else ""} for d, v, pg, qt, r in rows], hide_index=True)
            if not rows:
                st.warning(Q.ZERO_WARNING)
    st.caption(Q.accuracy_note(field).replace("  ", ""))

def show_family_rows(p, rows):
    field = p["field"]
    if p["tool"] == "count_by":
        counts = {}
        for root, v, hit, note in rows:
            counts.setdefault(v if v is not None else "(not found)", []).append(root)
        if p.get("value"):
            want = Q.norm_value(p["value"])
            match = [r for r, v, h, n in rows if v is not None and want in str(v).lower()]
            st.metric(f"Agreements with {field} = {p['value']}", f"{len(match)} of {len(rows)}")
            st.write(", ".join(match))
            return
        st.dataframe([{"value": v, "agreements": len(r), "which": ", ".join(r)} for v, r in
                      sorted(counts.items(), key=lambda kv: -len(kv[1]))], hide_index=True)
        return
    absence = p["tool"] == "list_where" and rows and all(h is None for _, _, h, _ in rows) and field in Q.ABSENCE_CHECK
    if absence:
        check, likely, unknown = [], [], []
        confirmed = []
        for root, v, hit, note in rows:
            if Q.confirmed_absent(root, field):
                confirmed.append(root)
                continue
            n, heading, page = Q.absence_evidence(root, field)
            if n:
                check.append({"agreement": root, "limitation chunks": n, "start reading at": heading, "page": page})
            elif "master not in sample" in note:
                unknown.append(root)
            else:
                likely.append(root)
        st.write(f"**{len(rows)} agreements** with no {field} extracted. A second check searched each one's text:")
        st.markdown(f"🔎 **{len(check)} to check:** limitation language found, but nothing extracted (possible miss)")
        st.dataframe(sorted(check, key=lambda r: -r["limitation chunks"]), hide_index=True)
        if confirmed:
            st.markdown(f"🧑‍⚖️ **{len(confirmed)} confirmed none by a reviewer:** {', '.join(confirmed)}")
        st.markdown(f"✅ **{len(likely)} likely none:** {', '.join(likely) or '-'}")
        if unknown:
            st.markdown(f"❔ **{len(unknown)} unknown** (amendment only, master not in sample): {', '.join(unknown)}")
        return
    st.write(f"**{len(rows)} agreements** (each master merged with its amendments)")
    if not rows:
        st.warning(Q.ZERO_WARNING)
    table = [{"agreement": root, "value": v, "from": hit[1] if hit else "", "page": hit[2] if hit else "",
              "quote": (hit[3] or "")[:150] if hit else "no clause found (unverified)", "family": note}
             for root, v, hit, note in rows]
    st.dataframe(table, hide_index=True)

# ============================================================
# Pages
# ============================================================
def page_ask():
    st.title("Ask the agreements")
    st.caption("\"Find all / how many\" questions are answered from extracted data with SQL. "
               "\"Find examples\" questions use hybrid search. Every answer cites its source.")
    examples = ["Which contracts have no liability cap?", "How many agreements are governed by New York law?",
                "Show me examples of termination for convenience", "Agreements executed between 2000 and 2004",
                "Notice periods longer than 60 days", "Section 14.A"]
    pick = st.selectbox("Try an example, or type your own below", [""] + examples)
    question = st.text_input("Question", value=pick)
    c1, c2 = st.columns(2)
    mode = c1.radio("Mode", ["auto", "examples", "all"], horizontal=True,
                    help="'all' forces a complete answer from SQL; 'examples' forces search.")
    level = c2.radio("Count", ["Agreements", "Documents"], horizontal=True,
                     help="Agreements merge each master with its amendments.")
    if not question:
        return
    desk, reason = Q.route(question, mode)
    label = {"sql": "SQL (find all)", "search": "Search (find examples)", "both": "Both desks"}[desk]
    # A coverage gap is a warning, not routine info: the user asked for something complete we can't give.
    (st.warning if "isn't an extracted field" in reason else st.info)(f"**Router → {label}** because {reason}")
    if desk in ("sql", "both"):
        show_sql(question, level)
    if desk in ("search", "both"):
        show_search(question)

def page_agreement():
    st.title("Agreement")
    roots = Q.family_roots()
    root = st.selectbox("Agreement", roots, format_func=lambda d: f"{d}: {SAMPLE[d]['filename'][:90]}")
    family, terms = Q.family_terms(root)
    st.subheader("Family")
    st.caption("Master first, then amendments by date. Linked in Step 3 with explainable rules.")
    reasons = dict(Q.con.execute("SELECT doc_id, family_reason FROM documents").fetchall())
    st.dataframe([{"doc": d, "type": t, "effective": e, "linked because": reasons.get(d) or "",
                   "file": SAMPLE[d]["filename"][:70]} for d, t, e in family], hide_index=True)
    st.subheader("Terms in force")
    st.caption("A value stated in a later amendment overrides the master. Dates and parties come from the master.")
    if not terms:
        st.write("No terms extracted.")
        return
    st.dataframe([{"field": f, "value": v, "from": d, "page": pg, "quote": (qt or "")[:160]}
                  for f, (v, d, pg, qt) in terms.items()], hide_index=True)
    st.subheader("Check a citation")
    field = st.selectbox("Show the source of", list(terms))
    v, d, pg, qt = terms[field]
    st.write(f"**{field} = {v}**, from {d}, page {pg}")
    st.write(f"> {qt}")
    if pg:
        st.image(page_image(d, int(pg), qt), caption=f"{d}, page {pg}. Red boxes: the text Docling matched to the quote.")
    st.subheader("Read the agreement")
    full_document(root, key="ag-doc", highlight=(d, pg, qt))


# ============================================================
# Review and correct (human-in-the-loop)
# ============================================================
FIELD_HINT = {"parties": "Legal names separated by ; ", "effective_date": "YYYY-MM-DD",
              "governing_law": "State or country, e.g. New York", "termination_notice": "Days, e.g. 90"}
CAP_TYPES = ["fixed_amount", "fees_based", "exclusion_only", "none"]

def review_queue():
    """Two sources: answers the pipeline flagged, and 'no cap found' agreements whose text has limitation language."""
    items = []
    for doc, field, value, quote, page, conf in Q.con.execute(
            "SELECT doc_id, field, value_norm, quote, page, confidence FROM current_extractions "
            "WHERE needs_review = 1 ORDER BY doc_id"):
        items.append({"doc": doc, "field": field, "why": f"flagged by the pipeline (confidence {conf})",
                      "value": value, "quote": quote, "page": page})
    for root, v, hit, note in Q.family_answer("list_where", "cap_type", "none"):
        if Q.confirmed_absent(root, "cap_type"):
            continue
        n, heading, page = Q.absence_evidence(root, "cap_type")
        if n and Q.con.execute("SELECT 1 FROM reviews WHERE doc_id = ? AND field = 'liability_cap'", (root,)).fetchone() is None:
            chunk = Q.con.execute("SELECT text FROM chunks WHERE doc_id = ? AND heading_path LIKE ? ORDER BY seq LIMIT 1",
                                  (root, f"%{heading}%")).fetchone()
            items.append({"doc": root, "field": "liability_cap",
                          "why": f"no cap extracted, but {n} chunks mention limiting liability",
                          "value": None, "quote": chunk[0][:600] if chunk else "", "page": page})
    return items

def save_review(doc, field, status, value, quote, page, note, reviewer):
    Q.con.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
                  (doc, field, status, value, quote, page, note, reviewer))
    Q.con.commit()

def page_review():
    st.title("Review queue")
    st.caption("Attorneys confirm or correct what the AI extracted. Corrections are stored separately from the "
               "model's output (with reviewer and time) and every answer uses them immediately.")
    reviewer = st.sidebar.text_input("Reviewer", value="Ken")
    items = review_queue()
    done = Q.con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    c1, c2 = st.columns(2)
    c1.metric("Waiting for review", len(items))
    c2.metric("Reviewed so far", done)
    if not items:
        st.success("Queue is empty.")
    for i, it in enumerate(items):
        with st.expander(f"{it['doc']} · {it['field']}: {it['why']}", expanded=i == 0):
            st.write(f"**AI's answer:** {it['value'] if it['value'] is not None else '(nothing extracted)'}")
            if it["quote"]:
                st.write(f"> {it['quote'][:600]}")
            show_source(it["doc"], it["page"], it["quote"], key=f"rv-src-{i}")
            root = Q.con.execute("SELECT COALESCE(master_id, doc_id) FROM documents WHERE doc_id = ?",
                                 (it["doc"],)).fetchone()[0]
            if it["field"] == "liability_cap" and it["value"] is None:
                evidence = search_within(family_docs(root), Q.ABSENCE_CHECK["cap_type"])
                with st.expander(f"All {len(evidence)} passages that mention limiting liability"):
                    show_matches(evidence, key=f"ev-{i}")
            if st.toggle("Check the whole agreement", key=f"rv-full-{i}",
                         help="Search and page through every family document before deciding"):
                full_document(root, key=f"rv-doc-{i}", highlight=(it["doc"], it["page"], it["quote"]))
            st.divider()
            if it["field"] == "liability_cap":
                cap_type = st.selectbox("Correct cap type", CAP_TYPES, key=f"ct-{i}")
                amount = st.number_input("Cap amount in dollars (fixed_amount only, 0 = none)", min_value=0.0,
                                         step=1000.0, key=f"ca-{i}")
                corrected = json.dumps({"cap_type": cap_type, "cap_amount": amount or None})
            else:
                corrected = st.text_input(f"Correct value ({FIELD_HINT.get(it['field'], '')})",
                                          value=it["value"] or "", key=f"cv-{i}")
            note = st.text_input("Note (why)", key=f"nt-{i}")
            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Accept AI answer", key=f"ok-{i}", disabled=it["value"] is None):
                save_review(it["doc"], it["field"], "accepted", None, None, None, note, reviewer); st.rerun()
            if b2.button("✏️ Save correction", key=f"fix-{i}"):
                save_review(it["doc"], it["field"], "corrected", corrected, it["quote"][:300] if it["quote"] else None,
                            it["page"], note, reviewer); st.rerun()
            if b3.button("🚫 Confirm not present", key=f"no-{i}"):
                save_review(it["doc"], it["field"], "absent", None, None, None, note, reviewer); st.rerun()
    st.subheader("Audit trail")
    log = Q.con.execute("SELECT reviewed_at, reviewer, doc_id, field, status, value, note FROM reviews "
                        "ORDER BY reviewed_at DESC").fetchall()
    if log:
        st.dataframe([dict(zip(["when", "reviewer", "doc", "field", "decision", "value", "note"], r)) for r in log],
                     hide_index=True)
        if st.button("Undo the most recent review"):
            last = log[0]
            Q.con.execute("DELETE FROM reviews WHERE doc_id = ? AND field = ?", (last[2], last[3])); Q.con.commit()
            st.rerun()

# ============================================================
# Term benchmarking: what have we agreed to before?
# ============================================================
def page_benchmark():
    import altair as alt
    import pandas as pd
    st.title("Term benchmarking")
    st.caption("How a term is distributed across the agreements, per agreement (each master merged with its "
               "amendments, reviewer corrections applied). Use it to ground a negotiation position in past deals.")
    labels = {"termination_notice": "Termination notice (days)", "cap_type": "Liability cap type",
              "governing_law": "Governing law"}
    field = st.selectbox("Term", list(labels), format_func=labels.get)
    rows = Q.family_answer("count_by", field)
    found = [(root, v, hit, note) for root, v, hit, note in rows if v is not None]
    df = pd.DataFrame([{"value": str(int(float(v))) if field == "termination_notice" else str(v), "agreement": root}
                       for root, v, hit, note in found])
    c1, c2, c3 = st.columns(3)
    c1.metric("Agreements with a value", f"{len(found)} of {len(rows)}")
    if field == "termination_notice" and len(found):
        days = sorted(int(float(v)) for _, v, _, _ in found)
        c2.metric("Median notice", f"{days[len(days) // 2]} days")
        c3.metric("Range", f"{days[0]} to {days[-1]} days")
    if df.empty:
        st.write("No values extracted.")
        return
    counts = df.groupby("value").size().reset_index(name="agreements")
    order = sorted(counts["value"], key=lambda x: float(x)) if field == "termination_notice" else \
        list(counts.sort_values("agreements", ascending=False)["value"])
    chart = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2a78d6", size=28).encode(
        x=alt.X("value:N", sort=order, title=labels[field]),
        y=alt.Y("agreements:Q", title="Agreements", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("value:N", title=labels[field]), alt.Tooltip("agreements:Q", title="Agreements")],
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)
    st.caption(Q.accuracy_note(field).replace("  ", ""))
    pick = st.selectbox("Show the agreements with", order)
    for root, v, hit, note in found:
        shown = str(int(float(v))) if field == "termination_notice" else str(v)
        if shown == pick:
            st.write(f"**{root}** ({note}): from {hit[1]}, p.{hit[2]}")
            st.write(f"> {(hit[3] or '')[:300]}")
            show_source(hit[1], hit[2], hit[3], key=f"bm-{root}")


# ============================================================
# Quality: how good is it, measured?
# ============================================================
SERIES = {"precision": "#2a78d6", "recall": "#eb6834"}   # validated palette, categorical slots 1 and 2

def page_quality():
    import altair as alt
    import pandas as pd
    st.title("Quality")
    st.caption("Every number here was measured, not assumed. Model output is scored on its own; reviewer "
               "corrections are not counted in the model's scores.")

    st.subheader("Extraction accuracy by field")
    rows = Q.con.execute("SELECT eval_set, field, precision, recall, n, prompt_version, run_at FROM eval_results").fetchall()
    if not rows:
        st.write("Run scripts/07_evaluate.py and scripts/07b_holdout.py to populate this.")
    else:
        df = pd.DataFrame(rows, columns=["set", "field", "precision", "recall", "n", "prompt", "run_at"])
        chosen = st.radio("Answer key", sorted(df["set"].unique()), horizontal=True)
        d = df[df["set"] == chosen].melt(id_vars=["field"], value_vars=["precision", "recall"],
                                         var_name="measure", value_name="score")
        chart = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=18).encode(
            x=alt.X("field:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="measure:N",
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("measure:N", title=None, scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                            legend=alt.Legend(orient="top")),
            tooltip=["field", "measure", alt.Tooltip("score:Q", format=".0%")],
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df[df["set"] == chosen][["field", "precision", "recall", "n", "prompt", "run_at"]]
                     .style.format({"precision": "{:.0%}", "recall": "{:.0%}"}), hide_index=True)

    st.subheader("What each fix bought (recorded runs, Oct 16)")
    st.caption("v2 fixed prompt and checker bugs; v3 fixed retrieval and prompts and was tested on 30 unseen contracts.")
    st.dataframe([
        {"measure": "Review flags (of 200 answers)", "v1": "48", "v2": "7", "v3": "8"},
        {"measure": "My 10: all fields, precision / recall", "v1": "-", "v2": "92% / 88%", "v3": "92% / 92%"},
        {"measure": "Holdout: parties", "v1": "-", "v2": "85% / 76%", "v3": "93% / 93%"},
        {"measure": "Holdout: effective date", "v1": "-", "v2": "81% / 68%", "v3": "86% / 76%"},
        {"measure": "Holdout: governing law", "v1": "-", "v2": "88% / 95%", "v3": "92% / 100%"},
        {"measure": "Errors caught by review (my 10)", "v1": "-", "v2": "0 of 7", "v3": "1 of 4"},
    ], hide_index=True)

    st.subheader("Search: which method finds the right clause? (10 test questions)")
    st.dataframe([
        {"method": "keyword only (BM25)", "hit@5": "70%", "precision@5": "26%", "good at": "exact references, numbers"},
        {"method": "meaning only (embeddings)", "hit@5": "60%", "precision@5": "38%", "good at": "plain-English questions"},
        {"method": "hybrid (RRF)", "hit@5": "80%", "precision@5": "44%", "good at": "both; next step is a reranker"},
    ], hide_index=True)

    st.subheader("Routing (39 held-out questions)")
    st.dataframe([
        {"router": "hand rules", "accuracy": "92%", "costly misses (find-all sent to search)": 3},
        {"router": "trained model", "accuracy": "95%", "costly misses (find-all sent to search)": 1},
    ], hide_index=True)
    st.caption("Production uses both: explicit mode, then rules, then the model if 65%+ sure, else run both desks.")

    st.subheader("Human review so far")
    stats = dict(Q.con.execute("SELECT status, COUNT(*) FROM reviews GROUP BY status").fetchall())
    total = sum(stats.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviewed", total)
    c2.metric("Accepted as is", stats.get("accepted", 0))
    c3.metric("Corrected", stats.get("corrected", 0))
    c4.metric("Confirmed absent", stats.get("absent", 0))
    if total:
        st.caption(f"Override rate: {stats.get('corrected', 0) / total:.0%} of reviewed items were changed by a human. "
                   "Tracked over time, this is the most direct measure of how much to trust the AI.")

# ============================================================
# Production gaps: what this demo is NOT
# ============================================================
def page_gaps():
    st.title("From demo to production")
    st.caption("What this local demo does at each layer, and what a real agreement platform would add.")
    st.dataframe([
        {"layer": "1. Ingestion", "this demo": "40 PDFs from a fixed sample; SHA-256 per file; re-runs skip done work",
         "production adds": "Connectors to the DMS and deal rooms; incremental, event-driven updates; source permissions copied"},
        {"layer": "2. Parsing", "this demo": "Docling; per-document quality signals (headings, chars per item, seconds per page)",
         "production adds": "Fallback parsers and OCR routing for bad parses; models pre-downloaded to approved storage"},
        {"layer": "3. Families", "this demo": "Rules: parties + type + date, with reasons; 12 of 12 on known answers",
         "production adds": "AI extraction of references, corporate name-change lookups, side letters, partial amendments"},
        {"layer": "4. Raw store", "this demo": "Original PDFs untouched in a folder",
         "production adds": "Versioned object storage, retention and legal hold"},
        {"layer": "5. Metadata", "this demo": "SQLite; model output and human reviews in separate tables",
         "production adds": "Managed database or lakehouse with time travel (what did we believe on a date?)"},
        {"layer": "6. Indexes", "this demo": "FTS5 + brute-force vectors in one file",
         "production adds": "Search engine or vector database at scale; re-embedding plan when the model changes"},
        {"layer": "8. Extraction", "this demo": "5 fields, local 14B model, quote checks, review queue",
         "production adds": "Attorney-defined schema (carve-outs, indemnification, MFN...), stronger model, prompt and model versioning in CI"},
        {"layer": "9. Query", "this demo": "Router + parameterized SQL tools + hybrid search + coverage check",
         "production adds": "Reranker, query rewriting, metadata filters, permission-aware retrieval"},
        {"layer": "11. Evaluation", "this demo": "10 hand labels + 30-contract holdout; search and router evals",
         "production adds": "Two-attorney gold sets, agreement scores, regression tests on every change, monitoring"},
        {"layer": "12. Governance", "this demo": "Runs fully offline; review audit trail",
         "production adds": "Entitlements enforced inside the index, query audit logs, lineage, vendor terms (privilege)"},
    ], hide_index=True)

PAGES = {"Ask": page_ask, "Agreement": page_agreement, "Review queue": page_review, "Benchmarking": page_benchmark,
         "Quality": page_quality, "Production gaps": page_gaps}
choice = st.sidebar.radio("Screens", list(PAGES))
st.sidebar.caption("Local demo: Docling, qwen2.5:14b and nomic-embed-text via Ollama, SQLite. No data leaves this Mac.")
PAGES[choice]()
