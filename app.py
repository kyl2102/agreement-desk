"""Step 8: demo UI. A thin display layer over the existing pipeline.

All logic (routing, search, SQL tools, family terms, reviews) lives in scripts/06_query.py.
This file only DISPLAYS it, so a more polished front end can replace it later without touching anything underneath.
Design: answer first, sources one click away, technical detail tucked under "How this was answered".

Run:  uv run streamlit run app.py
"""
import csv
import importlib.util
import json
import re
import sqlite3
import subprocess
import sys

import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
from PIL import ImageDraw

st.set_page_config(page_title="Agreement Desk", page_icon="⚖️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
/* Tokens from the Claude Design canvas: cool paper, ink, one ledger-blue accent, citation red. */
:root { --paper:#F6F7F9; --card:#FFFFFF; --ink:#1C2333; --muted:#4A5366; --line:#D9DEE7; --blue:#2A5DB0;
        --blue-ink:#1C3F7A; --blue-tint:#E8EEF8; --red:#C8102E; --amber-ink:#8A3B00; --amber-tint:#FDF1E6; }
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, input, textarea, button { font-family: 'Public Sans', 'Helvetica Neue', sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; font-weight: 600 !important; letter-spacing: -0.01em; color: var(--ink); }
h1 { font-size: 1.9rem !important; } h2 { font-size: 1.35rem !important; } h3 { font-size: 1.15rem !important; }
.block-container { padding-top: 1rem; max-width: 1360px; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
/* white cards on grey paper */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { background: var(--card); border-color: var(--line) !important; border-radius: 10px; }
.brand { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.3rem; padding-top: 0.35rem; }
.topline { border-bottom: 1px solid var(--line); margin: 0.2rem 0 1.2rem; }
.answer { font-size: 1.05rem; line-height: 1.5; margin: 0.3rem 0 0.8rem; }
.headline { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.7rem; line-height: 1.25; margin: 0.4rem 0 0.7rem; color: var(--ink); text-wrap: pretty; }
.muted { color: var(--muted); font-size: 0.92rem; }
.result-title { font-weight: 600; margin-bottom: 0.1rem; }
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin: 0.2rem 0.5rem 0.3rem 0; }
.badge-complete { background: var(--blue-tint); color: var(--blue-ink); border: 1px solid #A9BEE0; }
.badge-examples { background: var(--amber-tint); color: var(--amber-ink); border: 1px solid #F0C9A4; }
.chip { display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.86rem; background: var(--card); color: var(--ink); margin: 0 0.35rem 0.35rem 0; border: 1px solid #B7C0CF; }
.quote { background: var(--paper); border-radius: 8px; padding: 0.7rem 0.9rem; font-family: 'Source Serif 4', Georgia, serif; font-size: 1rem; line-height: 1.55; color: var(--ink); margin: 0.4rem 0 0.7rem; }
.quote mark { background: #FFE08A; color: var(--ink); padding: 0 2px; }
.task-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--blue-ink); margin-top: 0.4rem; }
.task { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.5rem; line-height: 1.3; margin: 0.2rem 0 0.6rem; }
.flag { background: var(--amber-tint); border: 1px solid #F0C9A4; border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.92rem; color: #3A2410; margin: 0.3rem 0 0.8rem; }
.docstrip { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0.4rem 0 0.2rem; }
.doc { display: flex; align-items: center; gap: 10px; background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 6px 14px 6px 6px; }
.num { width: 24px; height: 24px; border-radius: 12px; background: var(--ink); color: #fff; font-size: 12px; font-weight: 700; display: grid; place-items: center; }
.arrow { color: #8A93A5; }
.st-key-nav-home button p { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.3rem; color: var(--ink); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Pipeline and data
# ============================================================
@st.cache_resource
def load_pipeline():
    spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
    q = importlib.util.module_from_spec(spec)
    sys.argv = ["app"]
    spec.loader.exec_module(q)
    q.con = sqlite3.connect(q.DB, check_same_thread=False)   # Streamlit reruns pages in different threads
    return q

Q = load_pipeline()
SAMPLE = {r["doc_id"]: r for r in csv.DictReader(open("data/sample.csv", encoding="utf-8"))}

@st.cache_data(show_spinner=False)
def plan_question(question):
    return Q.plan(question)

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
    if field == "liability_cap" and str(v).startswith("{"):
        cap = json.loads(v)
        amount = f", ${cap['cap_amount']:,.0f}" if cap.get("cap_amount") else ""
        return CAP_NAMES.get(cap.get("cap_type"), cap.get("cap_type")) + amount
    return str(v)

# ---------- Readable agreement names instead of SEC filenames ----------
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

@st.cache_data
def names():
    types = dict(Q.con.execute("SELECT doc_id, doc_type FROM documents").fetchall())
    parties = dict(Q.con.execute("SELECT doc_id, value_norm FROM extractions WHERE field='parties' AND found=1").fetchall())
    out = {}
    for d, r in SAMPLE.items():
        kind = _kind(r["filename"])
        if types.get(d) == "amendment" and "amend" not in kind.lower():
            kind = "Amendment to " + kind
        ps = [_short_party(x) for x in (parties.get(d) or "").split(";") if x.strip()][:2]
        out[d] = f"{kind}, {' and '.join(ps)}" if ps else kind
    return out

def name(doc_id):
    return names().get(doc_id, doc_id)

# ============================================================
# Sources: the PDF page, with the cited text boxed
# ============================================================
def _squash(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

@st.cache_data(show_spinner=False, max_entries=64)
def page_image(doc_id, page_no, snippet):
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
    return img

def page_count(doc_id):
    return len(json.load(open(f"data/parsed/{doc_id}.json"))["pages"])

def family_docs(root):
    return [r[0] for r in Q.con.execute("SELECT doc_id FROM documents WHERE doc_id = ? OR master_id = ? "
                                        "ORDER BY doc_type != 'master', doc_id", (root, root))]

def root_of(doc_id):
    return Q.con.execute("SELECT COALESCE(master_id, doc_id) FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()[0]

def source_panel(doc_id, page, snippet, key, docs=None):
    """The right-hand panel: one PDF page, with the cited text boxed. Page through it, or open the original."""
    docs = docs or [doc_id]
    if len(docs) > 1:
        doc_id = st.selectbox("Document", docs, index=docs.index(doc_id) if doc_id in docs else 0,
                              format_func=name, key=f"{key}-d")
    total = page_count(doc_id)
    c1, c2 = st.columns([1, 1])
    p = c1.number_input(f"Page, of {total}", 1, total, min(int(page or 1), total), key=f"{key}-p-{doc_id}-{page}")   # page in the key: a new citation resets the viewer to its page
    if c2.button("Open original PDF", key=f"{key}-o", use_container_width=True):
        subprocess.run(["open", SAMPLE[doc_id]["pdf_path"]])
    st.image(page_image(doc_id, int(p), snippet if int(p) == int(page or 1) else ""), use_container_width=True)
    st.caption(f"{name(doc_id)}. Red box: the cited text.")

def search_within(docs, fts_query, limit=50):
    marks = ",".join("?" * len(docs))
    return Q.con.execute(
        f"SELECT c.doc_id, c.heading_path, c.page_start, c.text FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.chunk_id "
        f"WHERE chunks_fts MATCH ? AND c.doc_id IN ({marks}) ORDER BY c.doc_id, c.seq LIMIT ?",
        [fts_query] + docs + [limit]).fetchall()

def words_to_fts(text):
    return " OR ".join(f'"{w}"' for w in (w.strip(".,") for w in re.findall(r"[a-z0-9$.,]+", text.lower())) if w)

def section_of(heading_path):
    return heading_path.split(" > ")[-1][:80]

def quote_block(text, limit=420):
    text = (text or "").strip()
    st.markdown(f"<div class='quote'>{text[:limit]}{'…' if len(text) > limit else ''}</div>", unsafe_allow_html=True)

# ============================================================
# Navigation
# ============================================================
NAV = {"Work": ["Start here", "Ask", "Agreements", "Review"], "About this demo": ["Quality", "Demo vs production"]}
PAGES = [p for group in NAV.values() for p in group]

def go_to(page):
    """Button callback: runs before the page redraws, so it may change the sidebar's selection."""
    st.session_state.page = page

# ============================================================
# Start here
# ============================================================
def page_start():
    n_docs = Q.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_agreements = len(Q.family_roots())
    n_chunks = Q.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    waiting = len(review_queue())
    st.title("Agreement desk")
    st.markdown(f"<p class='answer'>Ask questions across {n_agreements} agreements ({n_docs} contracts and amendments "
                f"from public SEC filings). Counts come from data an AI extracted once and a person can correct. "
                f"Examples come from searching the text. Every answer opens to the page it came from.</p>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Ask a question")
        st.markdown("<p class='muted'>\"Which agreements have no liability cap?\" or \"Show me termination for "
                    "convenience clauses.\"</p>", unsafe_allow_html=True)
        st.button("Ask", type="primary", use_container_width=True, on_click=go_to, args=("Ask",))
    with c2:
        st.subheader("Open an agreement")
        st.markdown("<p class='muted'>The master, its amendments, the terms in force today, and the source of each.</p>",
                    unsafe_allow_html=True)
        st.button("Browse agreements", use_container_width=True, on_click=go_to, args=("Agreements",))
    with c3:
        st.subheader("Review the AI")
        st.markdown(f"<p class='muted'>{waiting} answers are waiting for a person to confirm or correct.</p>",
                    unsafe_allow_html=True)
        st.button("Open review", use_container_width=True, on_click=go_to, args=("Review",))
    st.divider()
    st.subheader("How it works")
    steps = [("Read", "Docling turns each PDF into text with headings and page numbers."),
             ("Link", "Rules connect each amendment to the agreement it changes."),
             ("Extract", "A local AI fills five fields per contract, each with a quote and page, checked by code."),
             ("Index", f"{n_chunks:,} clause-sized passages are searchable by words and by meaning."),
             ("Answer", "\"Find all\" questions are counted from the extracted data; \"find examples\" are searched.")]
    for col, (title, text) in zip(st.columns(5), steps):
        col.markdown(f"**{title}**")
        col.markdown(f"<p class='muted'>{text}</p>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Everything runs on this Mac. No document leaves it.</p>", unsafe_allow_html=True)

# ============================================================
# Ask
# ============================================================
EXAMPLES = ["Which agreements have no liability cap?", "How many agreements are governed by New York law?",
            "Show me termination for convenience clauses", "Agreements executed between 2000 and 2004",
            "Notice periods longer than 60 days", "Which agreements have strong indemnification clauses?"]

def page_ask():
    if "q" not in st.session_state:
        st.session_state.q = ""
    if st.session_state.get("q-pending"):
        st.session_state.q = st.session_state.pop("q-pending")
    with st.container(border=True):
        question = st.text_input("Ask about the agreements", key="q", placeholder="e.g. Which agreements can be "
                                 "terminated on 60 days notice?")
        c1, c2, c3 = st.columns([3.2, 1.1, 2.7], vertical_alignment="center")
        # The mode is visible, not hidden in a menu: "complete list" vs "examples" is the most important choice here.
        mode = c1.segmented_control("Answer as", ["Let the desk decide", "Complete list", "Examples"],
                                    default="Let the desk decide", key="mode") or "Let the desk decide"
        with c2.popover("Count by", use_container_width=True):
            level = st.radio("Count", ["Agreements", "Documents"], label_visibility="collapsed",
                             help="Agreements merge each master with its amendments.")
        pick = c3.selectbox("Try one", [""] + EXAMPLES, key="ask-pick", format_func=lambda x: x or "Try an example…",
                            label_visibility="collapsed")
        if pick and pick != st.session_state.get("last_pick"):
            st.session_state.last_pick = pick
            st.session_state["q-pending"] = pick
            st.rerun()
    mode = {"Let the desk decide": "Decide for me"}.get(mode, mode)
    if not question:
        st.markdown("<p class='muted'>Questions about every agreement (\"which\", \"how many\", \"list\") get a "
                    "complete answer from the extracted data. Questions about wording get the best-matching clauses.</p>",
                    unsafe_allow_html=True)
        return
    desk, reason = Q.route(question, {"Decide for me": "auto", "Complete list": "all", "Examples": "examples"}[mode])
    if "isn't an extracted field" in reason:
        topic = re.search(r"'([^']+)' isn't an extracted field", reason).group(1)
        st.warning(f"A complete list isn't available: {topic} isn't one of the fields this system extracts. "
                   f"Here are the best-matching clauses instead.")
    details = {"router": reason}
    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True):
        if desk in ("sql", "both"):
            answer_complete(question, level, details)
        if desk in ("search", "both"):
            answer_examples(question, details)
        how = st.expander("How this was answered")
    with right.container(border=True):
        sel = st.session_state.get("ask-source")
        if sel and sel.get("q") == question:
            source_panel(sel["doc"], sel["page"], sel["snippet"], key="ask-src")
        else:
            st.markdown("<p class='muted'>Source</p><p>Select a result to see the page it came from, "
                        "with the cited text boxed in red.</p>", unsafe_allow_html=True)
    with how:
        st.write(f"**Routing:** {details['router']}")
        if "plan" in details:
            st.write("**The AI's plan** (it only fills these blanks; code builds the query):")
            st.json(details["plan"])
        if "sql" in details:
            st.code(details["sql"], language="sql")
        if "ranks" in details:
            st.write("**Search ranks** (keyword, meaning) merged with Reciprocal Rank Fusion:")
            st.dataframe(pd.DataFrame(details["ranks"]), hide_index=True)

def pick_source(question, doc, page, snippet):
    st.session_state["ask-source"] = {"q": question, "doc": doc, "page": page, "snippet": snippet}

def answer_complete(question, level, details):
    with st.spinner("Working out the query..."):
        p = plan_question(question)
    p, repaired = Q.repair_plan(p)
    details["plan"] = {k: v for k, v in p.items() if v is not None}
    if repaired:
        details["router"] += f" | Plan repaired by code: {repaired}."
    try:
        Q.validate_plan(p)
    except ValueError as e:
        if p.get("party") and not Q.agreements_with(p["party"]):
            show_mentions(question, p["party"])        # a named company we have no agreement with: say where it appears
            return
        st.error(f"This question doesn't fit the available queries ({e}). Try rephrasing, or choose Examples in Options.")
        return
    if p["tool"] == "count_documents":
        documents_answer(question)
        return
    field = p["field"]
    label = FIELD_NAMES.get(field, field)
    mismatch = Q.plan_mismatch(question, p)     # a plan about a term the question never mentioned: say so
    if mismatch:
        st.warning(mismatch)
    years = re.findall(r"\b(?:19|20)\d\d\b", question)
    if years and not (p.get("start") or p.get("end")) and p["tool"] != "date_between":
        st.warning(f"Your question mentions {', '.join(years)}, but this answer is not filtered by date. "
                   "Try rephrasing, e.g. \"agreements effective after 2000\".")
    if st.session_state.get("drop_scope") == question:
        p = dict(p, start=None, end=None)                       # user removed the date filter
    st.markdown("<span class='badge badge-complete'>Complete list</span><span class='muted'>Every agreement in the set "
                "was checked, not a sample</span>", unsafe_allow_html=True)
    chips = [FIELD_NAMES.get(field, field)]
    if p.get("value"):
        chips.append(f"= {p['value']}")
    if p.get("op") and p.get("number") is not None:
        chips.append(f"{p['op']} {p['number']:g}")
    if p.get("start") or p.get("end"):
        chips.append(f"effective {p.get('start') or 'any date'} to {p.get('end') or 'today'}")
    chips.append("per agreement" if level == "Agreements" else "per document")
    st.markdown("<span class='muted'>Filters&nbsp;&nbsp;</span>" + "".join(f"<span class='chip'>{c}</span>" for c in chips),
                unsafe_allow_html=True)
    scoped = p["tool"] != "date_between" and (p.get("start") or p.get("end"))
    if scoped and st.button("Remove the date filter", key="drop-scope"):
        st.session_state.drop_scope = question
        st.rerun()
    if scoped:
        unknown = Q.scope_unknown()
        rng = f"{p.get('start') or 'any date'} to {p.get('end') or 'today'}"
        st.info(f"Limited to agreements effective {rng}. {len(unknown)} agreements have no known effective date "
                f"and are left out: {', '.join(unknown)}.")
    if p["tool"] == "terms_in_force":
        # One agreement: answer the term asked about, with its source, instead of sending people elsewhere.
        family, terms = Q.terms_in_force(p["doc_id"].upper())
        root = family[0][0] if family else p["doc_id"].upper()
        root = next((d for d, t, _ in family if t == "master"), root)
        if field in terms and field != "parties":
            v, d, pg, q = terms[field]
            st.markdown(f"<p class='headline'>{label} for <b>{name(root)}</b>: <b>{nice_value(field, v)}</b> "
                        f"<span class='muted'>({market_note(field, v) or 'in force'})</span></p>", unsafe_allow_html=True)
            rows = [(field, terms[field])]
        else:
            if field not in terms and field != "parties":
                st.markdown(f"<p class='headline'>No {label.lower()} was extracted for <b>{name(root)}</b>. "
                            f"Its other terms in force:</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='headline'>Terms in force for <b>{name(root)}</b>:</p>", unsafe_allow_html=True)
            rows = list(terms.items())
        selectable_table(question, [{"term": FIELD_NAMES.get(f, f), "value": nice_value(f, v), "from": d, "page": pg,
                                     "quote": q or "", "_doc": d, "_pg": pg} for f, (v, d, pg, q) in rows], key="ask-terms")
        return
    if level == "Documents" and p["tool"] != "count_by":
        sql, params, rows = Q.run_plan(p)
        details["sql"] = sql + f"\n-- params: {params}"
        table = [{"agreement": name(d), "id": d, label: nice_value(field, v), "page": pg, "quote": q or "",
                  "_doc": d, "_pg": pg} for d, v, pg, q, r in rows]
        if not rows and field == "parties":
            show_mentions(question, p["value"])
        else:
            st.markdown(f"<p class='headline'>{len(rows)} documents match</p><p class='muted'>Amendments counted on their own.</p>",
                        unsafe_allow_html=True)
            selectable_table(question, table, key="ask-docs")
    else:
        rows = Q.family_answer(p["tool"], field, p["value"], p["op"], p["number"], p.get("start"), p.get("end"))
        complete_family(question, p, rows, label)
    st.caption(accuracy_line(field))

def documents_answer(question):
    """'How many amendments are there?': structure from the documents table, no extraction involved."""
    kinds = {"master": "Master agreements", "amendment": "Amendments", "ancillary": "Related documents"}
    docs = Q.document_counts()
    amendments = [(d, r) for d, t, r in docs if t == "amendment"]
    orphans = sum(1 for d, r in amendments if d == r)
    st.markdown("<span class='badge badge-complete'>Complete list</span><span class='muted'>Every document in the set "
                "was checked</span>", unsafe_allow_html=True)
    st.markdown(f"<p class='headline'>{len(amendments)} amendments, across {len({r for _, r in amendments})} "
                f"agreements</p>", unsafe_allow_html=True)
    if orphans:
        st.markdown(f"<p class='muted'>{orphans} of them are amendments whose master agreement isn't in this set.</p>",
                    unsafe_allow_html=True)
    order = ["amendment", "master", "ancillary"]
    groups = {}
    for d, t, r in docs:
        groups.setdefault(t, []).append(f"{name(d)} ({d})")
    st.dataframe([{"kind": kinds.get(t, t), "documents": len(v), "which": "; ".join(v)}
                  for t, v in sorted(groups.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 9)],
                 hide_index=True, use_container_width=True)

def complete_family(question, p, rows, label):
    field = p["field"]
    if p["tool"] == "count_by":
        if p.get("value"):
            want = Q.norm_value(p["value"])
            match = [(r, v, h) for r, v, h, n in rows if v is not None and want in str(v).lower()]
            st.markdown(f"<p class='headline'><b>{len(match)} of {len(rows)} agreements</b> have {label.lower()} "
                        f"{p['value']}.</p>", unsafe_allow_html=True)
            selectable_table(question, [{"agreement": name(r), "id": r, label: nice_value(field, v),
                                         "from": h[1], "page": h[2], "quote": h[3], "_doc": h[1], "_pg": h[2]}
                                        for r, v, h in match], key="ask-count")
            return
        missing = f"No {label.lower()} found"
        counts = {}
        for r, v, h, n in rows:
            counts.setdefault(nice_value(field, v) or missing, []).append(r)
        ordered = sorted(counts.items(), key=lambda kv: (kv[0] == missing, -len(kv[1])))
        st.markdown(f"<p class='headline'>{label} across <b>{len(rows)} agreements</b>:</p>", unsafe_allow_html=True)
        st.dataframe([{label: v, "agreements": len(r), "which": ", ".join(r)} for v, r in ordered],
                     hide_index=True, use_container_width=True)
        return
    absence = p["tool"] == "list_where" and rows and all(h is None for _, _, h, _ in rows) and field in Q.ABSENCE_CHECK
    if absence:
        table = []
        for r, v, h, note in rows:
            if Q.confirmed_absent(r, field):
                status, order = "Confirmed none by a reviewer", 3
            else:
                n, heading, page = Q.absence_evidence(r, field)
                if n:
                    status, order = f"Check: {n} passages mention limiting liability", 0
                elif "master not in sample" in note:
                    status, order = "Unknown: the master agreement isn't here", 2
                else:
                    status, order = "Likely none: no limitation language found", 1
            ev = Q.absence_evidence(r, field)
            table.append({"agreement": name(r), "id": r, "status": status, "_o": order, "_doc": r,
                          "_pg": ev[2] if ev and ev[0] else 1, "quote": ""})
        table.sort(key=lambda t: t["_o"])
        to_check = sum(t["_o"] == 0 for t in table)
        st.markdown(f"<p class='headline'><b>{len(rows)} agreements</b> have no {label.lower()} extracted. "
                    f"<b>{to_check}</b> of them mention limiting liability somewhere, so those may be misses "
                    f"worth checking.</p>", unsafe_allow_html=True)
        selectable_table(question, table, key="ask-absent")
        return
    if not rows and field == "parties":
        show_mentions(question, p["value"])
        return
    st.markdown(f"<p class='headline'>{len(rows)} agreements match</p>", unsafe_allow_html=True)
    if not rows:
        st.warning("No agreements match. That can mean none exist, or that the question didn't map to the data "
                   "the way you meant. Check \"How this was answered\" below.")
    selectable_table(question, [{"agreement": name(r), "id": r, label: nice_value(field, v),
                                 "from": h[1] if h else "", "page": h[2] if h else "", "quote": (h[3] or "") if h else "",
                                 "_doc": h[1] if h else r, "_pg": h[2] if h else 1} for r, v, h, n in rows], key="ask-list")

def show_mentions(question, value):
    """An empty party search is the most dangerous zero: it reads as "we have no contracts with them".
    So check the text itself, and say plainly which kind of "no" this is."""
    m = Q.party_mentions(value)
    if not m:
        st.markdown(f"<p class='headline'>No agreement lists <b>{value}</b> as a party, and the name doesn't "
                    f"appear anywhere in the agreements' text.</p>", unsafe_allow_html=True)
        return
    st.markdown(f"<p class='headline'>No agreement lists <b>{value}</b> as a party, but <b>{sum(r[1] for r in m)} "
                f"passages in {len(m)} documents</b> mention it. It may be a parent company, affiliate or "
                f"former name.</p>", unsafe_allow_html=True)
    selectable_table(question, [{"agreement": name(d), "id": d, "mentions": n, "page": pg, "quote": snip,
                                 "_doc": d, "_pg": pg} for d, n, h, pg, snip in m], key="ask-mentions")

def selectable_table(question, table, key):
    if not table:
        return
    df = pd.DataFrame(table)
    visible = [c for c in df.columns if not c.startswith("_")]
    event = st.dataframe(df[visible], hide_index=True, use_container_width=True, on_select="rerun",
                         selection_mode="single-row", key=key,
                         column_config={"quote": st.column_config.TextColumn("quote", width="large")})
    rows = event.selection.rows if event and event.selection else []
    if rows:
        r = table[rows[0]]
        pick_source(question, r["_doc"], r["_pg"] or 1, r.get("quote", ""))

def answer_examples(question, details):
    fused, kw, vec = Q.hybrid_search(question)
    st.markdown("<span class='badge badge-examples'>Examples</span><span class='muted'>Best-matching clauses, "
                "not a complete list</span>", unsafe_allow_html=True)
    details["ranks"] = []
    for n, cid in enumerate(fused, start=1):
        doc_id, heading, pages, text = Q.describe(cid)
        page = int(re.search(r"\d+", pages).group())
        st.markdown(f"<div class='result-title'>{name(doc_id)}</div>"
                    f"<div class='muted'>{section_of(heading)}, {pages.replace('p.', 'page ').replace('pp.', 'pages ')}</div>",
                    unsafe_allow_html=True)
        quote_block(text, 300)
        if st.button(f"Show page {page}", key=f"src-{cid}"):
            pick_source(question, doc_id, page, text)
            st.rerun()
        details["ranks"].append({"result": n, "agreement": doc_id, "keyword rank": kw.index(cid) + 1 if cid in kw else None,
                                 "meaning rank": vec.index(cid) + 1 if cid in vec else None})

def accuracy_line(field):
    """Uncertainty is only useful if it says what to do. Lowest measured recall decides the advice."""
    rows = Q.con.execute("SELECT eval_set, precision, recall, n FROM eval_results WHERE field = ?", (field,)).fetchall()
    if not rows:
        return "Not yet measured: spot-check against the source before relying on this."
    worst = min(rows, key=lambda r: r[2])
    misses = round((1 - worst[2]) * 10)
    measured = "; ".join(f"{p:.0%} precision, {r:.0%} recall on {s}" for s, p, r, n in rows)
    if worst[2] >= 0.9:
        advice = "Reliable on the contracts tested; open the source for anything high-stakes."
    else:
        advice = f"Spot-check before relying on this: the AI missed about {misses} in 10 on {worst[0]}."
    return f"{advice} (Measured: {measured}.)"

# ============================================================
# Agreements
# ============================================================
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
        import datetime
        return datetime.date.fromisoformat(iso).strftime("%b %-d, %Y")
    except (TypeError, ValueError):
        return "date unknown"

def term_history(family, field, root):
    """One term, document by document, keyed to the strip's numbers: '1 Set p.6 · 2 Restated p.1'."""
    src, expr = Q.FIELD_SQL[field]
    parts, prev = [], None
    for n, (doc, doc_type, eff) in enumerate(family, 1):
        if doc_type == "ancillary" or (field in ("effective_date", "parties") and doc != root):
            continue
        row = Q.con.execute(f"SELECT {expr}, e.page FROM current_extractions e WHERE e.doc_id = ? AND e.field = ? "
                            f"AND e.found = 1", (doc, src)).fetchone()
        if not row or row[0] in (None, "none"):
            continue
        v = nice_value(field, row[0])
        parts.append(f"{n} Set (p.{row[1]})" if prev is None else
                     f"{n} Restated, no change (p.{row[1]})" if v == prev else f"{n} Changed from {prev} (p.{row[1]})")
        prev = v
    return " · ".join(parts)

def page_agreements():
    roots = Q.family_roots()
    root = st.selectbox("Agreement", roots, format_func=lambda d: f"{name(d)} ({d})", label_visibility="collapsed")
    family, terms = Q.family_terms(root)
    master = next((d for d, t, _ in family if t == "master"), root)
    st.markdown(f"<h1 style='margin-bottom:0'>{name(master).split(',')[0]}</h1>"
                f"<p class='muted' style='font-size:1rem'>{', '.join(name(master).split(', ')[1:])}</p>",
                unsafe_allow_html=True)
    if not any(t == "master" for _, t, _ in family):
        st.markdown("<div class='flag'>The master agreement isn't in this set, so terms come from this amendment alone.</div>",
                    unsafe_allow_html=True)
    # The documents strip: what this agreement consists of, numbered, in the order later ones override earlier ones.
    strip = []
    for n, (d, t, eff) in enumerate(family, 1):
        strip.append(f"<div class='doc'><span class='num'>{n}</span><span><b>{doc_label(d, t)}</b> "
                     f"<span class='muted'>· {nice_date(eff)} · {d}</span></span></div>")
    st.markdown("<div class='docstrip'>" + "<span class='arrow'>›</span>".join(strip) + "</div>"
                "<p class='muted'>Ordered by date, then amendment number, then filing order. Later documents override "
                "earlier ones.</p>", unsafe_allow_html=True)
    tab_terms, tab_docs, tab_read = st.tabs(["Terms in force", "Why these are linked", "Read and search"])
    with tab_terms:
        left, right = st.columns([3, 2], gap="medium")
        with left.container(border=True):
            if not terms:
                st.write("Nothing was extracted for this agreement.")
            else:
                st.markdown("<h3 style='margin-top:0'>Terms in force</h3><p class='muted'>Select a term to see its "
                            "source.</p>", unsafe_allow_html=True)
                table = [{"term": FIELD_NAMES.get(f, f), "in force today": nice_value(f, v),
                          "history, by document": term_history(family, f, root),
                          "vs. other agreements": market_note(f, v),
                          "page": pg, "quote": q or "", "_doc": d, "_pg": pg} for f, (v, d, pg, q) in terms.items()]
                df = pd.DataFrame(table)
                event = st.dataframe(df[["term", "in force today", "history, by document", "vs. other agreements"]],
                                     hide_index=True, use_container_width=True, on_select="rerun",
                                     selection_mode="single-row", key="ag-terms")
                chosen = table[event.selection.rows[0]] if event.selection.rows else table[0]
        with right.container(border=True):
            if terms:
                st.markdown(f"<p class='muted'>Source for {chosen['term']}</p>", unsafe_allow_html=True)
                source_panel(chosen["_doc"], chosen["_pg"], chosen["quote"], key=f"ag-src-{root}",
                             docs=family_docs(root))
    with tab_docs:
        reasons = dict(Q.con.execute("SELECT doc_id, family_reason FROM documents").fetchall())
        st.dataframe([{"document": name(d), "id": d, "type": t, "effective": e or "",
                       "linked because": reasons.get(d) or ""} for d, t, e in family],
                     hide_index=True, use_container_width=True)
    with tab_read:
        reader(root, key=f"ag-read-{root}")

# Benchmarks used to be their own page. Nobody opens a benchmarks page out of curiosity: the question
# "is this normal?" comes up while looking at a term, so the answer lives next to the term.
@st.cache_data(show_spinner=False)
def market_values(field):
    return [v for r, v, h, n in Q.family_answer("count_by", field) if v is not None]

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

def reader(root, key):
    docs = family_docs(root)
    term = st.text_input("Search every document in this agreement", key=f"{key}-q", placeholder="e.g. liable, exceed, terminate")
    left, right = st.columns([2, 3], gap="large")
    hit = None
    with left:
        if term:
            found = search_within(docs, words_to_fts(term))
            st.markdown(f"<p class='muted'>{len(found)} passages{' (first 50)' if len(found) == 50 else ''}</p>",
                        unsafe_allow_html=True)
            for i, (d, heading, page, text) in enumerate(found):
                st.markdown(f"**{section_of(heading)}**  \n<span class='muted'>{d}, page {page}</span>",
                            unsafe_allow_html=True)
                quote_block(text, 220)
                if st.button("Show page", key=f"{key}-h{i}"):
                    st.session_state[f"{key}-sel"] = (d, page, text)
        hit = st.session_state.get(f"{key}-sel")
    with right:
        d, page, text = hit if hit else (docs[0], 1, "")
        source_panel(d, page, text, key=f"{key}-src", docs=docs)

# ============================================================
# Review
# ============================================================
QUESTION_FOR = {"parties": "Who are the parties?", "effective_date": "When does it take effect?",
                "governing_law": "Which law governs it?", "termination_notice": "How much notice to terminate?",
                "liability_cap": "Is liability capped, and how?"}
FIELD_HINT = {"parties": "Legal names separated by ;", "effective_date": "YYYY-MM-DD",
              "governing_law": "e.g. New York", "termination_notice": "Number of days, e.g. 90"}

def review_queue():
    items = []
    for doc, field, value, quote, page, conf in Q.con.execute(
            "SELECT doc_id, field, value_norm, quote, page, confidence FROM current_extractions "
            "WHERE needs_review = 1 ORDER BY doc_id"):
        items.append({"doc": doc, "field": field, "why": "The AI's answer failed one of the automatic checks.",
                      "value": value, "quote": quote, "page": page})
    for root, v, hit, note in Q.family_answer("list_where", "cap_type", "none"):
        if Q.confirmed_absent(root, "cap_type"):
            continue
        n, heading, page = Q.absence_evidence(root, "cap_type")
        reviewed = Q.con.execute("SELECT 1 FROM reviews WHERE doc_id = ? AND field = 'liability_cap'", (root,)).fetchone()
        if n and not reviewed:
            chunk = Q.con.execute("SELECT text FROM chunks WHERE doc_id IN (%s) AND heading_path LIKE ? ORDER BY seq LIMIT 1"
                                  % ",".join("?" * len(family_docs(root))), family_docs(root) + [f"%{heading}%"]).fetchone()
            items.append({"doc": root, "field": "liability_cap",
                          "why": f"The AI found no cap, but {n} passages mention limiting liability.",
                          "value": None, "quote": chunk[0] if chunk else "", "page": page})
    return items

def save_review(doc, field, status, value, quote, page, note, reviewer):
    Q.con.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
                  (doc, field, status, value, quote, page, note, reviewer))
    Q.con.commit()

def page_review():
    st.title("Review queue")
    items = review_queue()
    done = Q.con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    st.markdown(f"<p class='answer'><b>{len(items)}</b> answers waiting, <b>{done}</b> reviewed. Your decisions are "
                f"saved separately from the AI's answers, with your name and the time, and every answer in the app "
                f"uses them right away.</p>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Reviewing as Ken. In a real system this comes from your login, so the audit "
                "trail can't be edited.</p>", unsafe_allow_html=True)
    st.session_state.reviewer = "Ken"
    tab_queue, tab_log = st.tabs(["Queue", "Decisions so far"])
    with tab_queue:
        if not items:
            st.success("Nothing waiting. Every flagged answer has been reviewed.")
        else:
            labels = [f"{it['doc']}: {QUESTION_FOR.get(it['field'], it['field'])}" for it in items]
            i = st.selectbox("Item", range(len(items)), format_func=lambda k: f"{k + 1} of {len(items)}. {labels[k]}",
                             label_visibility="collapsed")
            review_item(items[i], i)
    with tab_log:
        log = Q.con.execute("SELECT reviewed_at, reviewer, doc_id, field, status, value, note FROM reviews "
                            "ORDER BY reviewed_at DESC").fetchall()
        if not log:
            st.write("No decisions yet.")
        else:
            st.dataframe([{"when": w, "reviewer": who, "agreement": name(d), "id": d,
                           "question": QUESTION_FOR.get(f, f), "decision": s, "value": nice_value(f, v), "note": n}
                          for w, who, d, f, s, v, n in log], hide_index=True, use_container_width=True)
            if st.button("Undo the most recent decision"):
                Q.con.execute("DELETE FROM reviews WHERE doc_id = ? AND field = ?", (log[0][2], log[0][3]))
                Q.con.commit()
                st.rerun()

# A review item is phrased as ONE question the reviewer can answer in one look, and the buttons are its answers.
CHECK_QUESTION = {
    "termination_notice": "Does the clause on page {p} give {v} notice to terminate?",
    "governing_law": "Does the clause on page {p} make {v} law govern the agreement?",
    "effective_date": "Does page {p} show the agreement taking effect on {v}?",
    "parties": "Are the parties {v}, as page {p} says?",
    "liability_cap": "Does the clause on page {p} cap liability as: {v}?",
}

def task_question(it):
    v = nice_value(it["field"], it["value"])
    if not v:
        return "The AI found no liability cap. Do any of these passages limit liability?"
    return CHECK_QUESTION.get(it["field"], "Is {v} correct, per page {p}?").format(v=v, p=it["page"] or "?")

def review_item(it, i):
    root = root_of(it["doc"])
    left, right = st.columns([2, 3], gap="medium")
    with left.container(border=True):
        st.markdown(f"<p class='muted'>{name(it['doc'])} · {FIELD_NAMES.get(it['field'], it['field'])}</p>"
                    f"<div class='task-label'>Your task</div><div class='task'>{task_question(it)}</div>",
                    unsafe_allow_html=True)
        if it["quote"]:
            quote_block(it["quote"], 500)
            if it["page"] and st.button(f"Show this clause on page {it['page']}", key=f"jump-{i}"):
                st.session_state[f"rv-sel-{i}"] = (it["doc"], it["page"], it["quote"])
        st.markdown(f"<div class='flag'><b>Why it's here:</b> {it['why']}</div>", unsafe_allow_html=True)
        if it["field"] == "liability_cap" and it["value"] is None:
            evidence = search_within(family_docs(root), Q.ABSENCE_CHECK["cap_type"])
            with st.expander(f"All {len(evidence)} passages about limiting liability"):
                for k, (d, heading, page, text) in enumerate(evidence):
                    st.markdown(f"**{section_of(heading)}**  \n<span class='muted'>{d}, page {page}</span>",
                                unsafe_allow_html=True)
                    quote_block(text, 300)
                    if st.button("Show page", key=f"ev-{i}-{k}"):
                        st.session_state[f"rv-sel-{i}"] = (d, page, text)
        term = st.text_input("Search the whole agreement", key=f"rv-q-{i}", placeholder="e.g. liable, exceed")
        if term:
            found = search_within(family_docs(root), words_to_fts(term))
            st.markdown(f"<p class='muted'>{len(found)} passages</p>", unsafe_allow_html=True)
            for k, (d, heading, page, text) in enumerate(found[:20]):
                if st.button(f"{d}, page {page}: {section_of(heading)[:50]}", key=f"rvs-{i}-{k}"):
                    st.session_state[f"rv-sel-{i}"] = (d, page, text)
        st.markdown("<h3>Your answer</h3>", unsafe_allow_html=True)
        reviewer = st.session_state.get("reviewer", "Ken")
        # Intentional friction against rubber-stamping: answers unlock only after confirming you read the clause.
        checked = st.checkbox(f"I read the source{' on page ' + str(it['page']) if it['page'] else ''}", key=f"chk-{i}")
        v = nice_value(it["field"], it["value"])
        if v and st.button(f"Yes, {v} is correct", key=f"ok-{i}", type="primary", disabled=not checked,
                           use_container_width=True):
            save_review(it["doc"], it["field"], "accepted", None, None, None, st.session_state.get(f"nt-{i}", ""), reviewer)
            st.rerun()
        with st.expander("No, it says something else" if v else "Yes, there is a cap: record it"):
            if it["field"] == "liability_cap":
                cap_type = st.selectbox("Cap", list(CAP_NAMES), format_func=CAP_NAMES.get, key=f"ct-{i}")
                amount = st.number_input("Dollar amount (fixed amount only)", min_value=0.0, step=1000.0, key=f"ca-{i}")
                corrected = json.dumps({"cap_type": cap_type, "cap_amount": amount or None})
            else:
                corrected = st.text_input(f"What it actually says ({FIELD_HINT.get(it['field'], '')})",
                                          value=it["value"] or "", key=f"cv-{i}")
            if st.button("Save correction", key=f"fix-{i}", disabled=not checked, use_container_width=True):
                save_review(it["doc"], it["field"], "corrected", corrected, (it["quote"] or "")[:300] or None,
                            it["page"], st.session_state.get(f"nt-{i}", ""), reviewer); st.rerun()
        absent = ("No, this agreement has no such clause" if v else "No, there's no cap in this agreement")
        if st.button(absent, key=f"no-{i}", disabled=not checked, use_container_width=True):
            save_review(it["doc"], it["field"], "absent", None, None, None, st.session_state.get(f"nt-{i}", ""), reviewer)
            st.rerun()
        st.text_input("Note (optional)", key=f"nt-{i}")
        st.markdown("<p class='muted'>Saved alongside the AI's answer, never over it, with your name and the time.</p>",
                    unsafe_allow_html=True)
    with right.container(border=True):
        d, page, text = st.session_state.get(f"rv-sel-{i}", (it["doc"], it["page"] or 1, it["quote"]))
        st.markdown("<p class='muted' style='color:#C8102E;font-weight:700'>Clause at issue, boxed in red</p>",
                    unsafe_allow_html=True)
        source_panel(d, page, text, key=f"rv-src-{i}", docs=family_docs(root))

# ============================================================
# Quality
# ============================================================
SERIES = {"precision": "#2a78d6", "recall": "#eb6834"}   # validated palette, categorical slots 1 and 2

def page_quality():
    import altair as alt
    st.title("Quality")
    st.markdown("<p class='answer'>How far to trust it, measured. The AI's own answers are scored; reviewers' "
                "corrections don't count toward its score.</p>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Extraction", "Search", "Routing", "Human review"])
    with t1:
        rows = Q.con.execute("SELECT eval_set, field, precision, recall, n FROM eval_results").fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["set", "field", "precision", "recall", "n"])
            df["field"] = df["field"].map(lambda f: FIELD_NAMES.get(f, f))
            chosen = st.radio("Answer key", sorted(df["set"].unique()), horizontal=True)
            d = df[df["set"] == chosen].melt(id_vars=["field"], value_vars=["precision", "recall"],
                                             var_name="measure", value_name="score")
            chart = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=16).encode(
                x=alt.X("field:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)), xOffset="measure:N",
                y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", gridOpacity=0.3)),
                color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values()))),
                tooltip=["field", "measure", alt.Tooltip("score:Q", format=".0%")]).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
            st.markdown("<p class='muted'>Precision: of the answers given, how many were right. Recall: of the answers "
                        "that exist, how many were found.</p>", unsafe_allow_html=True)
        st.subheader("What each round of fixes bought")
        st.dataframe([
            {"measure": "Answers flagged for review (of 200)", "v1": "48", "v2": "7", "v3": "8"},
            {"measure": "My 10 contracts, all fields", "v1": "", "v2": "92% / 88%", "v3": "92% / 92%"},
            {"measure": "30 unseen contracts: parties", "v1": "", "v2": "85% / 76%", "v3": "93% / 93%"},
            {"measure": "30 unseen contracts: effective date", "v1": "", "v2": "81% / 68%", "v3": "86% / 76%"},
            {"measure": "30 unseen contracts: governing law", "v1": "", "v2": "88% / 95%", "v3": "92% / 100%"},
        ], hide_index=True, use_container_width=True)
        st.caption("Precision / recall. Recorded from the runs on Oct 16.")
    with t2:
        st.markdown("<p class='muted'>40 plain-English questions. Each top-5 result was graded by an LLM judge "
                    "(Gemma 4 12B, a different model family from the one that extracts), calibrated first against "
                    "hand labels: right on 21 of 23 cases where it disagreed with keyword rules, and 15 of 15 on a "
                    "blind spot-check. A hit means at least one relevant clause in the top 5.</p>", unsafe_allow_html=True)
        st.dataframe([
            {"method": "Words only (BM25)", "found something useful": "72%", "top 5 that were right": "28%"},
            {"method": "Meaning only (embeddings)", "found something useful": "78%", "top 5 that were right": "60%"},
            {"method": "Both, merged", "found something useful": "82%", "top 5 that were right": "51%"},
            {"method": "Merged, then reranked (MiniLM)", "found something useful": "85%", "top 5 that were right": "64%"},
            {"method": "Merged, then reranked (BGE): in use", "found something useful": "90%", "top 5 that were right": "70%"},
        ], hide_index=True, use_container_width=True)
        st.caption("Of the 4 questions with nothing found: 2 have no such clause in these contracts (correct), and 2 "
                   "are real misses on non-solicitation clauses (\"poaching staff\" vs \"solicit personnel\").")
    with t3:
        st.markdown("<p class='muted'>39 held-out questions. The costly miss is a \"find all\" question sent to "
                    "search, which returns a partial list that looks complete.</p>", unsafe_allow_html=True)
        st.dataframe([
            {"router": "Hand-written rules", "correct": "92%", "costly misses": 3},
            {"router": "Trained model", "correct": "95%", "costly misses": 1},
        ], hide_index=True, use_container_width=True)
        st.caption("In use: your choice first, then rules, then the trained model if it's at least 65% sure, "
                   "otherwise both answers.")
    with t4:
        stats = dict(Q.con.execute("SELECT status, COUNT(*) FROM reviews GROUP BY status").fetchall())
        total = sum(stats.values())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reviewed", total)
        c2.metric("Accepted", stats.get("accepted", 0))
        c3.metric("Corrected", stats.get("corrected", 0))
        c4.metric("Not in agreement", stats.get("absent", 0))
        if total:
            st.markdown(f"<p class='muted'>Reviewers changed {stats.get('corrected', 0) / total:.0%} of the answers "
                        f"they looked at. Tracked over time, that's the clearest measure of how much to trust the AI.</p>",
                        unsafe_allow_html=True)

# ============================================================
# Demo vs production
# ============================================================
def page_production():
    st.title("Demo vs production")
    st.markdown("<p class='answer'>What this demo does at each layer, and what a real agreement platform would add.</p>",
                unsafe_allow_html=True)
    st.markdown("<p class='muted'>Built with: Docling (parsing), qwen2.5:14b (extraction) and nomic-embed-text "
                "(embeddings) running locally through Ollama, SQLite with FTS5 (storage and keyword search), "
                "scikit-learn (router), Streamlit (this app). Contracts: CUAD v1, The Atticus Project, CC BY 4.0.</p>",
                unsafe_allow_html=True)
    st.dataframe([
        {"layer": "Ingestion", "this demo": "40 PDFs from a fixed sample, fingerprinted; re-runs skip finished work",
         "production adds": "Connectors to the document system and deal rooms; incremental updates; source permissions"},
        {"layer": "Parsing", "this demo": "Docling, with per-document quality signals",
         "production adds": "Fallback parsers and OCR routing for bad parses; approved model storage"},
        {"layer": "Agreement families", "this demo": "Rules on parties, type and date, with reasons",
         "production adds": "Corporate name-change lookups, side letters, partial amendments"},
        {"layer": "Storage", "this demo": "One SQLite file; AI answers and human reviews kept apart",
         "production adds": "Versioned storage with legal hold; history of what the system believed on any date"},
        {"layer": "Search", "this demo": "Word and meaning indexes in one file",
         "production adds": "A search engine at scale, a reranker, filters, and a plan for re-embedding"},
        {"layer": "Extraction", "this demo": "Five fields from a local 14B model, with quote checks and review",
         "production adds": "An attorney-defined schema (carve-outs, indemnification, MFN), stronger models, versioned prompts"},
        {"layer": "Answering", "this demo": "Router, fixed query templates, coverage check",
         "production adds": "Permission-aware retrieval, query rewriting, usage analytics"},
        {"layer": "Evaluation", "this demo": "10 hand labels, a 30-contract holdout, search and routing tests",
         "production adds": "Gold sets from two attorneys, regression tests on every change, live monitoring"},
        {"layer": "Governance", "this demo": "Runs offline; review audit trail",
         "production adds": "Entitlements inside the index, query audit logs, lineage, vendor confidentiality terms"},
    ], hide_index=True, use_container_width=True)

# ============================================================
# Layout
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Start here"

@st.cache_data(ttl=15, show_spinner=False)
def waiting_count():
    return len(review_queue())

# Top bar, as on the design canvas: the three work screens up front, demo plumbing tucked under "About this demo".
bar = st.columns([2.2, 0.8, 1.1, 1.0, 4.2, 1.7], vertical_alignment="center")
bar[0].button("Agreement Desk", key="nav-home", type="tertiary", on_click=go_to, args=("Start here",))
for col, pg, label in [(bar[1], "Ask", "Ask"), (bar[2], "Agreements", "Agreements"),
                       (bar[3], "Review", f"Review ({waiting_count()})")]:
    col.button(label, key=f"nav-{pg}", on_click=go_to, args=(pg,), use_container_width=True,
               type="primary" if st.session_state.page == pg else "tertiary")
with bar[5].popover("About this demo", use_container_width=True):
    for pg in ["Start here", "Quality", "Demo vs production"]:
        st.button(pg, key=f"nav-{pg}", on_click=go_to, args=(pg,), use_container_width=True, type="tertiary")
    st.caption("40 agreements, CUAD sample. Runs entirely on this Mac.")
st.markdown("<div class='topline'></div>", unsafe_allow_html=True)

{"Start here": page_start, "Ask": page_ask, "Agreements": page_agreements,
 "Review": page_review, "Quality": page_quality, "Demo vs production": page_production}[st.session_state.page]()
