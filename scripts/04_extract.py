"""Step 4: extract a small schema from each contract with a local LLM, store it in SQLite.

AI reads, databases count. For every contract and every field:
  1. RETRIEVE: pick the few passages most likely to hold the answer (keyword scoring).
  2. EXTRACT:  the local model (via Ollama) fills a JSON schema: value, exact quote, confidence.
  3. VERIFY:   code checks the quote really appears in the contract, and finds its page.
  4. NORMALIZE: "ninety (90) days" -> 90, "January 1, 2004" -> 2004-01-01, "$1,000,000" -> 1000000.
  5. STORE:    one row per (contract, field) in data/legal.db, with model + prompt version.

Usage:
  uv run python scripts/04_extract.py --doc D004     # one contract (try this first)
  uv run python scripts/04_extract.py                # all 40 (skips what's already done)
  uv run python scripts/04_extract.py --redo         # re-extract everything
"""
import argparse
import csv
import json
import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path

import ollama

MODEL = "qwen2.5:14b"
PROMPT_VERSION = "v3"        # bump this whenever the prompts change, so results stay traceable
PARSED = Path("data/parsed")
DB = Path("data/legal.db")

# ============================================================
# Fields: what to look for (keywords) and what to ask the model
# ============================================================
FIELDS = {
    "parties": {
        # v3: dropped "party"/"parties": long contracts use them hundreds of times, so body text
        # outscored the opening paragraph where the names actually are.
        "keywords": ["by and between", "between", "entered into", "made", "inc", "corporation", "llc", "ltd"],
        "use_opening": True,   # parties are almost always in the first paragraph
        "question": ("Who are the parties that signed this agreement? List each party's full legal name as "
                     "written in the opening paragraph. Do not use defined-term nicknames or roles "
                     "(such as 'the Company' or 'Licensor'). If the names are left blank, set found=false."),
        "value_schema": {"type": "array", "items": {"type": "string"}},
    },
    "effective_date": {
        "keywords": ["effective date", "effective as of", "dated as of", "entered into", "dated", "effective"],
        "use_opening": True,
        "question": ("What is the effective date of this agreement? If only a signing/agreement date is given, "
                     "use that. If this document is an amendment, give the amendment's own effective date, "
                     "not the date of the original agreement it amends."),
        "value_schema": {"type": "string"},
    },
    "governing_law": {
        "keywords": ["governed by", "governing law", "laws of the state", "laws of", "construed in accordance"],
        "use_opening": False,
        "question": "Which jurisdiction's law governs this agreement? Answer with the state or country only.",
        "value_schema": {"type": "string"},
    },
    "termination_notice": {
        "keywords": ["terminate", "termination", "notice", "days", "prior written notice", "months"],
        "use_opening": False,
        "question": ("How much advance notice must a party give to terminate this agreement (for convenience "
                     "or to end it, including notice of non-renewal before a renewal date)? Answer as written, e.g. 'ninety (90) days'. "
                     "Ignore cure periods for breach."),
        "value_schema": {"type": "string"},
    },
    "liability_cap": {
        "keywords": ["liability", "liable", "shall not exceed", "in no event", "limitation of liability",
                     "aggregate", "consequential"],
        "use_opening": False,
        "question": ("Does this agreement cap or limit either party's liability? Classify the cap: "
                     "'fixed_amount' (a stated dollar figure), 'fees_based' (tied to fees or amounts paid), "
                     "'exclusion_only' (only excludes types of damages, e.g. consequential, with no overall cap), "
                     "or 'none'. If there is a dollar figure, give it as written. The cap must limit liability under the "
                     "agreement as a whole; a cap on only one section's obligations is not the overall cap."),
        "value_schema": {"type": "object",
                         "properties": {"cap_type": {"type": "string",
                                                     "enum": ["fixed_amount", "fees_based", "exclusion_only", "none"]},
                                        "cap_amount": {"type": ["string", "null"]}},
                         "required": ["cap_type", "cap_amount"]},
    },
}

# ============================================================
# 1. RETRIEVE: split the contract into passages, score them by keywords
# ============================================================
def load_passages(doc_id, max_len=1500, overlap=250):
    """Turn Docling's text items into passages of at most ~max_len characters, each with its page(s).
    Giant blobs (like D020's 14,000-character item) are cut into overlapping windows so they can be
    scored. The overlap means a clause cut at a window edge still appears whole in the next window."""
    doc = json.load(open(PARSED / f"{doc_id}.json"))
    passages = []
    for item in doc["texts"]:
        text = item["text"].strip()
        if not text:
            continue
        pages = sorted({p["page_no"] for p in item["prov"]})
        step = max_len - overlap
        for start in range(0, max(1, len(text) - overlap), step):
            passages.append({"text": text[start:start + max_len], "pages": pages})
    return passages

def retrieve(passages, field, budget=5000, opening_chars=3000):
    """Pick the highest-scoring passages for this field, up to ~budget characters in total.
    v3: for fields that live in the opening paragraph (parties, dates), the first ~3000 characters
    are ALWAYS included, then the rest of the budget goes to keyword matches. (v2 only gave the first
    3 passages a small bonus, and those were often just 'Exhibit 10.1' headers.)"""
    spec = FIELDS[field]
    chosen, used = [], 0
    if spec["use_opening"]:
        for i, p in enumerate(passages):
            if used >= opening_chars:
                break
            chosen.append(i)
            used += len(p["text"])
    scored = []
    for i, p in enumerate(passages):
        if i in chosen:
            continue
        low = p["text"].lower()
        score = sum(low.count(k) for k in spec["keywords"])
        if score > 0:
            scored.append((score, i))
    for score, i in sorted(scored, reverse=True):
        if used + len(passages[i]["text"]) > budget:
            continue
        chosen.append(i)
        used += len(passages[i]["text"])
    return [passages[i] for i in sorted(chosen)]   # keep contract order, easier to read

# ============================================================
# 2. EXTRACT: ask the local model, forcing a JSON shape
# ============================================================
def build_schema(field):
    return {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "value": FIELDS[field]["value_schema"],
            "quote": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["found", "value", "quote", "confidence"],
    }

SYSTEM = ("You extract facts from legal contracts. Use ONLY the excerpts provided. "
          "If the excerpts do not contain the answer, set found=false, value to an empty value, quote to \"\". "
          "The quote must be copied EXACTLY, word for word, from the excerpts (one sentence or phrase, "
          "under 300 characters). Never guess. confidence is 0 to 1: how sure you are that your ANSWER is "
          "correct. That includes found=false: if you are sure the excerpts do not contain it, confidence is high.")

def extract(field, excerpts):
    excerpt_text = "\n\n".join(f"[Excerpt {n}, page {','.join(map(str, p['pages']))}]\n{p['text']}"
                               for n, p in enumerate(excerpts, start=1))
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Question: {FIELDS[field]['question']}\n\nExcerpts:\n{excerpt_text}"}],
        format=build_schema(field),                       # the model MUST return this JSON shape
        options={"temperature": 0, "num_ctx": 8192},      # temperature 0 = same input, same output
    )
    return json.loads(response["message"]["content"])

# ============================================================
# 3. VERIFY: is the quote really in the contract? On which page?
# ============================================================
def words(s):
    """Lowercase words only. Punctuation, quote marks and spacing are ignored, so
    'April 15 , 2010' and 'April 15, 2010' compare equal."""
    return re.findall(r"[a-z0-9]+", s.lower())

def verify_quote(quote, passages):
    """Returns (status, page). status: 'exact', 'partial' (most words match, in order), or 'not_found'."""
    q = words(quote)
    if not q:
        return "not_found", None
    q_str = " ".join(q)
    best_ratio, best_page = 0.0, None
    for p in passages:
        t = words(p["text"])
        if q_str in " ".join(t):
            return "exact", p["pages"][0]
        # How many of the quote's words appear in the passage, in the same order?
        matched = sum(b.size for b in SequenceMatcher(None, q, t, autojunk=False).get_matching_blocks())
        ratio = matched / len(q)
        if ratio > best_ratio:
            best_ratio, best_page = ratio, p["pages"][0]
    return ("partial", best_page) if best_ratio >= 0.85 else ("not_found", None)

# ============================================================
# 4. NORMALIZE: turn text answers into comparable values
# ============================================================
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                      "august", "september", "october", "november", "december"], start=1)}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45,
                "forty five": 45, "sixty": 60, "ninety": 90, "one hundred twenty": 120, "one hundred eighty": 180}

ABBREV = {"jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june",
          "jul": "july", "aug": "august", "sep": "september", "sept": "september", "oct": "october",
          "nov": "november", "dec": "december"}

def norm_date(s):
    s = s.lower()
    s = re.sub(r"\b(" + "|".join(ABBREV) + r")\b\.?", lambda m: ABBREV[m[1]], s)   # "Sept." -> "september"
    s = re.sub(r"(\d)\s+(st|nd|rd|th)\b", r"\1\2", s)                               # "1 st" -> "1st"
    m = re.search(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})", s)
    if m:
        return f"{int(m[3]):04d}-{MONTHS[m[1]]:02d}-{int(m[2]):02d}"
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(" + "|".join(MONTHS) + r"),?\s+(\d{4})", s)
    if m:
        return f"{int(m[3]):04d}-{MONTHS[m[2]]:02d}-{int(m[1]):02d}"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        y = int(m[3]) + (2000 if len(m[3]) == 2 and int(m[3]) < 50 else 1900 if len(m[3]) == 2 else 0)
        return f"{y:04d}-{int(m[1]):02d}-{int(m[2]):02d}"
    return None

def norm_notice_days(s):
    """'ninety (90) days' -> 90, '3 months' -> 90 (months counted as 30 days), 'one year' -> 365."""
    s = s.lower()
    m = re.search(r"(\d+)\s*\)?\s*(calendar\s+|business\s+)?(day|week|month|year)", s)
    if m:
        n, unit = int(m[1]), m[3]
    else:
        m = re.search(r"(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")\s+(calendar\s+)?(day|week|month|year)", s)
        if not m:
            return None
        n, unit = NUMBER_WORDS[m[1]], m[3]
    return n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]

def norm_law(s):
    s = re.sub(r"(?i)\s+and\s+the\s+federal\s+laws.*$", "", s)                       # v3
    s = re.sub(r"(?i)\b(the\s+)?(laws?\s+of\s+)?(the\s+)?(state|commonwealth|province)\s+of\s+", "", s)
    s = re.sub(r"(?i)^(the\s+)?laws?\s+of\s+(the\s+)?", "", s).strip(" .,")
    s = re.sub(r"(?i),?\s*(the\s+)?united states(\s+of\s+america)?$", "", s).strip(" .,") or s   # "Colorado, United States"
    if s.isupper():
        s = s.lower()                                    # "COLORADO" -> "colorado" -> "Colorado" below
    small = {"and", "of", "the"}
    return " ".join(w if w.lower() in small else w[:1].upper() + w[1:] for w in s.split()) if s else None

def norm_amount(s):
    m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*(million)?", s or "", re.I)
    if not m:
        return None
    n = float(m[1].replace(",", ""))
    return n * 1_000_000 if m[2] else n

# Words that start a description AFTER a party's legal name, e.g.
# "NETGEAR, Inc., a wholly owned subsidiary of..." or "Holiday RV Superstores, Inc., d/b/a Recreation USA".
PARTY_DESCRIPTORS = re.compile(r",?\s+(?:a|an)\s+(?=[A-Za-z])|,?\s+(?:d/b/a|doing business as)\b|\s+through\s+its\b|,\s*(?:organized|having|located)\b|\s+\(", re.I)

def clean_party(name):
    """Keep the legal name, drop the description: 'XYBERNAUT CORPORATION, a Virginia corporation' -> 'XYBERNAUT CORPORATION'."""
    name = re.sub(r"\s+", " ", name).strip()
    name = PARTY_DESCRIPTORS.split(name, maxsplit=1)[0].strip(" ,")
    return name if re.search(r"[A-Za-z]", name) else ""     # v3: template blanks like "_____" are not names

def normalize(field, value):
    if field == "parties":
        names = [clean_party(v) for v in (value or []) if v and v.strip()]
        return "; ".join(n for n in names if n) or None
    if field == "effective_date":
        return norm_date(value or "")
    if field == "governing_law":
        return norm_law(value or "")
    if field == "termination_notice":
        d = norm_notice_days(value or "")
        return str(d) if d is not None else None
    if field == "liability_cap":
        amount = norm_amount(value.get("cap_amount")) if value else None
        return json.dumps({"cap_type": (value or {}).get("cap_type"), "cap_amount": amount})

# ============================================================
# 5. STORE: SQLite, one row per (contract, field)
# ============================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, filename TEXT, sha256 TEXT, pages INTEGER,
    doc_type TEXT, master_id TEXT, family_reason TEXT);
CREATE TABLE IF NOT EXISTS extractions (
    doc_id TEXT, field TEXT,
    found INTEGER,            -- did the model say the contract has this?
    value_raw TEXT,           -- the model's answer, as written
    value_norm TEXT,          -- normalized, so SQL can compare and count
    quote TEXT, page INTEGER, -- the citation
    quote_check TEXT,         -- exact / partial / not_found (checked by code, not the model)
    model_confidence REAL,    -- what the model said
    confidence REAL,          -- adjusted by our checks
    needs_review INTEGER,     -- 1 = send to a human
    model TEXT, prompt_version TEXT, seconds REAL, extracted_at TEXT,
    PRIMARY KEY (doc_id, field));
"""

# v3: an independent check the model can't fake. A governing law answer must name a known jurisdiction.
JURISDICTIONS = {j.lower() for j in """Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia
Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota
Mississippi Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania Tennessee Texas Utah Vermont
Virginia Washington Wisconsin Wyoming England Wales Scotland Ireland Canada Israel China Japan Germany France
Switzerland Netherlands Singapore Australia India Korea Ontario Quebec Alberta Bermuda Luxembourg Sweden Italy
Spain Mexico Brazil Taiwan""".split()} | {"new hampshire", "new jersey", "new mexico", "new york", "north carolina",
"north dakota", "rhode island", "south carolina", "south dakota", "west virginia", "district of columbia",
"hong kong", "british columbia", "united kingdom", "united states", "cayman islands", "new zealand"}

def known_jurisdiction(value):
    v = (value or "").lower()
    return any(re.search(r"\b" + re.escape(j) + r"\b", v) for j in JURISDICTIONS)

def adjusted_confidence(result, quote_check, value_norm, field=None):
    c = max(0.0, min(1.0, float(result.get("confidence") or 0)))
    if field == "governing_law" and result.get("found") and value_norm and not known_jurisdiction(value_norm):
        c *= 0.5                       # e.g. "Commonwealth": not a jurisdiction on its own
    if not result.get("found"):
        return c                       # a "not found" has no quote to check; trust the model's number
    if quote_check == "partial":
        c *= 0.7
    elif quote_check == "not_found":
        c *= 0.3                       # a quote we can't find is a strong hallucination signal
    if value_norm is None:
        c *= 0.5                       # an answer we couldn't normalize can't be counted reliably
    return round(c, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="only this doc_id, e.g. D004")
    ap.add_argument("--redo", action="store_true", help="re-extract even if already stored")
    ap.add_argument("--fields", help="comma-separated subset, e.g. parties,effective_date")
    ap.add_argument("--renormalize", action="store_true",
                    help="re-clean stored raw answers with the current normalize() code; no model calls")
    args = ap.parse_args()
    if args.renormalize:
        con = sqlite3.connect(DB)
        rows = con.execute("SELECT doc_id, field, found, value_raw, quote_check, model_confidence FROM extractions").fetchall()
        for doc_id, field, found, value_raw, quote_check, model_conf in rows:
            value_norm = normalize(field, json.loads(value_raw)) if found else None
            conf = adjusted_confidence({"found": found, "confidence": model_conf}, quote_check, value_norm, field)
            review = int(conf < 0.7 or (found and quote_check == "not_found"))
            con.execute("UPDATE extractions SET value_norm=?, confidence=?, needs_review=? WHERE doc_id=? AND field=?",
                        (value_norm, conf, review, doc_id, field))
        con.commit()
        print(f"Re-normalized {len(rows)} stored answers (no model calls).")
        return

    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)

    # Load documents + family links (from Steps 1 to 3) into the database
    families = {r["doc_id"]: r for r in csv.DictReader(open("data/families.csv", encoding="utf-8"))}
    for meta_path in sorted(PARSED.glob("*.meta.json")):
        m = json.loads(meta_path.read_text())
        f = families.get(m["doc_id"], {})
        con.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)",
                    (m["doc_id"], m["filename"], m["sha256"], m.get("pages"),
                     f.get("doc_type"), f.get("master_id") or None, f.get("reason")))
    con.commit()

    doc_ids = [args.doc] if args.doc else [r[0] for r in con.execute("SELECT doc_id FROM documents ORDER BY doc_id")]
    run_start = time.time()
    for n, doc_id in enumerate(doc_ids, start=1):
        passages = load_passages(doc_id)
        for field in (args.fields.split(",") if args.fields else FIELDS):
            done = con.execute("SELECT 1 FROM extractions WHERE doc_id=? AND field=?", (doc_id, field)).fetchone()
            if done and not args.redo:
                continue
            start = time.time()
            excerpts = retrieve(passages, field)
            if excerpts:
                result = extract(field, excerpts)
            else:
                # No passage mentions any keyword (e.g. no "liability" anywhere): strong sign it's absent.
                result = {"found": False, "value": None, "quote": "", "confidence": 0.8}
            quote_check, page = verify_quote(result.get("quote", ""), passages) if result.get("found") else ("n/a", None)
            value_norm = normalize(field, result.get("value")) if result.get("found") else None
            conf = adjusted_confidence(result, quote_check, value_norm, field)
            review = int(conf < 0.7 or (result.get("found") and quote_check == "not_found"))
            con.execute("INSERT OR REPLACE INTO extractions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
                        (doc_id, field, int(bool(result.get("found"))), json.dumps(result.get("value")),
                         value_norm, result.get("quote"), page, quote_check, result.get("confidence"),
                         conf, review, MODEL, PROMPT_VERSION, round(time.time() - start, 1)))
            con.commit()
            flag = "  <- REVIEW" if review else ""
            print(f"[{n}/{len(doc_ids)}] {doc_id} {field:18} {str(value_norm)[:50]:50} "
                  f"p.{page or '-'} quote:{quote_check:9} conf {conf}{flag}")

    total, review = con.execute("SELECT COUNT(*), SUM(needs_review) FROM extractions").fetchone()
    print(f"\nStored {total} extractions in {DB}  |  flagged for review: {review}  |  "
          f"this run: {round((time.time() - run_start) / 60, 1)} min")

if __name__ == "__main__":
    main()
