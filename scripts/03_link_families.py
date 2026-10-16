"""Step 3: link amendments (and related documents) to the master agreement they change.

Input:  data/sample.csv, data/parsed/<doc_id>.json
Output: data/families.csv  one row per contract: its type, the master it links to (if any), and WHY

Rules, not AI: every link must be explainable ("parties + type + date match").
A link is made only when the parties match AND (the agreement type OR a date matches).
No match -> "master not found". We never force a guess.
"""
import csv
import json
import re
from pathlib import Path

PARSED = Path("data/parsed")
sample = list(csv.DictReader(open("data/sample.csv", encoding="utf-8")))

def text_of(doc_id, limit):
    """The first `limit` characters of a parsed contract, as one string."""
    doc = json.load(open(PARSED / f"{doc_id}.json"))
    return " ".join(t["text"] for t in doc["texts"])[:limit]

# ---------- Rule 1: what kind of document is this? ----------
# An amendment's TITLE starts with the amendment phrase ("SECOND AMENDMENT TO...", "This Amendment is...").
# We only trust that phrase where a title sits: at the START of a text item, in the first 3000 characters.
# Why: masters often MENTION amendments mid-sentence ("no Amendment to this Agreement shall..."),
# and some PDFs have an amendment stapled on at the end (D006). Neither makes the doc an amendment.
AMEND_TITLE = re.compile(r"^\s*(this\s+)?((first|second|third|fourth|fifth)\s+)?amendment\b", re.I)
ANCILLARY = re.compile(r"power of attorney|side letter|joinder", re.I)

def doc_type(doc_id):
    doc = json.load(open(PARSED / f"{doc_id}.json"))
    seen = 0
    for item in doc["texts"]:
        if seen > 3000:
            break
        if AMEND_TITLE.search(item["text"]):
            return "amendment"
        seen += len(item["text"])
    if ANCILLARY.search(text_of(doc_id, 3000)):
        return "ancillary"   # related to a master, but doesn't change its terms
    return "master"

# ---------- Rule 2: dates, written many ways -> one standard form (ISO: 2004-01-01) ----------
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
M = "(" + "|".join(MONTHS) + ")"
DATE_PATTERNS = [
    (re.compile(M + r"\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})", re.I), ("m", "d", "y")),          # January 1, 2004
    (re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+" + M + r",?\s+(\d{4})", re.I), ("d", "m", "y")),  # 1st day of January, 1996
    (re.compile(r"(\d{1,2})\s+" + M + r"\s+(\d{4})", re.I), ("d", "m", "y")),                          # 31 December 2013
]

def dates_in(text):
    found = set()
    for pattern, order in DATE_PATTERNS:
        for match in pattern.finditer(text):
            parts = dict(zip(order, match.groups()))
            found.add(f"{int(parts['y']):04d}-{MONTHS[parts['m'].lower()]:02d}-{int(parts['d']):02d}")
    return found

# ---------- Rule 3: parties -> distinctive name words ----------
# "NETGEAR, Inc." and "Netgear" should match; "Inc", "Corporation" etc. carry no signal.
ORG = re.compile(r"((?:[A-Z][A-Za-z&'/-]+\s+){0,4}[A-Z][A-Za-z&'/-]+),?\s+(?i:Inc|Incorporated|Corporation|Corp|Company|LLC|Ltd|Limited|L\.P)\b")
GENERIC = {"the", "and", "of", "inc", "corp", "corporation", "company", "llc", "ltd", "limited",
           "international", "group", "holdings", "systems", "services", "this", "agreement", "between", "by"}

def party_words(text):
    words = set()
    for match in ORG.finditer(text):
        for w in re.findall(r"[a-z]+", match.group(1).lower()):
            if len(w) > 3 and w not in GENERIC:
                words.add(w)
    return words

# ---------- Rule 4: agreement type names, e.g. "Services Agreement" ----------
TYPE = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:and|[A-Z][a-z]+))?)\s+Agreement\b")

def agreement_types(text):
    return {m.group(1).lower() for m in TYPE.finditer(text) if m.group(1).lower() not in {"this", "the", "such", "existing"}}

# ---------- Build features for every contract ----------
docs = {}
for row in sample:
    head = text_of(row["doc_id"], 3000)       # where titles and parties appear
    body = text_of(row["doc_id"], 8000)       # where an amendment points back to its master
    docs[row["doc_id"]] = {
        "type": doc_type(row["doc_id"]),
        "head_upper": head.upper(),
        "dates": dates_in(head),
        "ref_dates": dates_in(body),
        "parties": party_words(head),
        "types": agreement_types(body),
    }

# ---------- Link each amendment / ancillary doc to its best master ----------
results = []
for doc_id, d in docs.items():
    row = {"doc_id": doc_id, "doc_type": d["type"], "master_id": "", "confidence": "", "reason": ""}
    if d["type"] in ("amendment", "ancillary"):
        best = None
        for mid, m in docs.items():
            if mid == doc_id or m["type"] != "master":
                continue
            shared_parties = d["parties"] & m["parties"]
            shared_types = {t for t in d["types"] if t.upper() + " AGREEMENT" in m["head_upper"]}
            shared_dates = d["ref_dates"] & m["dates"]
            if not shared_parties or not (shared_types or shared_dates):
                continue  # parties must match, plus at least one more signal
            score = 1 + bool(shared_types) + bool(shared_dates)
            reason = (f"parties: {', '.join(sorted(shared_parties))}"
                      + (f"; type: {', '.join(sorted(shared_types))}" if shared_types else "")
                      + (f"; date: {', '.join(sorted(shared_dates))}" if shared_dates else ""))
            if best is None or score > best[0]:
                best = (score, mid, reason)
        if best:
            row.update(master_id=best[1], confidence={3: "high", 2: "medium"}[best[0]], reason=best[2])
        else:
            row.update(reason="master not found in sample")
    results.append(row)

with open("data/families.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["doc_id", "doc_type", "master_id", "confidence", "reason"])
    w.writeheader()
    w.writerows(results)

# ---------- Check against the answer we already know ----------
# From reading the filenames: which master each linked document truly belongs to ("" = none in sample).
EXPECTED = {"D002": "D001", "D003": "D001", "D004": "D001", "D005": "D006", "D007": "D006",
            "D008": "D009", "D010": "D011", "D012": "", "D013": "", "D014": "",
            "D028": "", "D040": ""}  # D028: amendment, master not in sample. D040: a master, not an amendment.
print("doc   type       -> master  conf    reason")
for r in results:
    if r["doc_type"] != "master" or r["doc_id"] in EXPECTED:
        print(f"{r['doc_id']}  {r['doc_type']:10} -> {r['master_id'] or '-':6}  {r['confidence'] or '-':6}  {r['reason']}")
right = sum(1 for k, v in EXPECTED.items() if next(r for r in results if r["doc_id"] == k)["master_id"] == v)
extra = [r["doc_id"] for r in results if r["master_id"] and r["doc_id"] not in EXPECTED]
print(f"\nCorrect on the known answers: {right}/{len(EXPECTED)}  |  unexpected links: {extra or 'none'}")
print("Wrote data/families.csv")
