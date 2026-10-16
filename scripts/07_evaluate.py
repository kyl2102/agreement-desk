"""Step 7: score extraction against your hand labels (the gold set).

Input:  data/labels/gold_labels.csv (yours), data/legal.db (the model's answers)
Output: printed scorecard + data/eval/disagreements.csv (every case where you and the model differ)

Per field:
  precision = correct answers / answers the model gave        (false hits hurt this)
  recall    = correct answers / answers that truly exist      (misses hurt this)
  A WRONG value counts against both: the model gave an answer (precision) and missed the right one (recall).
"""
import csv
import importlib.util
import json
import re
import sqlite3
from pathlib import Path

# Reuse the exact normalization code from Step 4, so gold and model are cleaned the same way.
spec = importlib.util.spec_from_file_location("extract", "scripts/04_extract.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

gold_rows = list(csv.DictReader(open("data/labels/gold_labels.csv", encoding="utf-8-sig")))
con = sqlite3.connect("data/legal.db")
pred = {(d, f): dict(zip(["found", "value_norm", "quote", "page", "needs_review", "confidence"], rest))
        for d, f, *rest in con.execute(
            "SELECT doc_id, field, found, value_norm, quote, page, needs_review, confidence FROM extractions")}

# ---------- Turn gold and model answers into comparable values (None = not found) ----------
SUFFIXES = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "company", "co",
            "sas", "sa", "plc", "lp", "the", "a", "of", "and"}

def party_key(name):
    """'NETGEAR, Inc.' and 'Netgear Inc' -> {'netgear'}; drops suffixes and punctuation."""
    return frozenset(w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in SUFFIXES)

def gold_value(doc, field):
    row = next(r for r in gold_rows if r["doc_id"] == doc and r["field"] == field)
    if row["found"].strip().lower() != "yes" or not row["value"].strip():
        return None, row
    v = row["value"].strip()
    if field == "parties":
        return frozenset(party_key(p) for p in v.split(";") if p.strip()), row
    if field == "effective_date":
        return (X.norm_date(v) or v), row          # tolerate a spreadsheet turning 2004-01-01 into 1/1/04
    if field == "governing_law":
        return (X.norm_law(v) or v).lower(), row
    if field == "termination_notice_days":
        return int(float(v)), row
    if field == "cap_type":
        v = {"fee_based": "fees_based"}.get(v.lower(), v.lower())
        return (None if v == "none" else v), row
    if field == "cap_amount":
        return float(re.sub(r"[^\d.]", "", v)), row

def model_value(doc, field):
    src = {"termination_notice_days": "termination_notice", "cap_type": "liability_cap",
           "cap_amount": "liability_cap"}.get(field, field)
    p = pred.get((doc, src))
    if not p or not p["found"] or p["value_norm"] is None:
        return None, p
    v = p["value_norm"]
    if field == "parties":
        return frozenset(party_key(x) for x in v.split(";") if x.strip()), p
    if field == "effective_date":
        return v, p
    if field == "governing_law":
        return v.lower(), p
    if field == "termination_notice_days":
        return int(v), p
    cap = json.loads(v)
    if field == "cap_type":
        return (None if cap["cap_type"] in (None, "none") else cap["cap_type"]), p
    if field == "cap_amount":
        return cap["cap_amount"], p

def same(field, g, m):
    if field == "cap_amount":
        return abs(g - m) <= 0.01 * g
    return g == m


# Save the scores to the database, so every "find all" answer can show its measured accuracy.
def save_scores(eval_set, scores):
    import time as _t
    con.execute("CREATE TABLE IF NOT EXISTS eval_results (run_at TEXT, eval_set TEXT, field TEXT, "
                "precision REAL, recall REAL, n INTEGER, prompt_version TEXT)")
    pv = ",".join(sorted({r[0] for r in con.execute("SELECT DISTINCT prompt_version FROM extractions")}))
    con.execute("DELETE FROM eval_results WHERE eval_set = ?", (eval_set,))
    con.executemany("INSERT INTO eval_results VALUES (?,?,?,?,?,?,?)",
                    [(_t.strftime("%Y-%m-%d %H:%M"), eval_set, f, p, r, n, pv) for f, p, r, n in scores])
    con.commit()

# ---------- Score ----------
FIELDS = ["parties", "effective_date", "governing_law", "termination_notice_days", "cap_type", "cap_amount"]
docs = sorted({r["doc_id"] for r in gold_rows})
disagreements = []
print(f"Scored {len(docs)} contracts against your labels\n")
print(f"{'field':24} {'precision':>9} {'recall':>7}   correct  wrong  missed  made-up  both-absent   page-match")
totals = {"tp": 0, "fp": 0, "fn": 0}
saved = []
for field in FIELDS:
    tp = wrong = missed = made_up = both_absent = page_ok = 0
    for doc in docs:
        g, grow = gold_value(doc, field)
        m, p = model_value(doc, field)
        if g is None and m is None:
            both_absent += 1
            continue
        if g is not None and m is not None and same(field, g, m):
            tp += 1
            gold_page = grow["page"].strip()
            page_ok += int(bool(p and p["page"]) and gold_page.isdigit() and int(gold_page) == p["page"])
            continue
        kind = "wrong" if (g is not None and m is not None) else ("missed" if g is not None else "made_up")
        wrong += kind == "wrong"
        missed += kind == "missed"
        made_up += kind == "made_up"
        disagreements.append({
            "doc_id": doc, "field": field, "kind": kind,
            "gold": grow["value"], "model": (p or {}).get("value_norm"),
            "model_quote": ((p or {}).get("quote") or "")[:300],
            "flagged_for_review": (p or {}).get("needs_review"), "your_notes": grow["notes"]})
    fp, fn = wrong + made_up, wrong + missed
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
    print(f"{field:24} {precision:9.0%} {recall:7.0%}   {tp:7}  {wrong:5}  {missed:6}  {made_up:7}  {both_absent:11}   {page_ok}/{tp}")
    saved.append(({"termination_notice_days": "termination_notice"}.get(field, field), precision, recall, len(docs)))

t = totals
print(f"\n{'ALL FIELDS':24} {t['tp'] / (t['tp'] + t['fp']):9.0%} {t['tp'] / (t['tp'] + t['fn']):7.0%}")

save_scores("my 10 hand labels", saved)

# Did the review queue catch the errors? (An error nobody is told to check is the dangerous kind.)
flagged = sum(1 for d in disagreements if d["flagged_for_review"])
print(f"\nErrors flagged for review: {flagged} of {len(disagreements)}  "
      f"(the rest would have reached users looking correct)")

with open("data/eval/disagreements.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(disagreements[0].keys()) if disagreements else ["doc_id"])
    w.writeheader()
    w.writerows(disagreements)
print("Every disagreement: data/eval/disagreements.csv")
