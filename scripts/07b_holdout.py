"""Holdout test: score extraction on the 30 contracts you did NOT label, using CUAD's answer key.

Why: prompt fixes tuned on your 10 contracts might just "memorize" them (overfitting).
If a fix also helps on 30 contracts it has never seen, the improvement is real.

Usage: uv run python scripts/07b_holdout.py data/legal.db
Fields: parties, effective_date, governing_law (the three CUAD labels that match our schema).
Read results as v2 vs v3 on the SAME answer key; CUAD's rules differ slightly from your guide.
"""
import csv
import importlib.util
import re
import sqlite3
import sys

spec = importlib.util.spec_from_file_location("extract", "scripts/04_extract.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

DB = sys.argv[1] if len(sys.argv) > 1 else "data/legal.db"
LABELED = {r["doc_id"] for r in csv.DictReader(open("data/labels/gold_labels.csv", encoding="utf-8-sig"))}
sample = {r["doc_id"]: r["filename"] for r in csv.DictReader(open("data/sample.csv", encoding="utf-8"))}
cuad = {r["Filename"]: r for r in csv.DictReader(open("data/raw/cuad/CUAD_v1/master_clauses.csv", encoding="utf-8"))}
con = sqlite3.connect(DB)
pred = {(d, f): (found, v) for d, f, found, v in con.execute("SELECT doc_id, field, found, value_norm FROM extractions")}
holdout = [d for d in sample if d not in LABELED]

GENERIC = {"inc", "incorporated", "llc", "ltd", "ltc", "limited", "corp", "corporation", "company", "co", "sas", "sa",
           "plc", "lp", "l", "the", "a", "of", "and", "party", "parties", "each", "collectively", "individually"}

def party_words(name):
    name = re.sub(r"\([^)]*\)|[\"“”'‘’]", " ", name)          # drop CUAD's ("Adviser") nicknames
    return {w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in GENERIC and len(w) > 1}

# ---------- CUAD's answer -> our format (None = not found, "skip" = unusable label) ----------
def cuad_value(doc, field):
    r = cuad[sample[doc]]
    if field == "parties":
        # split on ";" but not inside parentheses: 'Pretzel Time, Inc. ("Pretzel Time"; "FRANCHISOR")'
        parts = re.split(r";(?![^()]*\))", r["Parties-Answer"])
        ents = [party_words(X.clean_party(p)) for p in parts]
        ents = [e for e in ents if e]                        # drops "[]", "(each a Party...)" fragments
        return ents or None
    if field == "effective_date":
        # Same rule as your guide: effective date, else the agreement date.
        for raw in (r["Effective Date-Answer"], r["Agreement Date-Answer"]):
            if "[]" in raw:
                return "skip"                                # date left blank in the contract
            for part in raw.split(";"):
                d = X.norm_date(part.strip())
                if d:
                    return d
        return None
    if field == "governing_law":
        return r["Governing Law-Answer"].strip().lower() or None

def model_value(doc, field):
    found, v = pred.get((doc, field), (0, None))
    if not found or v is None:
        return None
    if field == "parties":
        return [party_words(p) for p in v.split(";") if party_words(p)]
    return v.lower()

def same(field, g, m):
    if field == "parties":   # every CUAD party matches a model party and vice versa (shared distinctive word)
        return all(any(e & p for p in m) for e in g) and all(any(p & e for e in g) for p in m)
    if field == "governing_law":   # "england" vs "england and wales", "british columbia, canada" vs "british columbia"
        return g in m or m in g
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

saved = []
print(f"Holdout: {len(holdout)} contracts, CUAD answer key, database {DB}\n")
print(f"{'field':16} {'precision':>9} {'recall':>7}   correct wrong missed made-up both-absent skipped")
misses = []
for field in ["parties", "effective_date", "governing_law"]:
    tp = wrong = missed = made_up = absent = skipped = 0
    for doc in holdout:
        g, m = cuad_value(doc, field), model_value(doc, field)
        if g == "skip":
            skipped += 1
            continue
        if g is None and m is None:
            absent += 1
        elif g is not None and m is not None and same(field, g, m):
            tp += 1
        else:
            kind = "wrong" if g is not None and m is not None else "missed" if g is not None else "made_up"
            wrong += kind == "wrong"; missed += kind == "missed"; made_up += kind == "made_up"
            misses.append((doc, field, kind))
    p = tp / (tp + wrong + made_up) if tp + wrong + made_up else float("nan")
    rc = tp / (tp + wrong + missed) if tp + wrong + missed else float("nan")
    print(f"{field:16} {p:9.0%} {rc:7.0%}   {tp:7} {wrong:5} {missed:6} {made_up:7} {absent:11} {skipped:7}")
    saved.append((field, p, rc, len(holdout)))
if DB == "data/legal.db":
    save_scores("30-contract holdout (CUAD key)", saved)
print("\nDisagreements:", ", ".join(f"{d} {f} ({k})" for d, f, k in misses) or "none")
