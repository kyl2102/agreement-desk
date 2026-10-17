"""Step 6g: an LLM JUDGE for search results, calibrated against human calls before it is trusted.

Why: 06e grades search with regex rules ("contains 'assign' and 'consent'"). That is fast but dumb, and it
doesn't scale: every new test question needs a hand-written rule. A judge model reads the question and the
passage together and says relevant or not, with a reason.

Why a DIFFERENT model family: qwen produces this tool's extractions and plans; a judge from the same family
tends to agree with its own kind of output. Gemma (Google) grades instead.

Why calibrate: an unchecked judge is just a second unverified model. Stage 2 compares the judge with the regex
rules; YOU decide the disagreements; stage 3 reports how often the judge was right.

Usage:
  uv run python scripts/06g_judge_search.py            judge + compare + re-score (cached, so reruns are fast)
  uv run python scripts/06g_judge_search.py --check    after you fill the "ken" column in judge_disagreements.csv
  uv run python scripts/06g_judge_search.py --spotcheck        write a random sample of AGREEMENTS to label blind
  uv run python scripts/06g_judge_search.py --check-spot       score those labels (tests "agreed = both right")
  uv run python scripts/06g_judge_search.py --expand           also judge the JUDGE_ONLY questions; scoreboard on all
"""
import argparse
import csv
import importlib.util
import json
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true", help="score the judge against your labels in the CSV")
parser.add_argument("--spotcheck", action="store_true", help="sample judge/regex agreements to label blind")
parser.add_argument("--check-spot", action="store_true", help="score your spot-check labels")
parser.add_argument("--expand", action="store_true", help="add the judge-only questions to the scoreboard")
args = parser.parse_args()

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
Q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(Q)
from search_tests import TESTS, JUDGE_ONLY

JUDGE_MODEL = "gemma4:12b"
DISAGREE_CSV = "data/eval/judge_disagreements.csv"
SPOT_CSV = "data/eval/judge_spotcheck.csv"
SPOT_N = 15

JUDGE_PROMPT = """You grade results from a contract search tool. You see a user's question and ONE passage from a contract.
The passage is RELEVANT only if it contains contract language that directly addresses the question:
a clause a lawyer would point to when answering it.
NOT relevant: a passing mention of the topic, a definition or heading alone, or a clause about a different issue
that happens to share words (notice of a price change is not notice of termination).
Judge only this passage. Give a reason of at most 20 words."""
JUDGE_SCHEMA = {"type": "object", "properties": {"relevant": {"type": "boolean"}, "reason": {"type": "string"}},
                "required": ["relevant", "reason"]}

Q.con.execute("""CREATE TABLE IF NOT EXISTS judge_labels (
    question TEXT, chunk_id TEXT, judge_model TEXT, relevant INTEGER, reason TEXT,
    judged_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (question, chunk_id, judge_model))""")

chunks = {c: (h, t) for c, h, t in Q.con.execute("SELECT chunk_id, heading_path, text FROM chunks")}

def regex_relevant(chunk_id, rule):
    t = chunks[chunk_id][1].lower()
    return all(re.search(p, t) for p in rule)

def judge(question, chunk_id):
    """Cached: a (question, passage, judge) triple is only ever sent to the model once."""
    row = Q.con.execute("SELECT relevant, reason FROM judge_labels WHERE question=? AND chunk_id=? AND judge_model=?",
                        (question, chunk_id, JUDGE_MODEL)).fetchone()
    if row:
        return bool(row[0]), row[1]
    import ollama
    heading, text = chunks[chunk_id]
    r = ollama.chat(model=JUDGE_MODEL, format=JUDGE_SCHEMA, options={"temperature": 0},
                    messages=[{"role": "system", "content": JUDGE_PROMPT},
                              {"role": "user", "content": f"Question: {question}\n\nPassage (section: {heading}):\n{text[:1500]}"}])
    out = json.loads(r["message"]["content"])
    Q.con.execute("INSERT OR REPLACE INTO judge_labels (question, chunk_id, judge_model, relevant, reason) VALUES (?,?,?,?,?)",
                  (question, chunk_id, JUDGE_MODEL, int(out["relevant"]), out["reason"]))
    Q.con.commit()
    return bool(out["relevant"]), out["reason"]

def top5_by_method(question):
    kw, vec = Q.keyword_search(question), Q.vector_search(question)
    fused = Q.rrf(kw, vec)
    return {"keyword only": kw[:5], "meaning only": vec[:5], "hybrid (RRF)": fused[:5],
            "rerank MiniLM": Q.rerank(question, fused[:Q.RERANK_POOL], model="minilm"),
            "rerank BGE": Q.rerank(question, fused[:Q.RERANK_POOL], model="bge")}

# ---------- --check: how often was the judge right, on the cases where it disagreed with the regex? ----------
if args.check:
    rows = list(csv.DictReader(open(DISAGREE_CSV)))
    done = [r for r in rows if r["ken"].strip().lower() in ("y", "n")]
    if not done:
        sys.exit(f"No labels yet: put y or n in the 'ken' column of {DISAGREE_CSV}")
    judge_right = sum((r["judge"] == "y") == (r["ken"].strip().lower() == "y") for r in done)
    agreed = int(open(DISAGREE_CSV + ".agreed").read())
    total = agreed + len(done)
    # Where judge and regex AGREED we assume both were right (spot-check a few to test that assumption).
    print(f"You labeled {len(done)} of {len(rows)} disagreements.")
    print(f"  judge right {judge_right}, regex right {len(done) - judge_right}")
    print(f"Estimated judge accuracy: {(agreed + judge_right) / total:.0%}  "
          f"(regex: {(agreed + len(done) - judge_right) / total:.0%})  over {total} judgments")
    print("Rule of thumb: trust the judge on new questions only at roughly 90% or better.")
    sys.exit()

# ---------- --check-spot: were the AGREEMENTS really both right? (the assumption behind the accuracy estimate) ----------
if args.check_spot:
    rows = [r for r in csv.DictReader(open(SPOT_CSV)) if r["ken"].strip().lower() in ("y", "n")]
    if not rows:
        sys.exit(f"No labels yet: put y or n in the 'ken' column of {SPOT_CSV}")
    right = sum(judge(r["question"], r["chunk_id"])[0] == (r["ken"].strip().lower() == "y") for r in rows)
    print(f"Spot-check: you agreed with the graders' shared verdict on {right} of {len(rows)} ({right / len(rows):.0%}).")
    print("If that's near 100%, the 'agreed = both right' assumption holds and the accuracy estimate stands.")
    sys.exit()

# ---------- Stage 1: judge every passage any method put in a top 5 ----------
print(f"Judge: {JUDGE_MODEL}. Judging each question's top-5 results from all 5 methods (cached after the first run).\n")
results, pool = {}, {}
for question, rule in TESTS:
    results[question] = top5_by_method(question)
    pool[question] = sorted({c for top in results[question].values() for c in top})
n = sum(len(p) for p in pool.values())
for i, (question, chunk_ids) in enumerate(pool.items(), 1):
    for c in chunk_ids:
        judge(question, c)
    print(f"  [{i}/{len(pool)}] {question[:50]}: {len(chunk_ids)} passages")

# ---------- Stage 2: compare with the regex rules; write the disagreements for a human ----------
agree, disagreements = 0, []
both_yes = judge_only = regex_only = both_no = 0
for question, rule in TESTS:
    for c in pool[question]:
        j, reason = judge(question, c)
        r = regex_relevant(c, rule)
        both_yes += j and r; judge_only += j and not r; regex_only += r and not j; both_no += not j and not r
        if j == r:
            agree += 1
        else:
            heading, text = chunks[c]
            disagreements.append({"question": question, "chunk_id": c, "section": heading.split(" > ")[-1][:60],
                                  "regex": "y" if r else "n", "judge": "y" if j else "n", "judge_reason": reason,
                                  "text": re.sub(r"\s+", " ", text)[:500], "ken": ""})
with open(DISAGREE_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(disagreements[0]) if disagreements else ["question"])
    w.writeheader(); w.writerows(disagreements)
open(DISAGREE_CSV + ".agreed", "w").write(str(agree))

if args.spotcheck:
    import random
    agreed_rows = [(q, c) for q, rule in TESTS for c in pool[q] if judge(q, c)[0] == regex_relevant(c, rule)]
    random.seed(7)
    sample = random.sample(agreed_rows, min(SPOT_N, len(agreed_rows)))
    with open(SPOT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "chunk_id", "section", "text", "ken"])
        w.writeheader()
        for q, c in sample:   # BLIND: no verdict columns, so they can't sway you
            heading, text = chunks[c]
            w.writerow({"question": q, "chunk_id": c, "section": heading.split(" > ")[-1][:60],
                        "text": re.sub(r"\s+", " ", text)[:500], "ken": ""})
    sys.exit(f"Wrote {len(sample)} passages to {SPOT_CSV} with no verdicts shown. Tag y/n, then run --check-spot.")

print(f"\n{n} judgments. Judge and regex agree on {agree} ({agree / n:.0%}).")
print(f"                  regex: relevant   regex: not")
print(f"  judge: relevant      {both_yes:4}          {judge_only:4}")
print(f"  judge: not           {regex_only:4}          {both_no:4}")
print(f"\n{len(disagreements)} disagreements written to {DISAGREE_CSV}.")
print("  Open it, read each passage, and put y (relevant) or n in the 'ken' column. Then run with --check.")

# ---------- Stage 3: the search scoreboard, graded both ways ----------
print(f"\n{'':38}{'graded by regex':>22}{'graded by judge':>22}")
print(f"{'method':38}{'hit@5':>11}{'prec@5':>11}{'hit@5':>11}{'prec@5':>11}")
for method in results[TESTS[0][0]]:
    rh = rp = jh = jp = 0
    for question, rule in TESTS:
        top = results[question][method]
        rg = sum(regex_relevant(c, rule) for c in top)
        jg = sum(judge(question, c)[0] for c in top)
        rh += rg > 0; rp += rg / 5; jh += jg > 0; jp += jg / 5
    k = len(TESTS)
    print(f"{method:38}{rh / k:>11.0%}{rp / k:>11.0%}{jh / k:>11.0%}{jp / k:>11.0%}")
print("\nIf the two columns rank the methods the same way, the reranker decision holds under either grader.")

# ---------- Stage 4 (--expand): more questions, graded by the judge alone ----------
if args.expand:
    questions = [q for q, _ in TESTS] + JUDGE_ONLY
    print(f"\nExpanding to {len(questions)} questions ({len(JUDGE_ONLY)} judge-only). New passages get judged now...")
    for i, q in enumerate(JUDGE_ONLY, 1):
        results[q] = top5_by_method(q)
        for c in {c for top in results[q].values() for c in top}:
            judge(q, c)
        print(f"  [{i}/{len(JUDGE_ONLY)}] {q[:55]}")
    print(f"\n{'method':38}{'hit@5':>11}{'prec@5':>11}   (graded by judge, {len(questions)} questions)")
    for method in results[questions[0]]:
        hits = prec = 0
        for q in questions:
            g = sum(judge(q, c)[0] for c in results[q][method])
            hits += g > 0; prec += g / 5
        print(f"{method:38}{hits / len(questions):>11.0%}{prec / len(questions):>11.0%}")
    misses = [q for q in questions if not any(judge(q, c)[0] for c in results[q]["rerank BGE"])]
    print(f"\nBGE found nothing relevant for {len(misses)} questions:")
    for q in misses:
        print(f"  - {q}")
    print("A miss can mean search failed OR the 40 contracts don't contain that clause; check before blaming search.")
