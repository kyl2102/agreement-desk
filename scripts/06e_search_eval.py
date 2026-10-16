"""Step 6e: grade search quality: keyword only vs meaning only vs hybrid.

Each test question has a RELEVANCE RULE: a pattern a correct chunk must match (written in legal terms).
Most questions are phrased in plain English that does NOT use those terms, so keyword search gets no free ride.
Metrics on the top 5 results:
  hit@5        did at least one correct chunk appear? (did the user find anything useful?)
  precision@5  what share of the top 5 were correct?  (how much junk did they wade through?)
"""
import importlib.util
import re
import sys

spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
Q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(Q)

# (question, relevance rule: ALL patterns must match the chunk text, case-insensitive)
TESTS = [
    ("how much warning to end the deal early",       [r"terminat", r"convenience|without cause|for any reason|at any time", r"notice"]),
    ("who pays the lawyers if we fight",              [r"attorneys?'? ?fees|prevailing party|legal fees"]),
    ("can they hand this contract to someone else",   [r"\bassign", r"consent"]),
    ("what if a hurricane stops them from performing", [r"force majeure|acts? of god"]),
    ("keeping each other's secrets",                  [r"confidential", r"disclos"]),
    ("stop poaching each other's staff",              [r"solicit", r"employ"]),
    ("can we check their books",                      [r"audit|inspect", r"records|books"]),
    ("which state's law applies",                     [r"governed by|governing law|laws of the state"]),
    ("Section 14.A",                                  [r"\b14\.a\b"]),
    ("$150,000",                                      [r"150,000"]),
]

texts = dict(Q.con.execute("SELECT chunk_id, text FROM chunks").fetchall())

def relevant(chunk_id, rule):
    t = texts[chunk_id].lower()
    return all(re.search(p, t) for p in rule)

methods = {"keyword only": Q.keyword_search, "meaning only": Q.vector_search,
           "hybrid (RRF)": lambda q: Q.rrf(Q.keyword_search(q), Q.vector_search(q))}
totals = {m: [0, 0.0] for m in methods}
print(f"{'question':46} " + "  ".join(f"{m:>14}" for m in methods) + "   relevant chunks")
for question, rule in TESTS:
    n_rel = sum(relevant(c, rule) for c in texts)
    cells = []
    for name, search in methods.items():
        top = search(question)[:5]
        good = sum(relevant(c, rule) for c in top)
        totals[name][0] += good > 0
        totals[name][1] += good / 5
        cells.append(f"{good}/5 {'hit' if good else 'MISS':>4}")
    print(f"{question[:46]:46} " + "  ".join(f"{c:>14}" for c in cells) + f"   {n_rel}")

print(f"\n{'':46} " + "  ".join(f"{m:>14}" for m in methods))
print(f"{'hit@5 (found something useful)':46} " + "  ".join(f"{totals[m][0] / len(TESTS):>14.0%}" for m in methods))
print(f"{'precision@5 (share of top 5 correct)':46} " + "  ".join(f"{totals[m][1] / len(TESTS):>14.0%}" for m in methods))
