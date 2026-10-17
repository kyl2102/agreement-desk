"""Step 6e: grade search quality: keyword only vs meaning only vs hybrid vs hybrid + reranker.

Each test question has a RELEVANCE RULE: a pattern a correct chunk must match (written in legal terms).
Most questions are phrased in plain English that does NOT use those terms, so keyword search gets no free ride.
Metrics on the top 5 results:
  hit@5        did at least one correct chunk appear? (did the user find anything useful?)
  precision@5  what share of the top 5 were correct?  (how much junk did they wade through?)
"""
import importlib.util
import re
import sys

sys.path.insert(0, "scripts")

spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
Q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(Q)

from search_tests import TESTS   # shared with the LLM judge (06g)

texts = dict(Q.con.execute("SELECT chunk_id, text FROM chunks").fetchall())

def relevant(chunk_id, rule):
    t = texts[chunk_id].lower()
    return all(re.search(p, t) for p in rule)

methods = {"keyword only": Q.keyword_search, "meaning only": Q.vector_search,
           "hybrid (RRF)": lambda q: Q.rrf(Q.keyword_search(q), Q.vector_search(q)),
           "rerank MiniLM": lambda q: Q.rerank(q, Q.rrf(Q.keyword_search(q), Q.vector_search(q))[:Q.RERANK_POOL], model="minilm"),
           "rerank BGE": lambda q: Q.rerank(q, Q.rrf(Q.keyword_search(q), Q.vector_search(q))[:Q.RERANK_POOL], model="bge")}
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
