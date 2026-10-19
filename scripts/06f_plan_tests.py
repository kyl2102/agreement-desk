"""Regression tests for the query planner: run after ANY change to the planner prompt or tools.

Each test: a question + what the final plan (after code repairs) must look like.
Why: a prompt change made to fix one question ("notice periods after 2000") broke another
("agreements executed between 2000 and 2004"). Without tests, that's only caught by users.

Usage: uv run python scripts/06f_plan_tests.py      (about 1 minute; calls the local model)
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
Q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(Q)

TESTS = [
    ("Which agreements have no liability cap?",               {"tool": "list_where", "field": "cap_type"}),
    ("How many agreements are governed by New York law?",     {"tool": "count_by", "field": "governing_law"}),
    ("Agreements executed between 2000 and 2004",              {"tool": "date_between", "start": "2000-01-01", "end": "2004-12-31"}),
    ("list of agreements executed between 2000 and 2004",      {"tool": "date_between", "start": "2000-01-01"}),
    ("Notice periods longer than 60 days",                     {"tool": "compare", "field": "termination_notice", "op": ">"}),
    ("Notice periods distribution of agreements after 2000",   {"field": "termination_notice", "start": "2000-01-01"}),
    ("Which contracts are governed by California law?",        {"tool": "list_where", "field": "governing_law"}),
    ("Contracts effective before 2005",                        {"tool": "date_between", "end": "2004-12-31"}),
    ("Which agreements cap liability at a fixed dollar amount?", {"field": "cap_type"}),
    ("Count agreements by liability cap type",                 {"tool": "count_by", "field": "cap_type"}),
    # People name agreements by counterparty, not by ID (Ken's question that was refused):
    ("whats the termination notice period for ingram",         {"tool": "terms_in_force", "doc_id": "D006", "field": "termination_notice"}),
    ("What law governs the Netgear agreement?",               {"tool": "terms_in_force", "doc_id": "D006", "field": "governing_law"}),
    # Structure questions once came back as a governing-law count (Ken, Oct 16):
    ("how many amendments are there",                          {"tool": "count_documents"}),
    # The planner wrote value "null" and "> 0", which counted agreements whose notice IS "null": 0 of 33 (Oct 19)
    ("how many agreements have notice of termination clauses", {"tool": "count_by", "field": "termination_notice", "value": None}),
]

passed = 0
for question, expect in TESTS:
    plan, repaired = Q.repair_plan(Q.plan(question))
    try:
        Q.validate_plan(plan)
        valid = True
    except ValueError as e:
        valid = False
        reason = str(e)
    wrong = {k: (plan.get(k), v) for k, v in expect.items() if plan.get(k) != v}
    ok = valid and not wrong
    passed += ok
    detail = "" if ok else (f"  refused: {reason}" if not valid else f"  got vs expected: {wrong}")
    print(f"{'PASS' if ok else 'FAIL'}  {question}{'  (repaired)' if repaired else ''}{detail}")
print(f"\n{passed} of {len(TESTS)} passed")
