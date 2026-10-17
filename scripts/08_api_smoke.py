"""Step 8d smoke test: hit the running API with questions whose answers we already know.

Run the server first (uv run uvicorn api:app --port 8000), then in another tab:
  uv run python scripts/08_api_smoke.py
Read-only: it never saves a review. Uses the real local models, so the planner questions take a few seconds each.
"""
import json
import urllib.request

BASE = "http://localhost:8000/api/"

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.headers.get("Content-Type", ""), r.read()

def ask(question, mode="auto"):
    req = urllib.request.Request(BASE + "ask", data=json.dumps({"question": question, "mode": mode}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

def complete(r):
    return next((a for a in r["answers"] if a["badge"] == "complete"), {})

CHECKS = [
    ("meta: 33 agreements, 40 documents", lambda: json.loads(get("meta")[1])["agreements"] == 33),
    ("Ingram resolves to Netgear, 90 days", lambda: (lambda a: "90 days" in a.get("headline", "") and a.get("agreement_id") == "D006")(
        complete(ask("whats the termination notice period for ingram", "all")))),
    ("Bay Networks: mentions, not a bare zero", lambda: (lambda a: a.get("headline", "").startswith("No agreement lists")
        and len(a.get("rows", [])) > 0)(complete(ask("Which agreements are with Bay Networks?", "all")))),
    ("No-cap question: absence check runs", lambda: "have no liability cap extracted" in
        complete(ask("Which agreements have no liability cap?", "all")).get("headline", "")),
    ("New York count names the value", lambda: "New York" in
        complete(ask("How many agreements are governed by New York law?", "all")).get("headline", "")),
    ("Examples desk returns 5 passages", lambda: len(next(a for a in ask("who pays the lawyers if we fight", "examples")["answers"]
        if a["badge"] == "examples")["results"]) == 5),
    ("Netgear documents in order 1 D006, 2 D005, 3 D007", lambda: [d["id"] for d in json.loads(get("agreement/D006")[1])["documents"]]
        == ["D006", "D005", "D007"]),
    ("Netgear notice in force is 90 days", lambda: any(t["field"] == "termination_notice" and t["value"] == "90 days"
        for t in json.loads(get("agreement/D006")[1])["terms"])),
    ("Review queue loads", lambda: isinstance(json.loads(get("review")[1])["items"], list)),
    ("Page image renders", lambda: get("page/D006/6?q=ninety")[0].startswith("image/png")),
]

passed = 0
for label, check in CHECKS:
    try:
        ok = bool(check())
        detail = ""
    except Exception as e:
        ok, detail = False, f"  ({type(e).__name__}: {e})"
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}{detail}")
print(f"\n{passed} of {len(CHECKS)} passed")
