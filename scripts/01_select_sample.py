"""Step 2: pick a deliberate (stratified) sample of ~40 CUAD contracts.

Input:  data/raw/cuad/CUAD_v1/master_clauses.csv  (the answer key)
Output: data/sample.csv  (one row per chosen contract: file, path, why it was chosen)

We never copy or edit the raw PDFs. The sample is just a list pointing at them.
"""
import csv
import random
import re
from pathlib import Path

RAW = Path("data/raw/cuad/CUAD_v1")
OUT = Path("data/sample.csv")
random.seed(42)  # fixed seed = the "random" picks are the same every run (reproducible)

rows = list(csv.DictReader(open(RAW / "master_clauses.csv", encoding="utf-8")))

# Map each filename to its PDF on disk. Lowercase match, because the
# extensions are a mix of .pdf and .PDF (the case-sensitivity trap from earlier).
# sorted(): file listing order differs between computers, and the random
# picks below depend on order. Sorting keeps results identical everywhere.
pdfs = {p.name.lower(): p for p in sorted(RAW.glob("full_contract_pdf/**/*")) if p.suffix.lower() == ".pdf"}

chosen = {}  # filename -> reason. A dict, so a contract is never picked twice.

def pick(filenames, reason, limit=None):
    added = 0
    for f in filenames:
        if limit is not None and added >= limit:
            break
        if f not in chosen and f.lower() in pdfs:
            chosen[f] = reason
            added += 1

# 1. Agreement families: every file whose name contains one of these company prefixes.
FAMILIES = ["FEDERATEDGOVERNMENTINCOME", "NETGEAR", "BIOAMBER", "NEONSYSTEMS",
            "ADMA BioManufacturing", "Cerus Corporation", "Columbia Laboratories"]
for prefix in FAMILIES:
    pick([r["Filename"] for r in rows if r["Filename"].startswith(prefix)], f"family:{prefix}")

# 2. Liability caps that state a dollar figure, like "$1,000,000".
dollar = [r["Filename"] for r in rows if re.search(r"\$\s?[\d,]{4,}", r["Cap On Liability"])]
random.shuffle(dollar)
pick(dollar, "cap:dollar_amount", limit=6)

# 3. Fees-based caps, like "amounts paid ... in the twelve months preceding".
fees = [r["Filename"] for r in rows
        if re.search(r"(amounts?|fees?|charges?) (paid|payable)", r["Cap On Liability"], re.I)
        and re.search(r"exceed|limited to|maximum", r["Cap On Liability"], re.I)]
random.shuffle(fees)
pick(fees, "cap:fees_based", limit=5)

# 4. No governing law labeled: tests whether the model invents an answer.
no_law = [r["Filename"] for r in rows if not r["Governing Law-Answer"].strip()]
random.shuffle(no_law)
pick(no_law, "edge:no_governing_law", limit=4)

# 5. Fill to 40, one contract type at a time, so the rest is spread across types.
by_type = {}
for f, p in pdfs.items():
    by_type.setdefault(p.parent.name, []).append(p.name)
types = sorted(by_type)
random.shuffle(types)
name_lookup = {r["Filename"].lower(): r["Filename"] for r in rows}
while len(chosen) < 40:
    for t in types:
        candidates = [name_lookup[n.lower()] for n in by_type[t]
                      if n.lower() in name_lookup and name_lookup[n.lower()] not in chosen]
        if candidates and len(chosen) < 40:
            pick([random.choice(candidates)], f"fill:{t}")

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["doc_id", "filename", "pdf_path", "reason"])
    for i, (f, reason) in enumerate(chosen.items(), start=1):
        w.writerow([f"D{i:03d}", f, pdfs[f.lower()].as_posix(), reason])

# Count check: how do we know we got everything? Compare in vs out.
print(f"PDFs on disk: {len(pdfs)}  |  rows in answer key: {len(rows)}  |  sampled: {len(chosen)}")
for reason in sorted({r.split(':')[0] for r in chosen.values()}):
    print(f"  {reason}: {sum(1 for r in chosen.values() if r.startswith(reason))}")
print(f"Wrote {OUT}")
