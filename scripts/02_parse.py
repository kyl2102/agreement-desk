"""Step 2: parse all sampled contracts with Docling.

Input:  data/sample.csv (the 40 contracts chosen in Step 1)
Output: data/parsed/<doc_id>.json       full structure + page numbers (for the pipeline)
        data/parsed/<doc_id>.md         readable version (for you)
        data/parsed/<doc_id>.meta.json  small summary of the parse
        data/parsed/parse_log.csv       one row per contract, rebuilt each run

Safeguards:
  - Idempotent: a contract that already has a .meta.json is skipped, so re-running is fast and safe.
  - Fault tolerant: if one contract fails, log the error and keep going.
  - Fingerprinted: we record a SHA-256 hash of each PDF (a unique fingerprint of its bytes),
    so we can later prove exactly which version of a file we parsed.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

SAMPLE = Path("data/sample.csv")
OUT = Path("data/parsed")
OUT.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()  # loads Docling's layout models once, reused for every file
sample = list(csv.DictReader(open(SAMPLE, encoding="utf-8")))

for n, row in enumerate(sample, start=1):
    doc_id, pdf_path = row["doc_id"], Path(row["pdf_path"])
    meta_path = OUT / f"{doc_id}.meta.json"
    if meta_path.exists():
        print(f"[{n}/{len(sample)}] {doc_id} already parsed, skipping")
        continue

    print(f"[{n}/{len(sample)}] {doc_id} parsing {pdf_path.name[:60]}...")
    meta = {"doc_id": doc_id, "filename": row["filename"],
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest()}
    start = time.time()
    try:
        result = converter.convert(pdf_path)
        doc = result.document
        doc.save_as_json(OUT / f"{doc_id}.json")
        (OUT / f"{doc_id}.md").write_text(doc.export_to_markdown(), encoding="utf-8")
        meta.update(
            status=str(result.status.value),
            pages=len(doc.pages),
            text_items=len(doc.texts),
            headings=sum(1 for t in doc.texts if t.label in ("section_header", "title")),
            tables=len(doc.tables),
            chars=sum(len(t.text) for t in doc.texts),
        )
    except Exception as e:  # keep going; record what went wrong
        meta.update(status="error", error=str(e)[:300])
    meta["seconds"] = round(time.time() - start, 1)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"    -> {meta['status']}, {meta.get('pages', '?')} pages, "
          f"{meta.get('headings', '?')} headings, {meta['seconds']}s")

# Rebuild the log from every .meta.json, so it always reflects all runs.
metas = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.meta.json"))]
cols = ["doc_id", "status", "pages", "text_items", "headings", "tables", "chars",
        "seconds", "sha256", "filename", "error"]
with open(OUT / "parse_log.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for m in metas:
        w.writerow({c: m.get(c, "") for c in cols})

# The "did we get everything?" check: inputs vs outputs.
ok = sum(1 for m in metas if m["status"] == "success")
print(f"\nSample: {len(sample)}  |  parsed OK: {ok}  |  other: {len(metas) - ok}")
print(f"Total pages: {sum(m.get('pages', 0) for m in metas)}  |  "
      f"total time: {round(sum(m['seconds'] for m in metas) / 60, 1)} min")
print(f"Log: {OUT / 'parse_log.csv'}")
