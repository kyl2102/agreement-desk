"""Open a sampled contract's original PDF by its ID.  Usage: uv run python scripts/open_doc.py D012"""
import csv, subprocess, sys
doc_id = sys.argv[1].upper()
row = next(r for r in csv.DictReader(open("data/sample.csv", encoding="utf-8")) if r["doc_id"] == doc_id)
print(f"{doc_id}: {row['filename']}")
subprocess.run(["open", row["pdf_path"]])   # macOS: open in the default PDF app (Preview)
