"""Learning tool: show what Docling "saw" on each page.

Draws a colored box around every piece of text Docling found, using the
page number + bounding box (bbox) stored in its JSON output, and lists each
item next to the page. Writes an HTML file you open in your browser.

Usage: uv run python scripts/viz_parse.py <docling .json> <original .pdf>
"""
import base64
import html
import io
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium          # renders PDF pages to images (installed with Docling)
from PIL import ImageDraw           # draws boxes on images (also installed with Docling)

json_path, pdf_path = Path(sys.argv[1]), Path(sys.argv[2])
doc = json.load(open(json_path))
pdf = pdfium.PdfDocument(str(pdf_path))
SCALE = 2  # render at 2x so text is readable

COLORS = {"section_header": "#d62728", "title": "#d62728", "text": "#1f77b4",
          "list_item": "#2ca02c", "table": "#ff7f0e", "page_header": "#9467bd",
          "page_footer": "#9467bd"}

# Group Docling's items by page. Each item can carry one or more "prov"
# (provenance) entries: which page it came from and where on that page.
items_by_page = {}
for i, item in enumerate(doc["texts"]):
    for prov in item["prov"]:
        items_by_page.setdefault(prov["page_no"], []).append((i, item, prov))

sections = []
for page_no in sorted(items_by_page):
    page_height = doc["pages"][str(page_no)]["size"]["height"]
    img = pdf[page_no - 1].render(scale=SCALE).to_pil().convert("RGB")
    draw = ImageDraw.Draw(img)
    rows = []
    for i, item, prov in items_by_page[page_no]:
        b = prov["bbox"]
        # PDF coordinates start at the BOTTOM-left; images start at the TOP-left,
        # so we flip the y axis: image_y = page_height - pdf_y.
        box = (b["l"] * SCALE, (page_height - b["t"]) * SCALE,
               b["r"] * SCALE, (page_height - b["b"]) * SCALE)
        color = COLORS.get(item["label"], "#7f7f7f")
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 2), str(i), fill=color)
        rows.append(f'<tr><td>{i}</td><td style="color:{color}"><b>{item["label"]}</b></td>'
                    f'<td>{html.escape(item["text"][:200])}</td></tr>')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = base64.b64encode(buf.getvalue()).decode()
    sections.append(f'<h2>Page {page_no}</h2><div class="row">'
                    f'<img src="data:image/png;base64,{png}">'
                    f'<table><tr><th>#</th><th>label</th><th>text</th></tr>{"".join(rows)}</table></div>')

out = json_path.with_suffix(".boxes.html")
out.write_text(f"""<!doctype html><meta charset="utf-8"><title>Docling view: {html.escape(json_path.stem)}</title>
<style>body{{font:14px system-ui;margin:16px}} .row{{display:flex;gap:16px;align-items:flex-start}}
img{{width:55%;border:1px solid #ccc}} table{{border-collapse:collapse}} td,th{{border-bottom:1px solid #eee;padding:4px;vertical-align:top;text-align:left}}</style>
<h1>{html.escape(json_path.stem)}</h1>
<p>Colored boxes = what Docling found. Red = heading, blue = text, green = list item, orange = table, purple = page header/footer.
Numbers on the page match the # column.</p>{''.join(sections)}""")
print(f"Wrote {out}")
