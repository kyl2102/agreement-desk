# Labeling guide (Step 7 gold set)

Label from the ORIGINAL PDF only. Do not look at data/legal.db or the extraction output first.

One row per contract per field. Columns:
- found: yes / no  (no = the contract does not contain this)
- value: in the format below (leave blank if found = no)
- page:  the PDF page where you found it
- notes: anything ambiguous ("two notice periods", "date only in signature block")

| field | format | example |
|---|---|---|
| parties | legal names, separated by ; | Netgear, Inc.; Ingram Micro Inc. |
| effective_date | YYYY-MM-DD | 1996-03-01 |
| governing_law | state or country name only | New York |
| termination_notice_days | whole number of days (month = 30, year = 365) | 90 |
| cap_type | fixed_amount / fees_based / exclusion_only / none | fees_based |
| cap_amount | number, no $ or commas (only if fixed_amount) | 1000000 |

Rules for hard cases (decide once, apply everywhere):
- effective_date: use the stated effective date; if none, the agreement/signing date at the top.
- termination_notice_days: notice to terminate for convenience OR to prevent renewal. Ignore cure periods for breach. If several, use the general one and note the others.
- cap_type: exclusion_only = only excludes damage TYPES (e.g. consequential) with no overall cap.
- If you are unsure, still pick an answer and write why in notes. Those notes are the most valuable part.
