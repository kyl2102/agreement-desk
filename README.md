# Legal Doc Tool

A learning project: a small legal document tool where AI reads and databases count.

## Data and attribution

This repository includes contracts and content derived from them (parsed text, extracted fields,
quotes, page images) from **CUAD v1**, the Contract Understanding Atticus Dataset, by The Atticus Project.

- Source: https://www.atticusprojectai.org/cuad (downloaded from https://huggingface.co/datasets/theatticusproject/cuad)
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Paper: Hendrycks, Burns, Chen and Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review",
  NeurIPS 2021 Datasets and Benchmarks Track. https://arxiv.org/abs/2103.06268

Changes made: a 40-contract sample was selected; contracts were parsed with Docling, grouped into agreement
families, and run through local-model field extraction. The gold labels in `data/labels/` are this project's
own hand labels, not CUAD's annotations.
