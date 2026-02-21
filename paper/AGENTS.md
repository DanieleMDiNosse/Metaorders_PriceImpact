# Paper directory — Agent Guide (`paper/`)

## What this folder is
`paper/` contains the LaTeX manuscript and a **snapshot** of the figures used in the paper. The intended workflow is:
1) generate results into `out_files/` and `images/` via scripts, then  
2) curate/copy the exact final figures into `paper/images/` for a stable, versioned manuscript build.

Main files:
- `paper/main.tex` (paper source)
- `paper/references.bib` (bibliography)
- `paper/images/` (paper-stable figure snapshot)

---

## Build the PDF (avoid polluting repo root)
Run from repo root:
- `latexmk -cd -pdf paper/main.tex`

Why `-cd`: it ensures build artifacts stay under `paper/` (otherwise LaTeX tools may write `main.aux`, `main.log`, etc. in the repo root).

---

## Figure policy (snapshot is the source of truth for LaTeX)
### Two figure locations exist
- Generated pipeline outputs (not paper-stable):
  - `images/{DATASET_NAME}/...`
- Paper-stable snapshots (LaTeX should reference these):
  - `paper/images/...`

### Required sync rule for paper updates
When results are regenerated:
1. Identify which pipeline outputs correspond to each paper figure.
2. Copy only the final, publication-ready files into `paper/images/...`.
3. Commit `paper/images/...` alongside the manuscript changes.

This keeps the paper build reproducible even if upstream scripts are later refactored.

---

## Backend-to-paper traceability contract (paper-ready)
Every claim, number, and figure in `paper/main.tex` must be traceable to:
- a stored input table (`out_files/{DATASET_NAME}/...`) and/or a stored figure (`paper/images/...`), and
- the generating script + configuration + run provenance (git hash, command line, timestamp).

Minimum provenance to record per figure/table (store in your notes or a small manifest next to the artifact):
- Source script (e.g., `scripts/crowding_analysis.py`)
- Input dataset and paths (e.g., which `metaorders_info_sameday_filtered_*.parquet`)
- Config file(s) and the exact values for key knobs (metaorder definition, normalization mode)
- Seed and inference settings (bootstrap/permutation runs, clustering unit)

Do not manually edit generated images or hand-type summary statistics into LaTeX without a saved, reproducible table.

---

## Paper-grade statistical defensibility checklist
Before finalizing an updated paper section:
- Ensure inference descriptions match what code does (bootstrap unit, permutation logic, thresholds like `MIN_N`).
- Use dependence-aware uncertainty when reporting correlations (day clustering or another justified clustering).
- Report robustness checks for key empirical claims (metaorder filters, normalization modes, subsamples).
- Avoid TODO placeholders in the manuscript; either add uncertainty or clearly label results as descriptive.

