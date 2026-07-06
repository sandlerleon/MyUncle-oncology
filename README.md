# MyUncle — bistable order–repair engine for evolutionary cancer-therapy modelling

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)  <!-- added on Zenodo publish -->

A single-file Python framework that couples an evolutionary (sensitive/resistant) tumour model to a **bistable order–repair variable** representing multicellular information integrity. It reproduces, in one dynamical system, the atavism theory's therapeutic corollary — *target the weakness, not the strength* — and compares six therapy protocols across a virtual mouse cohort.

## Why it's useful
- One abstraction — *a directionally ordered medium eroded by a state-dependent flux and restored by repair* — instantiated for cancer as the `tissue_atavism` preset.
- Built-in therapy protocols: MTD, immunotherapy, adaptive dosing, SIM-block, re-differentiation, and combinations.
- Global (Latin-hypercube) sensitivity analysis that shows the qualitative conclusions are structural, not tuned.
- A phylostratigraphy pipeline (correlation-robust permutation test) ready for real TCGA/GTEx matrices — the empirical validation hook.
- `reproduce_paper()` regenerates the full figure suite, cover, and manuscript.

## Companion manuscript
This code is the reproducibility archive for the manuscript *"Cancer as a bistable order–repair transition: an in-silico comparison of evolutionary and atavism-targeting therapies"* (submitted to **Bulletin of Mathematical Biology**). The Zenodo archive DOI of this repository is cited in the manuscript's Code & Data Availability section.

## Invoking it via Claude
Ask Claude to "use MyUncle to reproduce the paper" or run directly:
```
python MyUncle.py --reproduce --preset tissue_atavism --full
python MyUncle.py --phylo-demo
```

## Requirements
Python 3.11+, numpy, scipy, matplotlib, python-docx. See `requirements.txt`.

## Tests
`pytest` runs a smoke suite (bistable engine, protocol ordering, phylostratigraphy).

## License
MIT © 2026 Leon Sandler
