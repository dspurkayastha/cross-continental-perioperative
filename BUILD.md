# Building the manuscript and supplementary PDFs

The submission consists of two PDFs: `jamia_main.pdf` and `jamia_supp.pdf`. Each is a separate compile target; they share a label namespace via the `xr` package, so cross-references between main and supplementary resolve in both directions when the `.aux` files are imported by the partner document.

## Dependencies

- TeX Live 2025 (or compatible): `pdflatex`, `bibtex`.
- Standard LaTeX packages used by the sources: `geometry`, `hyperref`, `xcolor`, `graphicx`, `booktabs`, `natbib`, `xr`, `rotating`, `caption`, `float`, `algorithm`, `algpseudocode`, `array`, `parskip`.

## Source layout

```
manuscript/
├── jamia_abstract.tex               # 250-word structured abstract
├── jamia_intro_results.tex          # Background and Significance + Results
├── jamia_methods.tex                # Materials and Methods
├── jamia_discussion.tex             # Discussion + Acknowledgments / COI /
│                                    # Funding / AI Use Disclosure /
│                                    # Data Availability / Ethics
├── jamia_supp.tex                   # Supplementary §S1–§S8
├── references.bib                   # Bibliography (Vancouver/numeric via plainnat)
├── figures/                         # 10 vector PDFs (4 main + 6 supplementary)
└── build/
    ├── jamia_canonical_compile_check.tex            # canonical-only build wrapper
    └── jamia_canonical_compile_check_with_supp.tex  # xr-integrated submission target
results/paper_numbers.tex            # \newcommand macros for every quoted number
                                     # (input by the wrappers via relative path)
```

## Compile sequence

The supplementary and the main wrapper cross-reference each other via the `xr` package, so they must be compiled in the order below. The canonical-only build wrapper is compiled first to produce the `.aux` file that the supplementary's `\externaldocument{build/jamia_canonical_compile_check}` reads to resolve supp→main references.

```bash
# 1. Compile the canonical-only build wrapper (produces the .aux that the
#    supplementary imports for supp→main cross-references).
cd manuscript/build/
pdflatex -interaction=nonstopmode jamia_canonical_compile_check.tex
bibtex   jamia_canonical_compile_check
pdflatex -interaction=nonstopmode jamia_canonical_compile_check.tex
pdflatex -interaction=nonstopmode jamia_canonical_compile_check.tex

# 2. Compile the supplementary (resolves supp→main references via xr).
cd ../
pdflatex -interaction=nonstopmode jamia_supp.tex
bibtex   jamia_supp
pdflatex -interaction=nonstopmode jamia_supp.tex
pdflatex -interaction=nonstopmode jamia_supp.tex

# 3. Compile the main wrapper (xr-integrated; this is the submission PDF and
#    resolves main→supp references via xr).
cd build/
pdflatex -interaction=nonstopmode jamia_canonical_compile_check_with_supp.tex
bibtex   jamia_canonical_compile_check_with_supp
pdflatex -interaction=nonstopmode jamia_canonical_compile_check_with_supp.tex
pdflatex -interaction=nonstopmode jamia_canonical_compile_check_with_supp.tex

# 4. Final supplementary passes so any supp→main references newly resolved
#    by step 3 settle.
cd ../
pdflatex -interaction=nonstopmode jamia_supp.tex
pdflatex -interaction=nonstopmode jamia_supp.tex
```

Skipping the `bibtex` pass causes citations to render as `[?]` (in supp) or as the literal `[ref:KEY]` placeholder text (in the main wrapper). If you see those in the rendered PDF, re-run the full sequence above.

## Verifying the build

After compilation:

- `manuscript/build/jamia_canonical_compile_check_with_supp.pdf` should be ~20 pages with no `??`, `[ref:`, or `[?` placeholders in the body text.
- `manuscript/jamia_supp.pdf` should be ~25 pages with the same cleanliness.

Both PDFs are also published in `results/jamia_main.pdf` (= the with-supp wrapper) and `results/jamia_supp.pdf` for direct reading.

## Macros and quoted numbers

Every quoted number in the manuscript (AUCs, p-values, sample sizes, percentages) is referenced via a `\pn*` macro defined in `results/paper_numbers.tex`. The macros are auto-generated from `results/paper_numbers.csv` by `src/preprocessing/generate_paper_numbers_tex.py`. To regenerate after any analysis change:

```bash
python -m src.preprocessing.generate_paper_numbers_tex
```

This guarantees that text numbers, table numbers, and figure annotations stay synchronized with the underlying CSV during revision cycles.

## Reproducing the analysis numbers

The analysis pipeline is under `src/`. Source data (INSPIRE from PhysioNet, MOVER from UCI ML Repository) require credentialed access (see `README.md`). Set the `CCPERIOP_DATA_ROOT` environment variable to point at the directory containing the credentialed-access source data; the pipeline expects the following layout under that root:

```
$CCPERIOP_DATA_ROOT/
├── inspire-…/                       # INSPIRE v1.3 PhysioNet release
├── mover/EPIC_EMR/EMR/              # MOVER UCI release (extracted CSVs)
└── derived/
    ├── phase1/                      # cohort + feature outputs
    ├── phase2/models/               # model training summary
    └── phase3/                      # external-validation outputs
```

The trained model objects under `artifacts/models/` are released so model retraining is optional for replication. `results/tables/` and `results/figures/` are the aggregate outputs cited by the manuscript; per-case predictions are not redistributed under the data-use-agreement constraints.
