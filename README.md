# Cross-Continental Perioperative Mortality Prediction

Reproducibility repository for:

> **When external validation isn't enough: Simpson's paradox, direction asymmetry, and calibration collapse in cross-continental perioperative mortality prediction.**
> Debaraj Shome Purkayastha, MS. *Journal of the American Medical Informatics Association* (under review). Preprint: medRxiv DOI [10.64898/2025.12.28.25343118](https://doi.org/10.64898/2025.12.28.25343118).

This repository contains the analysis code, figure-generation scripts, trained model objects, aggregate analysis outputs, and LaTeX manuscript sources. It is the citable reproducibility artifact deposited at Zenodo alongside the JAMIA submission.

---

## What this study reports

Eight clinical prediction models for in-hospital surgical mortality (XGBoost and logistic regression × preoperative-only and preoperative-plus-intraoperative feature sets × INSPIRE [Korea, n = 127,413] and MOVER [USA, n = 57,545] training cohorts) were bidirectionally externally validated. Three findings:

1. **Simpson's paradox in preoperative models.** Aggregate AUC of 0.756 (95% CI: 0.741–0.771) for the worst-affected model masks within-stratum AUCs near 0.59, a paradox gap of +16.5 percentage points (95% CI: +15.1 to +18.0 pp).
2. **Direction-asymmetric transfer.** Case-level paired-bootstrap-quantified asymmetry of +8.53 pp (95% CI: +6.91 to +10.24 pp; bootstrap p = 0.001) between training populations, surviving multi-dimensional case-mix matching at approximately 70% of the unmatched baseline.
3. **Discrimination transfers, calibration does not.** SHAP feature-importance ranks transfer near-perfectly across continents (Spearman ρ ≥ 0.93 for the best models), but probability calibration requires Platt-scaling recalibration uniformly across all eight cross-population validation runs.

Three TRIPOD+AI-aligned methodological commitments are demonstrated jointly: stratified within-cohort analysis, bidirectional case-mix-matched testing, and case-level paired bootstrap inference.

---

## Repository layout

```
cross-continental-perioperative/
├── README.md                        ← this file
├── BUILD.md                         ← LaTeX compile sequence + verification
├── LICENSE                          ← MIT
├── CITATION.cff                     ← Zenodo + paper citation metadata
├── manuscript/
│   ├── jamia_abstract.tex           ← structured abstract
│   ├── jamia_intro_results.tex      ← background, significance, results
│   ├── jamia_methods.tex            ← materials and methods
│   ├── jamia_discussion.tex         ← discussion + end-matter declarations
│   ├── jamia_supp.tex               ← supplementary material
│   ├── references.bib
│   ├── figures/                     ← 10 figure PDFs (4 main + 6 supplementary)
│   └── build/                       ← canonical + xr-integrated build wrappers
├── results/
│   ├── jamia_main.pdf               ← compiled main manuscript
│   ├── jamia_supp.pdf               ← compiled supplementary
│   ├── paper_numbers.tex            ← \newcommand macros for quoted numbers
│   ├── paper_numbers.csv            ← source CSV for the macros
│   ├── tables/                      ← 28 aggregate analysis CSVs (paradox gaps,
│   │                                  bootstrap CIs, DeLong comparisons, SHAP
│   │                                  rank correlations, calibration, DCA, etc.)
│   └── figures/                     ← rendered SHAP summary figures (vector)
├── artifacts/
│   └── models/                      ← 8 trained model directories per the
│                                      paper's 2×2×2 factorial; each contains
│                                      final_model.{json,pkl} + feature names +
│                                      hyperparameters + aggregate fold metrics
└── src/                             ← Python analysis pipeline
    ├── analysis/                    ← bootstrap, DeLong, calibration, SHAP, DCA
    ├── figures/                     ← figure-generation scripts
    ├── preprocessing/               ← parquet builder, paper_numbers generator
    └── validation/                  ← verification scripts
```

---

## Source data — credentialed access

The raw datasets are not redistributed in this repository. Per Methods §3.6 of the manuscript, the data-use agreements of both INSPIRE and MOVER are interpreted conservatively as prohibiting redistribution of patient-level data, including row-level patient-level derivatives such as per-case prediction outputs.

**INSPIRE:**
- Source: [PhysioNet INSPIRE v1.3](https://physionet.org/content/inspire/1.3/) (DOI [10.13026/46m4-f655](https://doi.org/10.13026/46m4-f655)).
- Institution: Seoul National University Hospital, Korea, 2011–2020.
- Access: PhysioNet credentialed-access agreement.

**MOVER:**
- Source: [UCI Machine Learning Repository, dataset 877](https://archive.ics.uci.edu/dataset/877/mover:+medical+informatics+operating+room+vitals+and+events+repository) (DOI [10.24432/C5VS5G](https://doi.org/10.24432/C5VS5G)).
- Institution: UC Irvine Medical Center, USA, 2015–2022.
- Access: UCI Machine Learning Repository credentialed-access agreement.

To reproduce: complete the credentialed-access agreement at each repository, download the source data to local paths, and run the analysis pipeline described in `BUILD.md`. The released models in `artifacts/models/` can be applied directly to the source data without retraining.

---

## What is included vs. excluded

**Included (open release):**
- Trained model objects (8 models — XGBoost JSONs and logistic-regression pickles), feature names, categorical encoders, hyperparameters, aggregate per-fold metrics, training summaries.
- Aggregate analysis outputs (paradox gaps, bootstrap AUC CIs, DeLong comparisons, calibration metrics, decision-curve net benefit, SHAP rank correlations, sex-stratified AUCs, race-stratified AUCs for MOVER).
- Analysis code and figure-generation scripts.
- LaTeX manuscript sources.
- Compiled main and supplementary PDFs.

**Excluded (data-use agreement constraint):**
- Raw INSPIRE and MOVER source data.
- Per-case prediction outputs (`*_predictions.parquet`).
- Per-case SHAP values (`*.npy`).
- Out-of-fold prediction CSVs (`oof_predictions.csv`).

A credentialed researcher reproducing this work re-credentials on PhysioNet and the UCI Machine Learning Repository, downloads source data, and either re-runs the pipeline or applies the released models — both options are supported.

---

## Reproduction quick-start

```bash
# 1. Clone the repo.
git clone https://github.com/dspurkayastha/cross-continental-perioperative.git
cd cross-continental-perioperative

# 2. Set up Python environment.
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn xgboost shap pyarrow scipy matplotlib

# 3. Acquire source data (credentialed access — see "Source data" above) and
#    point CCPERIOP_DATA_ROOT at the directory containing the downloaded
#    INSPIRE and MOVER source data; see BUILD.md for the expected layout.
export CCPERIOP_DATA_ROOT=/path/to/credentialed-data

# 4. Apply released models to source data.
python -m src.analysis.bootstrap_apply_models  # uses artifacts/models/*

# 5. (Optional) Recompute aggregate analysis outputs.
python -m src.analysis.bootstrap_auc_cis
python -m src.analysis.delong_comparisons

# 6. Rebuild the manuscript PDFs (see BUILD.md for the full sequence).
```

Random seeds are fixed throughout the pipeline (`SEED = 42`); cross-validation folds are reproducible via `StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)`.

---

## Citation

If you use this code, the released models, or the analysis outputs, please cite both the JAMIA paper (preferred citation) and the Zenodo deposit (software citation). See `CITATION.cff` for machine-readable metadata.

```bibtex
@article{ShomePurkayastha2026,
  author    = {Shome Purkayastha, Debaraj},
  title     = {When external validation isn't enough: {Simpson's} paradox,
               direction asymmetry, and calibration collapse in cross-continental
               perioperative mortality prediction},
  journal   = {Journal of the American Medical Informatics Association},
  year      = {2026},
  note      = {Under review at submission of this software release.}
}
```

The Zenodo software DOI is shown on the GitHub release page after the v1.0.0 tag is pushed.

---

## License

MIT (see `LICENSE`). The medRxiv preprint is separately under CC-BY-NC-ND 4.0 per medRxiv's default; that license applies to the preprint PDF only and does not extend to this code or model release.

---

## Contact

Debaraj Shome Purkayastha, MS
Department of Oncosurgery, Silchar Cancer Centre, ACCF
Silchar, Assam, India
Email: debaraj.purkayastha@accf.in
ORCID: [0009-0001-9641-4384](https://orcid.org/0009-0001-9641-4384)
