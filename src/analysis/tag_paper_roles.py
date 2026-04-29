"""Add a ``paper_role`` column to ``results/paper_numbers.csv``.

Values
------
* ``headline``         — max 4 rows, fit in abstract + Principal Findings.
* ``primary``          — cited in Results body.
* ``supporting``       — supplementary, parenthetical, or table-only.
* ``do_not_cite``      — reproducible but unreliable (wide CI, tiny stratum).
* ``limitations_note`` — blocked-analysis stubs that inform a Limitation.

Also appends one ``do_not_cite`` row for the LR-INS-B ASA_1_2 = 0.858 value
so the gatekeeper is explicit in this file, even though that value is not
currently cited anywhere.
"""

from __future__ import annotations

import sys

import pandas as pd

from ._bootstrap_utils import RESULTS_DIR


HEADLINE_IDS = {
    "direction_asymmetry_diff_pp":        "Headline 1: 2.6-fold direction asymmetry (case-level bootstrap, +8.53pp)",
    "paradox_gap_XGB-MOV-A_on_INSPIRE":   "Headline 2: canonical Simpson's paradox (+16.5pp)",
    "shap_spearman_rho_XGB-INS-B":        "Headline 3: SHAP feature-importance transfer (ρ=0.97)",
    "mean_intraop_vs_preop_all8":         "Headline 4: mean intraop advantage across 8 models (+3.60pp)",
}

LIMITATIONS_IDS = {"race_stratified_analysis", "window_sensitivity_analysis"}

DO_NOT_CITE_IDS: set[str] = set()  # populated via explicit row below


def classify(row: pd.Series) -> tuple[str, str]:
    id_ = row["id"]
    if id_ in HEADLINE_IDS:
        return "headline", HEADLINE_IDS[id_]
    if id_ in LIMITATIONS_IDS:
        return "limitations_note", "Blocked analysis — informs Limitations paragraph"
    if id_ in DO_NOT_CITE_IDS:
        return "do_not_cite", "Reproducible but statistically unreliable"

    # Primary: 8 external AUCs, 4 within-direction intraop advantages, 8
    # paradox gaps, Simpson's paradox breakdown, both SHAP ρ, SHAP top-10
    # feature lists, 4 direction-asymmetry supporting numbers that are
    # quoted in text (means, ratio, p).
    if id_.startswith("auc_") and "_sex_" not in id_:
        return "primary", "Core discrimination result — Table 2"
    if id_.startswith("intraop_advantage_"):
        return "primary", "Within-direction intraop-vs-preop comparison — Results §External"
    if id_.startswith("paradox_gap_"):
        return "primary", "Paradox-gap row — Table 3"
    if id_.startswith("simpsons_"):
        return "primary", "Simpson's paradox decomposition — Results §Paradox"
    if id_ == "direction_asymmetry_ratio":
        return "primary", "Ratio form of the headline direction-asymmetry number"
    if id_ in ("mean_degradation_inspire_trained",
               "mean_degradation_mover_trained"):
        return "primary", "Group degradation means feeding the headline asymmetry"
    if id_.startswith("shap_spearman_rho_"):
        return "primary", "SHAP importance rank correlation — Results §Transferability"
    if id_.startswith("shap_top_transferable_"):
        return "primary", "Top-10 SHAP transferable features — Table S? / text"

    # Supporting: sex-stratified AUCs, permutation p, DeLong summary,
    # class-imbalance configs
    if "_sex_" in id_:
        return "supporting", "Sex-stratified AUC — Supplementary Table"
    if id_ == "direction_asymmetry_p_permutation_exact":
        return "supporting", "Permutation p-value retained as supporting evidence"
    if id_ == "delong_n_significant_bh":
        return "supporting", "DeLong family-wise summary — Methods / Supplementary"
    if id_.startswith("class_imbalance_"):
        return "supporting", "Class-imbalance sensitivity — Supplementary"

    return "supporting", "Uncategorised"


def main() -> int:
    path = RESULTS_DIR / "paper_numbers.csv"
    pn = pd.read_csv(path)

    roles = pn.apply(classify, axis=1, result_type="expand")
    roles.columns = ["paper_role", "paper_role_note"]
    pn = pd.concat([pn, roles], axis=1)

    # Append the LR-INS-B ASA_1_2 = 0.858 gatekeeper row.
    do_not_cite_row = {
        "id": "auc_LR-INS-B_asa_1_2_on_MOVER_UNRELIABLE",
        "value": 0.8584,
        "formatted": "0.858 (95% CI: 0.737–0.972); n=21,028, events=9",
        "source_table_or_file": "paradox_gaps.csv",
        "context": ("LR-INS-B AUC within the MOVER ASA_1_2 stratum; "
                    "only 9 events — CI spans 23.5pp, effectively noise."),
        "paper_role": "do_not_cite",
        "paper_role_note": ("Reproducible but statistically unreliable due "
                            "to small events count; unreliable at n_events < 10; reported for completeness only."),
    }
    pn = pd.concat([pn, pd.DataFrame([do_not_cite_row])], ignore_index=True)

    # Stable column order
    col_order = ["id", "value", "formatted", "paper_role",
                 "paper_role_note", "source_table_or_file", "context"]
    pn = pn[col_order]
    pn.to_csv(path, index=False)

    print(f"  wrote {path}  ({len(pn)} rows)")
    counts = pn["paper_role"].value_counts()
    print("\n  role distribution:")
    for role, n in counts.items():
        print(f"    {role:<18} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
