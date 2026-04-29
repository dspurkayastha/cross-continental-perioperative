"""Master orchestrator for Phase 1 + Phase 2 bootstrap-based analyses.

Runs every module that uses the unified paired-bootstrap engine and
assembles ``results/paper_numbers.csv``. Heavy Phase-2 analyses that
require model artefacts (SHAP) or retraining (class imbalance) live in
their own top-level scripts (``shap_analysis.py``,
``class_imbalance_sensitivity.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import bootstrap_auc
from . import bootstrap_differences
from . import delong
from . import direction_asymmetry_bootstrap
from . import paradox_gap
from . import permutation_direction
from . import race_stratified
from . import sex_stratified
from . import window_sensitivity
from ._bootstrap_utils import RESULTS_DIR, TABLES_DIR


def _fmt_auc_ci(auc: float, lo: float, hi: float) -> str:
    return f"{auc:.3f} (95% CI: {lo:.3f}–{hi:.3f})"


def _fmt_pp(d: float, lo: float, hi: float) -> str:
    return (
        f"{d * 100:+.2f}pp (95% CI: {lo * 100:+.2f}–{hi * 100:+.2f}pp)"
    )


def build_paper_numbers(
    delong_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    diffs_df: pd.DataFrame,
    paradox_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    dir_boot_df: pd.DataFrame,
    sex_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add(id_: str, value, formatted: str, source: str, context: str) -> None:
        rows.append({
            "id": id_, "value": value, "formatted": formatted,
            "source_table_or_file": source, "context": context,
        })

    # 8 external AUCs + CIs
    for _, r in bootstrap_df.iterrows():
        add(
            id_=f"auc_{r['model_name']}_on_{r['test_dataset']}",
            value=round(r["auc_point_raw"], 4),
            formatted=_fmt_auc_ci(r["auc_point_raw"], r["ci_lower_025_raw"], r["ci_upper_975_raw"]),
            source="bootstrap_auc_cis.csv",
            context=f"{r['model_name']} trained on {r['train_dataset']}, validated on {r['test_dataset']}",
        )

    # Within-direction intraop advantages
    for _, r in diffs_df[diffs_df["comparison"] == "intraop_minus_preop"].iterrows():
        add(
            id_=f"intraop_advantage_{r['model_intraop']}_vs_{r['model_preop']}",
            value=round(r["diff_point"], 4),
            formatted=_fmt_pp(r["diff_point"], r["ci_lower_025"], r["ci_upper_975"]) +
                      f"; bootstrap p={r['bootstrap_p_vs_zero']:.4f}",
            source="bootstrap_auc_differences.csv",
            context=f"AUC difference: {r['model_intraop']} − {r['model_preop']} on {r['direction']}",
        )
    cross = diffs_df[diffs_df["comparison"] == "mean_intraop_minus_mean_preop_all8"]
    if len(cross) == 1:
        r = cross.iloc[0]
        add(
            id_="mean_intraop_vs_preop_all8",
            value=round(r["diff_point"], 4),
            formatted=(
                f"{r['mean_auc_intraop']:.3f} vs {r['mean_auc_preop']:.3f} "
                f"(Δ = {r['diff_point']*100:+.2f}pp; 95% CI: "
                f"{r['ci_lower_025']*100:+.2f}–{r['ci_upper_975']*100:+.2f}pp; "
                f"bootstrap p={r['bootstrap_p_vs_zero']:.4f})"
            ),
            source="bootstrap_auc_differences.csv",
            context="Mean external AUC of 4 intraop (B) vs 4 preop (A) models",
        )

    # 8 paradox gaps + CIs
    for _, r in paradox_df.iterrows():
        add(
            id_=f"paradox_gap_{r['model_name']}_on_{r['test_dataset']}",
            value=round(r["paradox_gap"], 4) if pd.notna(r["paradox_gap"]) else None,
            formatted=(
                "N/A" if pd.isna(r["paradox_gap"]) else
                f"{r['paradox_gap']*100:+.1f}pp (95% CI: "
                f"{r['gap_ci_lower']*100:+.1f}–{r['gap_ci_upper']*100:+.1f}pp)"
            ),
            source="paradox_gaps.csv",
            context=f"{r['paradox_definition']}; {r['model_name']} on {r['test_dataset']}",
        )

    xmm = paradox_df[
        (paradox_df["model_name"] == "XGB-MOV-A") & (paradox_df["test_dataset"] == "INSPIRE")
    ].iloc[0]
    add("simpsons_overall_xgbmova", round(xmm["overall_auc"], 4),
        f"{xmm['overall_auc']:.3f}", "paradox_gaps.csv",
        "Overall external AUC for XGB-MOV-A on INSPIRE (Simpson's paradox row)")
    add("simpsons_asa_1_2_xgbmova", round(xmm["auc_asa_1_2"], 4),
        f"{xmm['auc_asa_1_2']:.3f}", "paradox_gaps.csv",
        "Within ASA_1_2 AUC for XGB-MOV-A on INSPIRE (Simpson's paradox row)")
    add("simpsons_asa_3_plus_xgbmova", round(xmm["auc_asa_3_plus"], 4),
        f"{xmm['auc_asa_3_plus']:.3f}", "paradox_gaps.csv",
        "Within ASA_3_plus AUC for XGB-MOV-A on INSPIRE (Simpson's paradox row)")

    # Direction asymmetry — bootstrap (PRIMARY) and permutation (SUPPORTING)
    primary = dir_boot_df[dir_boot_df["inferential_claim"]].iloc[0]
    mean_ins = dir_boot_df[dir_boot_df["statistic"] == "mean_degradation_inspire"].iloc[0]
    mean_mov = dir_boot_df[dir_boot_df["statistic"] == "mean_degradation_mover"].iloc[0]
    ratio = dir_boot_df[dir_boot_df["statistic"] == "ratio_mov_over_ins"].iloc[0]

    add("direction_asymmetry_diff_pp", round(primary["point"], 4),
        f"{primary['point']*100:+.2f}pp (95% CI: {primary['ci_lower']*100:+.2f}–"
        f"{primary['ci_upper']*100:+.2f}pp; bootstrap p="
        f"{primary['bootstrap_p_vs_zero']:.4f})",
        "direction_asymmetry_bootstrap.csv",
        "PRIMARY inferential claim: mean degradation MOVER − INSPIRE, paired case bootstrap")
    add("direction_asymmetry_ratio", round(ratio["point"], 3),
        f"{ratio['point']:.2f}× (95% CI: {ratio['ci_lower']:.2f}–{ratio['ci_upper']:.2f})",
        "direction_asymmetry_bootstrap.csv",
        "Ratio mean(MOVER-degradation) / mean(INSPIRE-degradation), paired bootstrap")
    add("mean_degradation_inspire_trained", round(mean_ins["point"], 4),
        f"{mean_ins['point']*100:.1f}% (95% CI: {mean_ins['ci_lower']*100:.1f}–"
        f"{mean_ins['ci_upper']*100:.1f}%)",
        "direction_asymmetry_bootstrap.csv",
        "Mean external-AUC degradation across 4 INSPIRE-trained models, bootstrapped")
    add("mean_degradation_mover_trained", round(mean_mov["point"], 4),
        f"{mean_mov['point']*100:.1f}% (95% CI: {mean_mov['ci_lower']*100:.1f}–"
        f"{mean_mov['ci_upper']*100:.1f}%)",
        "direction_asymmetry_bootstrap.csv",
        "Mean external-AUC degradation across 4 MOVER-trained models, bootstrapped")

    pr = perm_df.iloc[0]
    add("direction_asymmetry_p_permutation_exact",
        float(pr["exact_p_value"]),
        f"p = {pr['exact_p_value']:.3f} (exact 4-vs-4 permutation, 70 splits)",
        "direction_asymmetry_permutation.csv",
        "SUPPORTING: exact permutation test; min achievable p is 1/70 ≈ 0.014, "
        "so p<0.001 is unreachable with this sample size")

    # DeLong significant count
    sig = delong_df[delong_df["significant_at_0.05"]]
    add("delong_n_significant_bh", int(len(sig)),
        f"{len(sig)} of {len(delong_df)} pairwise DeLong tests significant after BH",
        "delong_comparisons.csv",
        "Count of significant pairwise AUC comparisons after BH correction")

    # Sex-stratified AUCs
    for _, r in sex_df.iterrows():
        add(
            id_=f"auc_{r['model_name']}_on_{r['test_dataset']}_sex_{r['sex']}",
            value=round(r["auc"], 4) if pd.notna(r["auc"]) else None,
            formatted=(
                "N/A" if pd.isna(r["auc"]) else
                f"{r['auc']:.3f} (95% CI: {r['ci_lower']:.3f}–{r['ci_upper']:.3f}); "
                f"n={r['n_cases']:,}, events={r['n_events']}"
            ),
            source="sex_stratified_auc.csv",
            context=f"{r['model_name']} on {r['test_dataset']}, sex={r['sex']}",
        )

    return pd.DataFrame(rows)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 1 + PHASE 2 (non-retraining) STATISTICAL ANALYSES")
    print("=" * 72)

    print("\n[1/9] DeLong pairwise tests (12 pairs, BH-corrected) ...")
    delong_df = delong.run()

    print("\n[2/9] Bootstrap 95% CIs for each external AUC (2,000 iter) ...")
    auc_df = bootstrap_auc.run()

    print("\n[3/9] Paired-bootstrap AUC differences (2,000 iter) ...")
    diffs_df = bootstrap_differences.run()

    print("\n[4/9] Paradox-gap CIs (2,000 iter) ...")
    paradox_df = paradox_gap.run()

    print("\n[5/9] Direction-asymmetry permutation test (SUPPORTING) ...")
    perm_df = permutation_direction.run()

    print("\n[6/9] Direction-asymmetry bootstrap (PRIMARY, 2,000 iter) ...")
    dir_boot_df = direction_asymmetry_bootstrap.run()

    print("\n[7/9] Sex-stratified AUC (2,000 iter) ...")
    sex_df = sex_stratified.run()

    print("\n[8/9] Race-stratified AUC — BLOCKED stub ...")
    race_stratified.run()

    print("\n[9/9] 60-min window sensitivity — BLOCKED stub ...")
    window_sensitivity.run()

    print("\n--- Assembling paper_numbers.csv ---")
    numbers = build_paper_numbers(
        delong_df, auc_df, diffs_df, paradox_df, perm_df, dir_boot_df, sex_df,
    )
    numbers_path = RESULTS_DIR / "paper_numbers.csv"
    numbers.to_csv(numbers_path, index=False)
    print(f"  wrote {numbers_path}  ({len(numbers)} rows)")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
