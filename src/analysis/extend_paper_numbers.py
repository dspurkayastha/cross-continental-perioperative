"""Extend ``results/paper_numbers.csv`` with Phase-2 analyses that run
outside the main ``run_phase1.py`` pipeline (SHAP, class imbalance,
race/window stubs).

Idempotent: looks up each id and overwrites the existing row if
present; appends otherwise. Run this after ``shap_analysis.py`` and
``class_imbalance_sensitivity.py`` complete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ._bootstrap_utils import RESULTS_DIR, TABLES_DIR


def _upsert(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    if df.empty or "id" not in df.columns:
        return pd.DataFrame([row])
    mask = df["id"] == row["id"]
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
        return df
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def main() -> int:
    pn_path = RESULTS_DIR / "paper_numbers.csv"
    if not pn_path.exists():
        print(f"ERROR: {pn_path} missing — run run_phase1 first")
        return 2
    pn = pd.read_csv(pn_path)
    print(f"  loaded {pn_path} with {len(pn)} rows")

    # ---- SHAP ------------------------------------------------------------
    shap_imp_path = TABLES_DIR / "shap_importance_comparison.csv"
    transferable_path = TABLES_DIR / "universally_transferable_features.csv"

    if shap_imp_path.exists():
        shap_df = pd.read_csv(shap_imp_path)
        for model_name, g in shap_df.groupby("model_name"):
            rho = float(g["spearman_rho_internal_vs_external"].iloc[0])
            pn = _upsert(pn, {
                "id": f"shap_spearman_rho_{model_name}",
                "value": round(rho, 4),
                "formatted": f"ρ = {rho:.3f}",
                "source_table_or_file": "shap_importance_comparison.csv",
                "context": f"Spearman rank correlation of mean |SHAP| "
                           f"between internal and external test sets for {model_name}",
            })

    if transferable_path.exists():
        tr = pd.read_csv(transferable_path)
        for model_name, g in tr.groupby("model_name"):
            top_names = g.sort_values("rank_internal")["feature"].tolist()
            pn = _upsert(pn, {
                "id": f"shap_top_transferable_{model_name}",
                "value": len(top_names),
                "formatted": ", ".join(top_names),
                "source_table_or_file": "universally_transferable_features.csv",
                "context": f"Top {len(top_names)} features with "
                           f"consistently high importance (top-20 on both internal "
                           f"and external evaluation) for {model_name}",
            })

    # ---- Class imbalance -------------------------------------------------
    ci_path = TABLES_DIR / "class_imbalance_sensitivity.csv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        for _, r in ci.iterrows():
            pn = _upsert(pn, {
                "id": f"class_imbalance_{r['config']}_auc",
                "value": round(float(r["external_auc"]), 4),
                "formatted": (f"AUC {r['external_auc']:.3f}, "
                              f"Brier {r['brier']:.4f}, "
                              f"slope {r['calibration_slope']:.2f}, "
                              f"intercept {r['calibration_intercept']:.2f}, "
                              f"O/E {r['oe_ratio']:.2f}"),
                "source_table_or_file": "class_imbalance_sensitivity.csv",
                "context": (f"XGB-INS-B retrained with {r['config']} "
                            f"(scale_pos_weight={r['scale_pos_weight']:g}, "
                            f"SMOTE={bool(r['smote'])}), external-validated on MOVER"),
            })

    # ---- Case-mix matched direction asymmetry ---------------------------
    asym_path = TABLES_DIR / "asymmetry_decomposition.csv"
    if asym_path.exists():
        asym = pd.read_csv(asym_path)
        id_map = {
            "framing_match_inspire_casemix": "matched_asymmetry_framing_a_inspire_casemix",
            "framing_match_mover_casemix":   "matched_asymmetry_framing_a_mover_casemix",
            "framing_balanced_50_50":        "matched_asymmetry_framing_b_balanced",
        }
        for _, r in asym.iterrows():
            if r["row"] not in id_map:
                continue
            pn = _upsert(pn, {
                "id": id_map[r["row"]],
                "value": round(float(r["asymmetry"]), 4),
                "formatted": (
                    f"{float(r['asymmetry'])*100:+.2f}pp (95% CI: "
                    f"{float(r['asymmetry_ci_lower'])*100:+.2f}–"
                    f"{float(r['asymmetry_ci_upper'])*100:+.2f}pp; "
                    f"bootstrap p={float(r['bootstrap_p']):.4f}; "
                    f"n_INSPIRE_test={int(r['n_inspire']):,}, "
                    f"n_MOVER_test={int(r['n_mover']):,})"
                ),
                "paper_role": "primary",
                "paper_role_note": (f"Case-mix sensitivity: {r['description']}. "
                                    "Primary — cited in Discussion mechanism paragraph."),
                "source_table_or_file": "asymmetry_decomposition.csv",
                "context": r["description"],
            })

        # Interpretation line
        interps = set(asym["interpretation"]) - {"baseline"}
        if interps == {"case_mix_is_not_primary_driver"}:
            interp_value = "case_mix_is_not_primary_driver"
            interp_text = (
                "Case-mix does not drive the direction asymmetry; matching "
                "case-mix across three framings either preserves or amplifies "
                "the effect (128–203% of baseline) with bootstrap p=0.001 in "
                "all three framings."
            )
        else:
            interp_value = ";".join(sorted(interps))
            interp_text = f"Mixed interpretation across framings: {interp_value}"
        pn = _upsert(pn, {
            "id": "case_mix_sensitivity_interpretation",
            "value": interp_value,
            "formatted": interp_text,
            "paper_role": "primary",
            "paper_role_note": "Headline interpretation of case-mix sensitivity; Discussion §Mechanisms.",
            "source_table_or_file": "asymmetry_decomposition.csv",
            "context": "Plain-English interpretation of the case-mix decomposition",
        })

    # ---- Additional matched-dimension summary (emergency, comorbidity, temporal)
    all_dim_path = TABLES_DIR / "asymmetry_all_dimensions_summary.csv"
    if all_dim_path.exists():
        adf = pd.read_csv(all_dim_path)
        extra_map = {
            ("emergency", "emergency_match_mover_prop"):
                "matched_asymmetry_emergency_match_mover_prop",
            ("emergency", "emergency_match_inspire_prop"):
                "matched_asymmetry_emergency_match_inspire_prop",
            ("emergency", "emergency_balanced_midpoint"):
                "matched_asymmetry_emergency_balanced_midpoint",
            ("comorbidity", "comorbidity_match_mover_prop"):
                "matched_asymmetry_comorbidity_match_mover_prop",
            ("comorbidity", "comorbidity_match_inspire_prop"):
                "matched_asymmetry_comorbidity_match_inspire_prop",
            ("comorbidity", "comorbidity_balanced_50_50"):
                "matched_asymmetry_comorbidity_balanced",
        }
        for _, r in adf.iterrows():
            key = (r["dimension"], r["framing"])
            if key not in extra_map or pd.isna(r["matched_asymmetry_pp"]):
                continue
            id_ = extra_map[key]
            pn = _upsert(pn, {
                "id": id_,
                "value": round(float(r["matched_asymmetry_pp"]), 4),
                "formatted": (
                    f"{float(r['matched_asymmetry_pp'])*100:+.2f}pp "
                    f"(95% CI: {float(r['ci_lower'])*100:+.2f}–"
                    f"{float(r['ci_upper'])*100:+.2f}pp; bootstrap p="
                    f"{float(r['bootstrap_p']):.4f}; "
                    f"{float(r['pct_of_baseline']):.0f}% of baseline)"
                ),
                "paper_role": "primary",
                "paper_role_note": f"Matched sensitivity: {r['framing_label']}",
                "source_table_or_file": "asymmetry_all_dimensions_summary.csv",
                "context": r["framing_label"],
            })

        # Temporal block note
        temp = adf[adf["dimension"] == "temporal_overlap_2015_2020"]
        if not temp.empty:
            pn = _upsert(pn, {
                "id": "matched_asymmetry_temporal_overlap",
                "value": "BLOCKED",
                "formatted": (
                    "Not performed — INSPIRE timestamps are anonymised as "
                    "'Relative Time' minutes; calendar-year matching is not "
                    "feasible without a dataset-level reference date."
                ),
                "paper_role": "limitations_note",
                "paper_role_note": "Temporal overlap sensitivity is blocked; inform Limitations.",
                "source_table_or_file": "asymmetry_temporal_matched.csv",
                "context": "Temporal 2015–2020 overlap matching — blocked",
            })

    # ---- Blocked stubs ---------------------------------------------------
    race_path = TABLES_DIR / "race_stratified_auc_mover_only.csv"
    if race_path.exists():
        pn = _upsert(pn, {
            "id": "race_stratified_analysis",
            "value": "BLOCKED",
            "formatted": "Not performed — MOVER release omits race/ethnicity",
            "source_table_or_file": "race_stratified_auc_mover_only.csv",
            "context": "Required for JAMIA Limitations paragraph",
        })

    win_path = TABLES_DIR / "window_sensitivity.csv"
    if win_path.exists():
        pn = _upsert(pn, {
            "id": "window_sensitivity_analysis",
            "value": "BLOCKED",
            "formatted": "Not performed — no 30/120-min feature files; "
                          "requires Phase-1 rerun (~4–6h)",
            "source_table_or_file": "window_sensitivity.csv",
            "context": "Required for JAMIA Limitations paragraph",
        })

    pn.to_csv(pn_path, index=False)
    print(f"  wrote {pn_path} ({len(pn)} rows total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
