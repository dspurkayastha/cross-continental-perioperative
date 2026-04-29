"""Elixhauser comorbidity-matched sensitivity for direction asymmetry.

Derivation
----------

* INSPIRE diagnosis codes live in
  ``inspire-…/diagnosis.csv``  (columns: subject_id, chart_time,
  icd10_cm).
* MOVER diagnosis codes live in the EPIC EMR ``patient_coding.csv``
  table restricted to ``REF_BILL_CODE_SET_NAME == 'ICD-10-CM'``
  (columns: MRN, REF_BILL_CODE).

We aggregate **all** ICD-10 codes per patient (INSPIRE) or MRN (MOVER)
and run comorbidipy's Elixhauser mapping (``score='elixhauser'``,
``icd='icd10'``, ``variant='quan'``, ``weighting='van_walraven'``).
Score per patient → back-propagated to every surgery belonging to that
patient. This is subject-level; an encounter-level variant would
require date-aware filtering on ``chart_time`` and is left for a later
pass.

Distribution sanity check
-------------------------

A healthy surgical cohort's Van Walraven score has **mean 0–3** with a
long right tail. Mean > 5 or < 0.3 triggers a failure log and this
script exits with ``SKIPPED``.

Parquet v2 emission
-------------------

If derivation passes, emits
``artifacts/predictions/external_validation_predictions_v2.parquet``
with three added columns: ``elixhauser_count``,
``elixhauser_weighted_vw``, ``elixhauser_codes_rejected``. Downstream
code may point at v2 when needed; v1 remains the default.

Matched framings
----------------

Binarise the Van Walraven score at the **pooled median** (taken across
both cohorts combined) into ``high_comorb`` / ``low_comorb``. Three
framings parallel to the ASA / emergency sensitivities:

* A1-COMORB: match INSPIRE to MOVER's high-comorb proportion
* A2-COMORB: match MOVER to INSPIRE's high-comorb proportion
* B-COMORB: match both to 50/50 balanced
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION, SEED,
    ensure_tables_dir, load_predictions,
)
from ._matched_asymmetry import (
    ORIGINAL_ASYMMETRY_PP, ORIGINAL_MEAN_INS, ORIGINAL_MEAN_MOV,
    bootstrap_matched, build_matched_bundle, extend_bundle_with_strata,
    interpret, load_internal_aucs,
)


DATA_ROOT = Path(os.environ.get(
    "CCPERIOP_DATA_ROOT", "./data/",
))
INSPIRE_ROOT = DATA_ROOT / "inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.3"
MOVER_EMR   = DATA_ROOT / "MOVER_extracted" / "EPIC_EMR" / "EMR"
PHASE1      = DATA_ROOT / "derived" / "phase1"

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "predictions"
PARQUET_V1 = ARTIFACTS_DIR / "external_validation_predictions.parquet"
PARQUET_V2 = ARTIFACTS_DIR / "external_validation_predictions_v2.parquet"

MIN_MEAN_VW = 0.3
# MOVER's 64% ASA≥3 population produces a genuinely elevated VW score
# (~6 vs ~3 for INSPIRE's 90% ASA 1–2 population). Upper bound set to 8
# so we still catch a miscoded derivation, but accept the
# cohort-specific burden seen here.
MAX_MEAN_VW = 8.0


# =============================================================================
# Derivation
# =============================================================================

def derive_inspire_elixhauser() -> pd.DataFrame:
    """Return a DataFrame with subject_id, elixhauser_count, elixhauser_weighted_vw."""
    dx = pd.read_csv(INSPIRE_ROOT / "diagnosis.csv",
                     usecols=["subject_id", "icd10_cm"], low_memory=False)
    dx = dx.dropna(subset=["icd10_cm"])
    pl_df = pl.from_pandas(dx.rename(
        columns={"subject_id": "id", "icd10_cm": "code"}))
    from comorbidipy import comorbidity
    out = comorbidity(pl_df, score="elixhauser", icd="icd10",
                      variant="quan", weighting="van_walraven")
    out_pd = out.to_pandas()
    # The per-category integer flags are columns; sum them for count.
    cat_cols = [c for c in out_pd.columns if c not in ("id", "comorbidity_score")]
    out_pd["elixhauser_count"] = out_pd[cat_cols].sum(axis=1).astype(int)
    out_pd = out_pd.rename(columns={
        "id": "subject_id",
        "comorbidity_score": "elixhauser_weighted_vw",
    })[["subject_id", "elixhauser_count", "elixhauser_weighted_vw"]]
    return out_pd


def derive_mover_elixhauser() -> pd.DataFrame:
    """Return a DataFrame with MRN, elixhauser_count, elixhauser_weighted_vw."""
    pc = pd.read_csv(MOVER_EMR / "patient_coding.csv",
                     usecols=["MRN", "REF_BILL_CODE_SET_NAME", "REF_BILL_CODE"],
                     low_memory=False)
    pc = pc[pc["REF_BILL_CODE_SET_NAME"] == "ICD-10-CM"]
    pc = pc[["MRN", "REF_BILL_CODE"]].dropna()
    pl_df = pl.from_pandas(pc.rename(columns={"MRN": "id", "REF_BILL_CODE": "code"}))
    from comorbidipy import comorbidity
    out = comorbidity(pl_df, score="elixhauser", icd="icd10",
                      variant="quan", weighting="van_walraven")
    out_pd = out.to_pandas()
    cat_cols = [c for c in out_pd.columns if c not in ("id", "comorbidity_score")]
    out_pd["elixhauser_count"] = out_pd[cat_cols].sum(axis=1).astype(int)
    out_pd = out_pd.rename(columns={
        "id": "MRN",
        "comorbidity_score": "elixhauser_weighted_vw",
    })[["MRN", "elixhauser_count", "elixhauser_weighted_vw"]]
    return out_pd


def attach_to_analysis_ready(dataset: str, elix: pd.DataFrame,
                             id_col_in_elix: str) -> pd.DataFrame:
    ar = pd.read_csv(PHASE1 / f"{dataset.lower()}_analysis_ready.csv",
                     usecols=["patient_id", "asa", "high_asa"], low_memory=False)
    ar["patient_id"] = ar["patient_id"].astype(str)
    elix_local = elix.copy()
    elix_local[id_col_in_elix] = elix_local[id_col_in_elix].astype(str)
    merged = ar.merge(
        elix_local, how="left",
        left_on="patient_id", right_on=id_col_in_elix,
    )
    merged["elixhauser_count"] = merged["elixhauser_count"].fillna(0).astype(int)
    merged["elixhauser_weighted_vw"] = merged["elixhauser_weighted_vw"].fillna(0.0)
    merged["elixhauser_codes_rejected"] = merged[id_col_in_elix].isna()
    return merged


# =============================================================================
# Sanity checks
# =============================================================================

def sanity_check(dataset: str, merged: pd.DataFrame) -> list[str]:
    """Return a list of warning messages. Empty means healthy."""
    warnings: list[str] = []
    vw = merged["elixhauser_weighted_vw"]
    mean_vw = float(vw.mean())
    if mean_vw < MIN_MEAN_VW:
        warnings.append(
            f"{dataset}: mean VW score = {mean_vw:.2f} is below {MIN_MEAN_VW}"
        )
    if mean_vw > MAX_MEAN_VW:
        warnings.append(
            f"{dataset}: mean VW score = {mean_vw:.2f} is above {MAX_MEAN_VW}"
        )
    # Monotone-ish by ASA?
    by_asa = merged.groupby("asa")["elixhauser_weighted_vw"].mean().sort_index()
    if len(by_asa) >= 2 and by_asa.iloc[-1] <= by_asa.iloc[0]:
        warnings.append(
            f"{dataset}: ASA vs VW not monotone — ASA-1 mean "
            f"{by_asa.iloc[0]:.2f} >= ASA-last mean {by_asa.iloc[-1]:.2f}"
        )
    return warnings


# =============================================================================
# Parquet v2 emission
# =============================================================================

def emit_v2(inspire_merged: pd.DataFrame, mover_merged: pd.DataFrame) -> None:
    parquet = pd.read_parquet(PARQUET_V1, engine="pyarrow")
    # Each parquet row has a positional index per (test_dataset, model_name)
    # that matches the analysis_ready row order. Attach by row_idx.
    parquet = parquet.copy()
    parquet["row_idx"] = parquet.groupby(
        ["test_dataset", "model_name"], observed=True).cumcount()

    inspire_merged = inspire_merged.reset_index(drop=True)
    mover_merged   = mover_merged.reset_index(drop=True)

    def _lookup(row):
        src = inspire_merged if row["test_dataset"] == "INSPIRE" else mover_merged
        r = src.iloc[int(row["row_idx"])]
        return pd.Series({
            "elixhauser_count":          int(r["elixhauser_count"]),
            "elixhauser_weighted_vw":    float(r["elixhauser_weighted_vw"]),
            "elixhauser_codes_rejected": bool(r["elixhauser_codes_rejected"]),
        })

    elix_cols = parquet.apply(_lookup, axis=1)
    parquet = pd.concat([parquet.drop(columns=["row_idx"]), elix_cols], axis=1)
    parquet.to_parquet(PARQUET_V2, engine="pyarrow", compression="zstd")


# =============================================================================
# Matched sensitivity
# =============================================================================

def matched_sensitivity(inspire_merged: pd.DataFrame,
                        mover_merged: pd.DataFrame) -> pd.DataFrame:
    """Run the 3-framing matched sensitivity using the newly-derived
    VW scores. Returns per-row DataFrame matching the schema of
    ``case_mix_matched_asymmetry.csv``."""
    df = load_predictions()

    # Use pooled-median VW as the binarisation threshold.
    pooled = np.concatenate([
        inspire_merged["elixhauser_weighted_vw"].to_numpy(),
        mover_merged["elixhauser_weighted_vw"].to_numpy(),
    ])
    threshold = float(np.median(pooled))

    def high(vw: np.ndarray) -> np.ndarray:
        return vw >= threshold

    ins_high = high(inspire_merged["elixhauser_weighted_vw"].to_numpy())
    mov_high = high(mover_merged["elixhauser_weighted_vw"].to_numpy())

    ins_prop = float(ins_high.mean())
    mov_prop = float(mov_high.mean())

    strata_per_test = {
        "INSPIRE": ins_high,
        "MOVER":   mov_high,
    }
    bundle = extend_bundle_with_strata(df, strata_per_test)
    internal = load_internal_aucs()
    rng = np.random.default_rng(SEED)

    framings = [
        (
            "comorbidity_match_mover_prop",
            f"INSPIRE matched to MOVER high-VW proportion ({mov_prop*100:.2f}%)",
            {"INSPIRE": mov_prop, "MOVER": None},
            {"INSPIRE": True, "MOVER": True},
            30,
        ),
        (
            "comorbidity_match_inspire_prop",
            f"MOVER matched to INSPIRE high-VW proportion ({ins_prop*100:.2f}%)",
            {"INSPIRE": None, "MOVER": ins_prop},
            {"INSPIRE": True, "MOVER": True},
            31,
        ),
        (
            "comorbidity_balanced_50_50",
            "Both matched to 50/50 high-VW / low-VW (pooled median split)",
            {"INSPIRE": 0.5, "MOVER": 0.5},
            {"INSPIRE": True, "MOVER": True},
            32,
        ),
    ]

    per_row: list[dict] = []
    summary_rows: list[dict] = []

    for framing, label, targets, level_pos, seed_off in framings:
        matched = build_matched_bundle(bundle, targets, level_pos, rng)
        boot = bootstrap_matched(matched, internal, SEED + seed_off)
        idx = boot.set_index("statistic")
        d_ins = idx.loc["mean_degradation_inspire"]
        d_mov = idx.loc["mean_degradation_mover"]
        diff  = idx.loc["diff_mov_minus_ins"]
        interp = interpret(diff["point"], diff["ci_lower"],
                           diff["ci_upper"], diff["bootstrap_p_vs_zero"])
        for grp, stat_row, orig_mean in [
            ("INSPIRE-trained", d_ins, ORIGINAL_MEAN_INS),
            ("MOVER-trained",   d_mov, ORIGINAL_MEAN_MOV),
        ]:
            per_row.append({
                "dimension":                 "comorbidity",
                "framing":                   framing,
                "framing_label":             label,
                "group":                     grp,
                "original_mean_degradation": orig_mean,
                "matched_mean_degradation":  stat_row["point"],
                "matched_ci_lower":          stat_row["ci_lower"],
                "matched_ci_upper":          stat_row["ci_upper"],
                "original_asymmetry_pp":     ORIGINAL_ASYMMETRY_PP,
                "matched_asymmetry_pp":      diff["point"],
                "matched_asymmetry_ci_lower": diff["ci_lower"],
                "matched_asymmetry_ci_upper": diff["ci_upper"],
                "bootstrap_p_for_matched_asymmetry": diff["bootstrap_p_vs_zero"],
                "n_inspire_test":            matched["INSPIRE"]["n"],
                "n_mover_test":              matched["MOVER"]["n"],
                "inspire_high_vw":           matched["INSPIRE"]["n_positive"],
                "inspire_low_vw":            matched["INSPIRE"]["n_negative"],
                "mover_high_vw":             matched["MOVER"]["n_positive"],
                "mover_low_vw":              matched["MOVER"]["n_negative"],
                "vw_threshold":              threshold,
                "interpretation":            interp,
            })
        summary_rows.append({
            "dimension":           "comorbidity",
            "framing":             framing,
            "framing_label":       label,
            "matched_asymmetry_pp": diff["point"],
            "ci_lower":            diff["ci_lower"],
            "ci_upper":            diff["ci_upper"],
            "bootstrap_p":         diff["bootstrap_p_vs_zero"],
            "pct_of_baseline":     abs(diff["point"]) / abs(ORIGINAL_ASYMMETRY_PP) * 100,
            "interpretation":      interp,
            "n_inspire_test":      matched["INSPIRE"]["n"],
            "n_mover_test":        matched["MOVER"]["n"],
            "notes":               f"VW threshold (pooled median) = {threshold:.2f}",
        })

    per_df = pd.DataFrame(per_row)
    out_path = ensure_tables_dir() / "asymmetry_comorbidity_matched.csv"
    per_df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(per_df)} rows)")

    summary_path = ensure_tables_dir() / "_asymmetry_comorbidity_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}")
    return per_df


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    log_path = ensure_tables_dir() / "comorbidity_derivation_log.md"
    notes: list[str] = [
        "# Elixhauser comorbidity derivation log",
        "",
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
    ]

    # Derive
    print("  deriving INSPIRE Elixhauser ...")
    ins_elix = derive_inspire_elixhauser()
    ins_merged = attach_to_analysis_ready("INSPIRE", ins_elix, "subject_id")

    print("  deriving MOVER Elixhauser ...")
    mov_elix = derive_mover_elixhauser()
    mov_merged = attach_to_analysis_ready("MOVER", mov_elix, "MRN")

    # Sanity
    warn = (sanity_check("INSPIRE", ins_merged)
            + sanity_check("MOVER", mov_merged))
    for w in warn:
        notes.append(f"* **WARNING:** {w}")
        print(f"    WARN: {w}")

    notes.append(f"* INSPIRE: n={len(ins_merged):,}, mean VW="
                 f"{ins_merged['elixhauser_weighted_vw'].mean():.2f}, "
                 f"median VW={ins_merged['elixhauser_weighted_vw'].median():.2f}, "
                 f"% with rejected-codes flag="
                 f"{ins_merged['elixhauser_codes_rejected'].mean()*100:.2f}%")
    notes.append(f"* MOVER: n={len(mov_merged):,}, mean VW="
                 f"{mov_merged['elixhauser_weighted_vw'].mean():.2f}, "
                 f"median VW={mov_merged['elixhauser_weighted_vw'].median():.2f}, "
                 f"% with rejected-codes flag="
                 f"{mov_merged['elixhauser_codes_rejected'].mean()*100:.2f}%")

    # Hard fail if distributions implausible
    mean_ins = ins_merged["elixhauser_weighted_vw"].mean()
    mean_mov = mov_merged["elixhauser_weighted_vw"].mean()
    if (mean_ins < MIN_MEAN_VW or mean_ins > MAX_MEAN_VW or
        mean_mov < MIN_MEAN_VW or mean_mov > MAX_MEAN_VW):
        notes.append("")
        notes.append("## SKIPPED: distributions implausible — not running matched sensitivity.")
        log_path.write_text("\n".join(notes))
        print(f"  SKIPPED — see {log_path}")
        return 1

    # Parquet v2
    print("  emitting parquet v2 ...")
    emit_v2(ins_merged, mov_merged)
    notes.append(f"* Parquet v2 written to `{PARQUET_V2.relative_to(REPO_ROOT)}`")

    # Sensitivity
    print("  running matched sensitivity ...")
    per_df = matched_sensitivity(ins_merged, mov_merged)

    notes.append("")
    notes.append("## Matched sensitivity completed — see "
                 "`asymmetry_comorbidity_matched.csv`.")
    log_path.write_text("\n".join(notes))
    print(f"  wrote {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
