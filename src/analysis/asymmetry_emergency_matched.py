"""Emergency-proportion matched sensitivity for the direction asymmetry.

Three framings, same bootstrap machinery as the ASA case-mix analysis:

* A1-EMERG: match INSPIRE's emergency proportion **up** to MOVER's
  15.6%; MOVER unchanged.
* A2-EMERG: match MOVER's emergency proportion **down** to INSPIRE's
  7.9%; INSPIRE unchanged.
* B-EMERG: match both to the midpoint (11.75%).

Writes ``results/tables/asymmetry_emergency_matched.csv`` and seeds
rows for ``asymmetry_all_dimensions_summary.csv``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ._bootstrap_utils import (
    SEED, ensure_tables_dir, load_predictions,
)
from ._matched_asymmetry import (
    ORIGINAL_ASYMMETRY_PP, ORIGINAL_MEAN_INS, ORIGINAL_MEAN_MOV,
    bootstrap_matched, build_matched_bundle, extend_bundle_with_strata,
    interpret, load_internal_aucs,
)


MOVER_EMERGENCY_PROP   = 0.156
INSPIRE_EMERGENCY_PROP = 0.079
MIDPOINT_EMERGENCY_PROP = (MOVER_EMERGENCY_PROP + INSPIRE_EMERGENCY_PROP) / 2


FRAMINGS = [
    # (name, label, targets_per_test, level_positive_per_test, seed_offset)
    (
        "emergency_match_mover_prop",
        "INSPIRE matched to MOVER emergency rate (15.6%)",
        {"INSPIRE": MOVER_EMERGENCY_PROP, "MOVER": None},
        {"INSPIRE": True, "MOVER": True},
        20,
    ),
    (
        "emergency_match_inspire_prop",
        "MOVER matched to INSPIRE emergency rate (7.9%)",
        {"INSPIRE": None, "MOVER": INSPIRE_EMERGENCY_PROP},
        {"INSPIRE": True, "MOVER": True},
        21,
    ),
    (
        "emergency_balanced_midpoint",
        f"Both matched to midpoint emergency rate ({MIDPOINT_EMERGENCY_PROP*100:.2f}%)",
        {"INSPIRE": MIDPOINT_EMERGENCY_PROP, "MOVER": MIDPOINT_EMERGENCY_PROP},
        {"INSPIRE": True, "MOVER": True},
        22,
    ),
]


def run() -> pd.DataFrame:
    df = load_predictions()

    # Build bundle with emergency strata per test dataset
    strata_per_test: dict[str, np.ndarray] = {}
    for ts in ("INSPIRE", "MOVER"):
        sub = df[(df["test_dataset"] == ts) &
                 (df["model_name"].isin({"XGB-INS-A", "XGB-MOV-A"}))].copy()
        sub["row_idx"] = sub.groupby("model_name", observed=True).cumcount()
        if ts == "INSPIRE":
            first = sub[sub["model_name"] == "XGB-MOV-A"].sort_values("row_idx")
        else:
            first = sub[sub["model_name"] == "XGB-INS-A"].sort_values("row_idx")
        strata_per_test[ts] = first["emergency"].astype(bool).to_numpy()

    bundle = extend_bundle_with_strata(df, strata_per_test)

    internal = load_internal_aucs()
    rng = np.random.default_rng(SEED)

    per_row: list[dict] = []
    summary_rows: list[dict] = []

    for framing, label, targets, level_pos, seed_off in FRAMINGS:
        matched = build_matched_bundle(bundle, targets, level_pos, rng)
        boot = bootstrap_matched(matched, internal, SEED + seed_off)
        idx = boot.set_index("statistic")

        d_ins = idx.loc["mean_degradation_inspire"]
        d_mov = idx.loc["mean_degradation_mover"]
        diff  = idx.loc["diff_mov_minus_ins"]

        interp = interpret(
            diff["point"], diff["ci_lower"], diff["ci_upper"],
            diff["bootstrap_p_vs_zero"],
        )

        for grp, stat_row, orig_mean in [
            ("INSPIRE-trained", d_ins, ORIGINAL_MEAN_INS),
            ("MOVER-trained",   d_mov, ORIGINAL_MEAN_MOV),
        ]:
            per_row.append({
                "dimension":                 "emergency",
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
                "inspire_emergency":         matched["INSPIRE"]["n_positive"],
                "inspire_elective":          matched["INSPIRE"]["n_negative"],
                "mover_emergency":           matched["MOVER"]["n_positive"],
                "mover_elective":            matched["MOVER"]["n_negative"],
                "interpretation":            interp,
            })
        summary_rows.append({
            "dimension":           "emergency",
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
            "notes":               "",
        })

    per_df     = pd.DataFrame(per_row)
    summary_df = pd.DataFrame(summary_rows)

    out_path = ensure_tables_dir() / "asymmetry_emergency_matched.csv"
    per_df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(per_df)} rows)")

    # Partial summary for later aggregation.
    summary_path = ensure_tables_dir() / "_asymmetry_emergency_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}  ({len(summary_df)} rows)")
    return per_df


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
