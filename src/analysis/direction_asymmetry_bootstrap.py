"""Direction asymmetry — case-level paired bootstrap (PRIMARY test).

Replaces the permutation test as the paper's centerpiece inferential
claim. The permutation test lives on as a sensitivity analysis
(``permutation_direction.py``) because its combinatorial floor (1/70)
is informative but its power ceiling (1/70 ≈ 0.014) is not clinically
useful.

Setup
-----

* Internal AUC per model is fixed (read from
  ``derived/phase2/models/phase2_3_training_summary.json``);
  these are out-of-fold CV estimates and are treated as
  population-level for this test.
* External AUC is bootstrapped: 2,000 case-level paired resamples.
  Two independent resamples per iteration — one for the MOVER test
  set (INSPIRE-trained models) and one for the INSPIRE test set
  (MOVER-trained models).
* Degradation per model per iteration:
      degradation = (internal − external_boot) / internal
* Group statistics per iteration:
      mean_ins = mean of 4 INSPIRE-trained degradations
      mean_mov = mean of 4 MOVER-trained degradations
      diff = mean_mov − mean_ins
      ratio = mean_mov / mean_ins (undefined if mean_ins ≤ 0)

Reported: point, 95% CI, two-sided bootstrap p-value for H0: diff = 0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION, N_BOOTSTRAPS, SEED,
    ensure_tables_dir, load_predictions, paired_bootstrap, safe_auc, wide_all,
)

DATA_ROOT = Path(os.environ.get(
    "CCPERIOP_DATA_ROOT", "./data/",
))
TRAINING_SUMMARY = (
    DATA_ROOT / "derived" / "phase2" / "models" / "phase2_3_training_summary.json"
)

INSPIRE_MODELS = ["XGB-INS-A", "XGB-INS-B", "LR-INS-A", "LR-INS-B"]
MOVER_MODELS   = ["XGB-MOV-A", "XGB-MOV-B", "LR-MOV-A", "LR-MOV-B"]

FALLBACK_INTERNAL_AUC = {
    "XGB-INS-A": 0.8072, "XGB-INS-B": 0.8604,
    "LR-INS-A":  0.8513, "LR-INS-B":  0.8867,
    "XGB-MOV-A": 0.9475, "XGB-MOV-B": 0.9415,
    "LR-MOV-A":  0.9139, "LR-MOV-B":  0.9312,
}


def _load_internal_aucs() -> tuple[dict[str, float], str]:
    if TRAINING_SUMMARY.exists():
        with TRAINING_SUMMARY.open() as f:
            data = json.load(f)
        return {m: float(v["auc"]) for m, v in data["models"].items()}, TRAINING_SUMMARY.name
    return dict(FALLBACK_INTERNAL_AUC), "hard-coded fallback"


def run() -> pd.DataFrame:
    df = load_predictions()
    bundle = wide_all(df, "y_pred_prob_raw")
    internal, internal_src = _load_internal_aucs()

    model_test = {m: "MOVER"   for m in INSPIRE_MODELS}
    model_test.update({m: "INSPIRE" for m in MOVER_MODELS})

    def degradation(m: str, idx: dict[str, np.ndarray] | None) -> float:
        ts = model_test[m]
        b = bundle[ts]
        y = b["y_true"]; s = b["scores"][m].to_numpy()
        if idx is not None:
            ii = idx[ts]
            y, s = y[ii], s[ii]
        ext = safe_auc(y, s)
        return (internal[m] - ext) / internal[m]

    def stat_fn(idx: dict[str, np.ndarray] | None) -> dict[str, float]:
        degs_ins = np.array([degradation(m, idx) for m in INSPIRE_MODELS])
        degs_mov = np.array([degradation(m, idx) for m in MOVER_MODELS])
        mean_ins = float(np.nanmean(degs_ins))
        mean_mov = float(np.nanmean(degs_mov))
        diff = mean_mov - mean_ins
        ratio = mean_mov / mean_ins if mean_ins > 0 else float("nan")
        return {
            "mean_degradation_inspire": mean_ins,
            "mean_degradation_mover":   mean_mov,
            "diff_mov_minus_ins":       diff,
            "ratio_mov_over_ins":       ratio,
        }

    n_by_group = {ts: b["n"] for ts, b in bundle.items()}
    boot = paired_bootstrap(
        stat_fn, n_by_group, n_bootstraps=N_BOOTSTRAPS, seed=SEED + 3,
    )

    # Reshape for readable CSV: one row per reported statistic.
    boot = boot.set_index("statistic")

    # Permutation-test numbers for cross-reference (already produced by
    # permutation_direction.py). Read them if the CSV exists; otherwise
    # leave blank.
    perm_path = ensure_tables_dir() / "direction_asymmetry_permutation.csv"
    perm_exact_p: float | str = ""
    perm_mc_p:    float | str = ""
    if perm_path.exists():
        perm_df = pd.read_csv(perm_path)
        if len(perm_df) == 1:
            perm_exact_p = float(perm_df["exact_p_value"].iloc[0])
            perm_mc_p    = float(perm_df["mc_p_value"].iloc[0])

    rows = []
    def _row(role: str, key: str):
        r = boot.loc[key]
        rows.append({
            "role":            role,
            "statistic":       key,
            "point":           r["point"],
            "bootstrap_median": r["bootstrap_median"],
            "ci_lower":        r["ci_lower"],
            "ci_upper":        r["ci_upper"],
            "bootstrap_p_vs_zero": r["bootstrap_p_vs_zero"],
            "n_bootstraps":    N_BOOTSTRAPS,
            "seed":            SEED + 3,
            "inferential_claim": role == "PRIMARY",
            "permutation_exact_p":      perm_exact_p,
            "permutation_mc_p":         perm_mc_p,
            "internal_auc_source":      internal_src,
        })

    _row("SUPPORTING",   "mean_degradation_inspire")
    _row("SUPPORTING",   "mean_degradation_mover")
    _row("PRIMARY",      "diff_mov_minus_ins")
    _row("SUPPORTING",   "ratio_mov_over_ins")

    out = pd.DataFrame(rows)
    out_path = ensure_tables_dir() / "direction_asymmetry_bootstrap.csv"
    out.to_csv(out_path, index=False)
    primary = out[out["inferential_claim"]].iloc[0]
    print(
        f"  wrote {out_path}  "
        f"(diff={primary['point']:+.4f}, 95% CI [{primary['ci_lower']:+.4f},"
        f"{primary['ci_upper']:+.4f}], bootstrap p={primary['bootstrap_p_vs_zero']:.4f})"
    )
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
