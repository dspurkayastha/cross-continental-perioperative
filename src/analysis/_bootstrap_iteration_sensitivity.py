#!/usr/bin/env python3
"""
Bootstrap iteration-count sensitivity sweep for Table S8.

Re-runs the headline direction-asymmetry case-level paired bootstrap at
B = 500, B = 1000, and B = 2000 (canonical) using cached per-model
predictions. Output: a small CSV with point estimate, 95% percentile CI,
and CI width per B value.

Reads: <DATA_ROOT>/derived/phase3/predictions/*.csv
Writes: results/tables/bootstrap_iteration_sensitivity.csv

Usage:
    python src/analysis/_bootstrap_iteration_sensitivity.py
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CCPERIOP_DATA_ROOT", "./data/"))
PHASE3 = DATA_ROOT / "derived" / "phase3"
PRED_DIR = PHASE3 / "predictions"
OUT_CSV = REPO_ROOT / "results" / "tables" / "bootstrap_iteration_sensitivity.csv"

INSPIRE_MODELS = ["XGB-INS-A", "XGB-INS-B", "LR-INS-A", "LR-INS-B"]
MOVER_MODELS = ["XGB-MOV-A", "XGB-MOV-B", "LR-MOV-A", "LR-MOV-B"]
SEED = 42

INTERNAL_AUC = {
    "XGB-INS-A": 0.8072, "XGB-INS-B": 0.8604,
    "LR-INS-A": 0.8513,  "LR-INS-B": 0.8867,
    "XGB-MOV-A": 0.9475, "XGB-MOV-B": 0.9415,
    "LR-MOV-A": 0.9139,  "LR-MOV-B": 0.9312,
}


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n = int(y_true.size); n_pos = int(y_true.sum()); n_neg = n - n_pos
    if n == 0 or n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(y_score)
    sum_r_pos = float(r[y_true.astype(bool)].sum())
    return (sum_r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_pred(model_id, test_ds):
    csv = PRED_DIR / f"{model_id}_on_{test_ds}_predictions.csv"
    df = pd.read_csv(csv)
    return df["y_true"].to_numpy(), df["y_prob_original"].to_numpy()


def asymmetry_bootstrap(B: int, seed: int = SEED) -> dict:
    """Compute the direction-asymmetry mean-difference across B iterations."""
    # Load all 8 prediction sets
    ins_data = [load_pred(m, "MOVER") for m in INSPIRE_MODELS]
    mov_data = [load_pred(m, "INSPIRE") for m in MOVER_MODELS]

    n_mover = len(ins_data[0][0])
    n_inspire = len(mov_data[0][0])

    rng = np.random.default_rng(seed)

    diffs = np.empty(B, dtype=float)
    for b in range(B):
        idx_mover = rng.integers(0, n_mover, size=n_mover)
        idx_inspire = rng.integers(0, n_inspire, size=n_inspire)

        # INSPIRE-trained on MOVER: degradation = (internal - external) / internal
        d_ins = np.array([
            (INTERNAL_AUC[m] - safe_auc(y[idx_mover], p[idx_mover])) / INTERNAL_AUC[m]
            for (y, p), m in zip(ins_data, INSPIRE_MODELS)
        ])
        # MOVER-trained on INSPIRE: same degradation calc
        d_mov = np.array([
            (INTERNAL_AUC[m] - safe_auc(y[idx_inspire], p[idx_inspire])) / INTERNAL_AUC[m]
            for (y, p), m in zip(mov_data, MOVER_MODELS)
        ])
        diffs[b] = float(np.nanmean(d_mov) - np.nanmean(d_ins))

    return {
        "B": B,
        "point_estimate_pp": float(np.nanmean(diffs) * 100),
        "median_bootstrap_pp": float(np.nanmedian(diffs) * 100),
        "ci_lower_pp": float(np.nanpercentile(diffs, 2.5) * 100),
        "ci_upper_pp": float(np.nanpercentile(diffs, 97.5) * 100),
        "ci_width_pp": float((np.nanpercentile(diffs, 97.5) - np.nanpercentile(diffs, 2.5)) * 100),
        "seed": seed,
    }


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for B in [500, 1000, 2000]:
        print(f"running B = {B} ...", flush=True)
        rows.append(asymmetry_bootstrap(B))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(df.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
