"""4-vs-4 model-level permutation test for direction asymmetry.

A sensitivity check (per Methods §3 / Supplementary §S3.3) for the
direction-asymmetry claim, run alongside the primary case-level paired
bootstrap. With 4 INSPIRE-trained vs. 4 MOVER-trained models the exact
permutation space has $\\binom{8}{4} = 70$ unique splits and the
minimum achievable p ≈ 0.014; both the exact and Monte Carlo p are
reported here, consistent with the bound stated in Methods §3 and the
case-level inference motivated there.

Setup
-----

* Degradation per model = (internal_auc − external_auc) / internal_auc.
* Four INSPIRE-trained models and four MOVER-trained models.
* Internal AUCs are read from the cached training summary at
  ``derived/phase2/models/phase2_3_training_summary.json``; if
  unavailable, hard-coded manuscript Table 2 values
  (``FALLBACK_INTERNAL_AUC``) are used.
* External AUCs are read from the canonical predictions parquet.

Test statistic: ``|mean_degradation_INSPIRE − mean_degradation_MOVER|``.

Procedure: pool the 8 degradation values, randomly assign 4 to each
group without replacement, compute the statistic, repeat
``N_PERMUTATIONS`` times. p = (#permutations with statistic ≥ observed
+ 1) / (N_PERMUTATIONS + 1). The exact 70-split enumeration is also
reported.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION,
    ensure_tables_dir, load_predictions, wide_scores,
)


SEED = 45
N_PERMUTATIONS = 10_000

DATA_ROOT = Path(
    os.environ.get(
        "CCPERIOP_DATA_ROOT",
        "./data/",
    )
)
TRAINING_SUMMARY = (
    DATA_ROOT / "derived" / "phase2" / "models" / "phase2_3_training_summary.json"
)

INSPIRE_MODELS = ["XGB-INS-A", "XGB-INS-B", "LR-INS-A", "LR-INS-B"]
MOVER_MODELS   = ["XGB-MOV-A", "XGB-MOV-B", "LR-MOV-A", "LR-MOV-B"]

# Manuscript Table 2 values, used only if training_summary.json unavailable.
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
        return {m: float(v["auc"]) for m, v in data["models"].items()}, "phase2_3_training_summary.json"
    return dict(FALLBACK_INTERNAL_AUC), "hard-coded fallback (manuscript Table 2)"


def _external_aucs(df: pd.DataFrame) -> dict[str, float]:
    aucs: dict[str, float] = {}
    for train_ds, test_ds in DIRECTIONS:
        y_true, scores, _ = wide_scores(df, test_ds)
        for m in MODELS_BY_DIRECTION[(train_ds, test_ds)]:
            aucs[m] = float(roc_auc_score(y_true, scores[m].to_numpy()))
    return aucs


def _test_stat(degradations: np.ndarray, labels: np.ndarray) -> float:
    """labels == 0 for INSPIRE, 1 for MOVER."""
    ins = degradations[labels == 0]
    mov = degradations[labels == 1]
    return float(abs(np.mean(ins) - np.mean(mov)))


def run() -> pd.DataFrame:
    df = load_predictions()
    internal, internal_src = _load_internal_aucs()
    external = _external_aucs(df)

    degradations: dict[str, float] = {}
    for m in INSPIRE_MODELS + MOVER_MODELS:
        degradations[m] = (internal[m] - external[m]) / internal[m]

    deg_array = np.array(
        [degradations[m] for m in INSPIRE_MODELS + MOVER_MODELS],
        dtype=float,
    )
    obs_labels = np.array([0] * 4 + [1] * 4)
    observed = _test_stat(deg_array, obs_labels)

    # Exact enumeration of all C(8,4) = 70 splits (fast, since it's tiny)
    exact_stats: list[float] = []
    all_indices = list(range(8))
    for combo in itertools.combinations(all_indices, 4):
        labels = np.zeros(8, dtype=int)
        labels[list(combo)] = 1
        exact_stats.append(_test_stat(deg_array, labels))
    exact_stats = np.array(exact_stats)
    # Exact two-sided permutation p-value
    exact_p = float(np.mean(exact_stats >= observed - 1e-12))

    # Also run the requested Monte Carlo permutation with 10,000 draws.
    rng = np.random.default_rng(SEED)
    mc_stats = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        perm = rng.permutation(8)
        labels = np.zeros(8, dtype=int)
        labels[perm[:4]] = 0
        labels[perm[4:]] = 1
        mc_stats[i] = _test_stat(deg_array, labels)
    mc_hits = int((mc_stats >= observed - 1e-12).sum())
    mc_p = (mc_hits + 1) / (N_PERMUTATIONS + 1)  # +1/+1 adjustment

    summary_row = {
        "observed_statistic":       observed,
        "mean_degradation_inspire": float(np.mean(deg_array[:4])),
        "mean_degradation_mover":   float(np.mean(deg_array[4:])),
        "degradation_ratio_mov_over_ins":
            float(np.mean(deg_array[4:]) / np.mean(deg_array[:4]))
            if np.mean(deg_array[:4]) != 0 else float("inf"),
        "n_permutations_mc":    N_PERMUTATIONS,
        "mc_p_value":           mc_p,
        "mc_p_value_formatted": f"{mc_hits}/{N_PERMUTATIONS} + 1",
        "exact_n_splits":       len(exact_stats),
        "exact_p_value":        exact_p,
        "achievable_min_p":     float(1.0 / len(exact_stats)),  # one tie with observed
        "null_dist_mean":       float(exact_stats.mean()),
        "null_dist_2.5_pct":    float(np.percentile(exact_stats, 2.5)),
        "null_dist_97.5_pct":   float(np.percentile(exact_stats, 97.5)),
        "internal_auc_source":  internal_src,
        "seed": SEED,
    }

    # Per-model degradation values — useful downstream.
    per_model = pd.DataFrame({
        "model": list(degradations),
        "group": ["INSPIRE-trained"] * 4 + ["MOVER-trained"] * 4,
        "internal_auc": [internal[m] for m in degradations],
        "external_auc": [external[m] for m in degradations],
        "degradation":  [degradations[m] for m in degradations],
    })

    out = pd.DataFrame([summary_row])
    out_path = ensure_tables_dir() / "direction_asymmetry_permutation.csv"
    out.to_csv(out_path, index=False)
    per_model_path = ensure_tables_dir() / "direction_asymmetry_per_model.csv"
    per_model.to_csv(per_model_path, index=False)
    print(f"  wrote {out_path}  (exact p={exact_p:.4f}, MC p={mc_p:.4f})")
    print(f"  wrote {per_model_path}")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
