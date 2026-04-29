"""Shared engine for all matched-asymmetry sensitivity analyses.

The engine takes a ``strata`` vector (a 2-level categorical) and a set
of framings, each of which specifies a target proportion for one or
both test sets. It then subsamples, recomputes external AUC per model
on the matched data, and runs a 2,000-iteration case-level paired
bootstrap for the direction asymmetry.

Used by:
    * case_mix_matched_asymmetry.py   (ASA strata)
    * asymmetry_emergency_matched.py  (emergency / elective)
    * asymmetry_comorbidity_matched.py (Elixhauser high/low)
    * asymmetry_temporal_matched.py   (in-overlap / outside)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION, N_BOOTSTRAPS, SEED,
    paired_bootstrap, safe_auc, wide_all,
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


def load_internal_aucs() -> dict[str, float]:
    if TRAINING_SUMMARY.exists():
        with TRAINING_SUMMARY.open() as f:
            return {m: float(v["auc"]) for m, v in json.load(f)["models"].items()}
    return dict(FALLBACK_INTERNAL_AUC)


# =============================================================================
# Subsample to a two-stratum target proportion
# =============================================================================

def matched_indices(strata: np.ndarray, level_positive: object,
                    target_positive_pct: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Return indices that achieve ``target_positive_pct`` of
    ``level_positive`` in ``strata`` while maximising n. Same logic as
    ``case_mix_matched_asymmetry.matched_indices`` but parameterised by
    any two-level strata vector.
    """
    idx_pos = np.where(strata == level_positive)[0]
    idx_neg = np.where(strata != level_positive)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    # Option 1: keep all positives, subsample negatives
    n_neg_needed = int(round(n_pos * (1 - target_positive_pct) / target_positive_pct))
    opt1_valid   = n_neg_needed <= n_neg
    total_1 = n_pos + n_neg_needed if opt1_valid else -1

    # Option 2: keep all negatives, subsample positives
    n_pos_needed = int(round(n_neg * target_positive_pct / (1 - target_positive_pct)))
    opt2_valid   = n_pos_needed <= n_pos
    total_2 = n_pos_needed + n_neg if opt2_valid else -1

    if not opt1_valid and not opt2_valid:
        raise ValueError(
            f"Cannot achieve target {target_positive_pct:.3f}: "
            f"n_positive={n_pos}, n_negative={n_neg}"
        )

    use_opt1 = opt1_valid and (not opt2_valid or total_1 >= total_2)

    if use_opt1:
        chosen_neg = (rng.choice(idx_neg, size=n_neg_needed, replace=False)
                      if n_neg_needed < n_neg else idx_neg)
        chosen_pos = idx_pos
    else:
        chosen_pos = (rng.choice(idx_pos, size=n_pos_needed, replace=False)
                      if n_pos_needed < n_pos else idx_pos)
        chosen_neg = idx_neg

    return np.sort(np.concatenate([chosen_pos, chosen_neg]))


# =============================================================================
# Bundle wrangling
# =============================================================================

def extend_bundle_with_strata(
    df: pd.DataFrame, strata_columns: dict[str, pd.Series],
    score_col: str = "y_pred_prob_raw",
) -> dict:
    """Build ``bundle`` analogous to ``wide_all`` but attaches a
    user-provided strata vector per test_dataset.

    ``strata_columns`` maps ``test_dataset`` → strata values in row
    order. The length must match the test-dataset row count.
    """
    bundle = wide_all(df, score_col)
    for ts, strata in strata_columns.items():
        if len(strata) != bundle[ts]["n"]:
            raise AssertionError(
                f"{ts}: strata length {len(strata)} != bundle n {bundle[ts]['n']}"
            )
        bundle[ts]["strata"] = np.asarray(strata)
    return bundle


def build_matched_bundle(bundle: dict,
                         targets_per_test: dict,
                         level_positive_per_test: dict,
                         rng: np.random.Generator) -> dict:
    """Subsample each test_dataset bundle using ``targets_per_test``.

    ``targets_per_test[test_ds]`` is either ``None`` (no subsampling) or
    a float target proportion of ``level_positive_per_test[test_ds]``.
    """
    matched: dict = {}
    for test_ds, b in bundle.items():
        target = targets_per_test.get(test_ds)
        if target is None:
            idx = np.arange(b["n"])
        else:
            idx = matched_indices(
                b["strata"], level_positive_per_test[test_ds], target, rng,
            )
        m = {
            "y_true":       b["y_true"][idx],
            "asa_stratum":  b["asa_stratum"][idx],
            "scores":       b["scores"].iloc[idx].reset_index(drop=True),
            "models":       b["models"],
            "n":            len(idx),
            "train_dataset": b["train_dataset"],
            "strata":       b["strata"][idx],
            "n_positive":   int((b["strata"][idx] == level_positive_per_test[test_ds]).sum())
                            if level_positive_per_test[test_ds] is not None else 0,
        }
        m["n_negative"] = m["n"] - m["n_positive"]
        matched[test_ds] = m
    return matched


# =============================================================================
# Bootstrap wrapper
# =============================================================================

def bootstrap_matched(matched: dict, internal: dict, seed: int) -> pd.DataFrame:
    model_test = {m: "MOVER"   for m in INSPIRE_MODELS}
    model_test.update({m: "INSPIRE" for m in MOVER_MODELS})

    def degradation(m: str, idx: dict[str, np.ndarray] | None) -> float:
        ts = model_test[m]
        b = matched[ts]
        y = b["y_true"]; s = b["scores"][m].to_numpy()
        if idx is not None:
            ii = idx[ts]; y, s = y[ii], s[ii]
        return (internal[m] - safe_auc(y, s)) / internal[m]

    def stat_fn(idx: dict[str, np.ndarray] | None) -> dict[str, float]:
        d_ins = np.array([degradation(m, idx) for m in INSPIRE_MODELS])
        d_mov = np.array([degradation(m, idx) for m in MOVER_MODELS])
        mi, mm = float(np.nanmean(d_ins)), float(np.nanmean(d_mov))
        return {
            "mean_degradation_inspire": mi,
            "mean_degradation_mover":   mm,
            "diff_mov_minus_ins":       mm - mi,
            "ratio_mov_over_ins":       (mm / mi) if mi > 0 else float("nan"),
        }

    n_by_group = {ts: b["n"] for ts, b in matched.items()}
    return paired_bootstrap(stat_fn, n_by_group, n_bootstraps=N_BOOTSTRAPS, seed=seed)


# =============================================================================
# Baseline reference numbers (unmatched, from direction_asymmetry_bootstrap.csv)
# =============================================================================

ORIGINAL_ASYMMETRY_PP = 0.08529513967311195
ORIGINAL_CI_LOWER     = 0.06906403
ORIGINAL_CI_UPPER     = 0.10239782
ORIGINAL_P            = 0.001
ORIGINAL_MEAN_INS     = 0.054009624625638714
ORIGINAL_MEAN_MOV     = 0.13930476429875055


def interpret(matched_diff: float, ci_lo: float, ci_hi: float, p: float) -> str:
    rel = abs(matched_diff) / abs(ORIGINAL_ASYMMETRY_PP) if ORIGINAL_ASYMMETRY_PP else 0
    ci_spans_zero = (ci_lo <= 0 <= ci_hi)
    if rel < 0.50 and ci_spans_zero:
        return "driver_primary"
    if rel >= 0.80 and p < 0.01:
        return "not_primary_driver"
    if (0.50 <= rel < 0.80) or (p < 0.05 and rel < 0.80):
        return "partial_driver"
    return "indeterminate"
