"""5-fold cross-fit Platt recalibration with bootstrap CIs.

For each external validation run, the test cohort is stratified into 5
folds (binary outcome, seed 42). The Platt calibrator is fit on 4 folds
(``LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)``)
and applied to the held-out fold; concatenating the 5 held-out
predictions yields a recalibrated probability vector that is independent
of any single calibrator fit. Calibration metrics (slope, intercept,
O:E, Brier) are computed on this held-out vector and 95% percentile CIs
are obtained via 2,000 case-level bootstrap resamples (seed 42, same
convention as ``src.analysis._bootstrap_utils``).

The calibration slope/intercept regression uses the logit of the
predicted probability as the linear predictor.

Inputs
------

The 8 cached per-model prediction CSVs under
``derived/phase3/predictions/`` (columns ``y_true, y_prob_original,
y_prob_recalibrated``). Only ``y_prob_original`` is consumed; the CSV
is rewritten with the cross-fit ``y_prob_recalibrated`` column.

Outputs
-------

* Updates ``y_prob_recalibrated`` column in each of the 8 prediction
  CSVs on ``DATA_ROOT``.
* Updates the ``recalibrated`` sub-dict in ``phase3_results.json``.
* Refreshes post-recal columns in ``phase3_summary.csv``
  (``cal_slope_recal``, ``oe_ratio_recal``, ``brier_recal``).

No model retraining occurs.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_BOOTSTRAPS = 2_000
N_SPLITS = 5

DATA_ROOT = Path(os.environ.get(
    "CCPERIOP_DATA_ROOT", "./data/",
))
PHASE3_DIR = DATA_ROOT / "derived" / "phase3"
PREDICTIONS_DIR = PHASE3_DIR / "predictions"
RESULTS_JSON = PHASE3_DIR / "phase3_results.json"
SUMMARY_CSV = PHASE3_DIR / "phase3_summary.csv"


def _fit_platt(y_prob: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(y_prob.reshape(-1, 1), y_true)
    return clf


def cross_fit_platt(y_true: np.ndarray, y_prob: np.ndarray,
                    n_splits: int = N_SPLITS, seed: int = SEED) -> np.ndarray:
    """Return held-out recalibrated probabilities (length n)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_prob_recal = np.empty_like(y_prob, dtype=float)
    for train_idx, test_idx in skf.split(y_prob, y_true):
        cal = _fit_platt(y_prob[train_idx], y_true[train_idx])
        y_prob_recal[test_idx] = cal.predict_proba(
            y_prob[test_idx].reshape(-1, 1)
        )[:, 1]
    return y_prob_recal


def _calibration_metrics_point(y_true: np.ndarray,
                               y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    """Point estimates on (y_true, y_prob). Matches phase3 convention."""
    eps = 1e-10
    y_p_clip = np.clip(y_prob, eps, 1 - eps)
    lp = np.log(y_p_clip / (1 - y_p_clip))

    cal_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    cal_model.fit(lp.reshape(-1, 1), y_true)
    slope = float(cal_model.coef_[0][0])
    intercept = float(cal_model.intercept_[0])

    observed = float(y_true.sum())
    expected = float(y_prob.sum())
    oe = observed / expected if expected > 0 else np.nan

    brier = float(brier_score_loss(y_true, y_prob))
    return slope, intercept, float(oe), brier


def calibration_metrics_with_ci(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bootstrap: int = N_BOOTSTRAPS,
                                seed: int = SEED) -> Dict:
    rng = np.random.default_rng(seed)
    slope, intercept, oe, brier = _calibration_metrics_point(y_true, y_prob)

    n = len(y_true)
    slopes, intercepts, oes, briers = [], [], [], []
    attempts = 0
    while len(slopes) < n_bootstrap and attempts < n_bootstrap * 3:
        attempts += 1
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            s, i, o, b = _calibration_metrics_point(y_true[idx], y_prob[idx])
            slopes.append(s)
            intercepts.append(i)
            oes.append(o)
            briers.append(b)
        except Exception:
            continue

    def _ci(arr):
        return [float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5))]

    return {
        "calibration_slope": float(slope),
        "calibration_slope_ci": _ci(slopes),
        "calibration_intercept": float(intercept),
        "calibration_intercept_ci": _ci(intercepts),
        "oe_ratio": float(oe),
        "oe_ratio_ci": _ci(oes),
        "brier": float(brier),
        "brier_ci": _ci(briers),
        "n_bootstrap_success": len(slopes),
    }


def process_model(pred_csv: Path) -> Dict:
    df = pd.read_csv(pred_csv)
    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob_original"].to_numpy(dtype=float)

    y_prob_recal_cv = cross_fit_platt(y_true, y_prob)
    metrics = calibration_metrics_with_ci(y_true, y_prob_recal_cv)

    df["y_prob_recalibrated"] = y_prob_recal_cv
    df.to_csv(pred_csv, index=False)

    return {
        "y_prob_recalibrated": y_prob_recal_cv,
        "metrics": metrics,
    }


def run() -> None:
    assert RESULTS_JSON.exists(), f"missing {RESULTS_JSON}"
    assert SUMMARY_CSV.exists(), f"missing {SUMMARY_CSV}"

    with RESULTS_JSON.open() as f:
        results = json.load(f)
    summary = pd.read_csv(SUMMARY_CSV)

    for key, entry in results.items():
        model_id = entry["model_id"]
        external = entry["external_dataset"]
        pred_csv = PREDICTIONS_DIR / f"{model_id}_on_{external}_predictions.csv"
        assert pred_csv.exists(), f"missing predictions: {pred_csv}"

        print(f"[{key}] cross-fit Platt on n={entry['n_samples']:,}, "
              f"events={entry['n_events']:,} ...", flush=True)

        result = process_model(pred_csv)
        m_new = result["metrics"]
        m_old = entry["recalibrated"]

        # Preserve discrimination metrics untouched (AUC etc).
        new_recal = copy.deepcopy(m_old)
        for k in ("calibration_slope", "calibration_slope_ci",
                  "calibration_intercept", "calibration_intercept_ci",
                  "oe_ratio", "oe_ratio_ci", "brier"):
            new_recal[k] = m_new[k]
        new_recal["brier_ci"] = m_new["brier_ci"]
        new_recal["recalibration_method"] = "platt_5fold_cv"
        new_recal["n_bootstrap_success"] = m_new["n_bootstrap_success"]

        entry["recalibrated"] = new_recal
        entry["recalibration_method"] = "platt_5fold_cv"

        row_mask = summary["validation"] == key
        summary.loc[row_mask, "cal_slope_recal"] = m_new["calibration_slope"]
        summary.loc[row_mask, "oe_ratio_recal"] = m_new["oe_ratio"]
        summary.loc[row_mask, "brier_recal"] = m_new["brier"]

    with RESULTS_JSON.open("w") as f:
        json.dump(results, f, indent=2)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {RESULTS_JSON}")
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    run()
