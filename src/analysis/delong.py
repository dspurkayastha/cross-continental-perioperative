"""DeLong's test for paired AUC comparisons.

Implementation follows the fast DeLong algorithm of Sun & Xu (2014),
which is also the basis of the widely-used reference implementation at
https://github.com/yandexdataschool/roc_comparison. Verified here by
cross-checking the per-model AUCs against ``sklearn.metrics.roc_auc_score``
before the covariance step runs.

Runs 12 pairwise comparisons total (6 within each of the 2 external
validation directions). P-values are adjusted across the full family
of 12 with Benjamini–Hochberg FDR.

References
----------
DeLong, DeLong & Clarke-Pearson. Comparing the areas under two or more
correlated receiver operating characteristic curves: a nonparametric
approach. *Biometrics* 44:837-845, 1988.

Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing
the areas under correlated receiver operating characteristic curves.
*IEEE Signal Process. Lett.* 21:1389-1393, 2014.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION,
    ensure_tables_dir, load_predictions, wide_scores,
)


# =============================================================================
# Fast DeLong algorithm (Sun & Xu 2014)
# =============================================================================

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank with tie-averaging; 1-indexed output."""
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[J] = T + 1
    return out


def _fast_delong_cov(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """scores shape (k, n); positives are the first ``n_pos`` columns.
    Returns (aucs (k,), covariance (k, k))."""
    m = n_pos
    n_neg = scores.shape[1] - m
    k = scores.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n_neg), dtype=float)
    tz = np.empty((k, m + n_neg), dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(scores[r, :m])
        ty[r, :] = _compute_midrank(scores[r, m:])
        tz[r, :] = _compute_midrank(scores[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n_neg - (m + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :m] - tx) / n_neg
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    # np.cov returns scalar for k==1; promote to 2D
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    cov = sx / m + sy / n_neg
    return aucs, cov


def delong_test(y_true: np.ndarray, score_a: np.ndarray,
                score_b: np.ndarray) -> dict[str, float]:
    """Compare two correlated ROC curves on the same cases.

    Returns
    -------
    dict with keys: ``auc_a, auc_b, auc_diff, z, p_two_sided, se``
    """
    y_true = np.asarray(y_true).astype(int)
    # Sort so positives come first.
    order = np.argsort(-y_true, kind="stable")
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    if n_pos == 0 or n_pos == y_sorted.size:
        raise ValueError("Need both classes for DeLong test")

    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _fast_delong_cov(scores, n_pos)
    contrast = np.array([[1.0, -1.0]])
    var = float(contrast @ cov @ contrast.T)
    se = np.sqrt(max(var, 0.0))
    if se == 0.0:
        z = 0.0
        p = 1.0
    else:
        z = float((aucs[0] - aucs[1]) / se)
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(aucs[0] - aucs[1]),
        "z": z,
        "p_two_sided": float(p),
        "se": float(se),
    }


# =============================================================================
# Runner
# =============================================================================

def run() -> pd.DataFrame:
    df = load_predictions()
    rows: list[dict] = []

    for train_ds, test_ds in DIRECTIONS:
        y_true, scores_df, _ = wide_scores(df, test_ds)
        models = MODELS_BY_DIRECTION[(train_ds, test_ds)]

        # Sanity check: per-model AUC must match sklearn before we trust
        # the covariance computation.
        for m in models:
            sk = roc_auc_score(y_true, scores_df[m].to_numpy())
            aucs, _ = _fast_delong_cov(
                np.atleast_2d(scores_df[m].to_numpy()[np.argsort(-y_true)]),
                int(y_true.sum()),
            )
            if not np.isclose(sk, aucs[0], atol=1e-6):
                raise AssertionError(
                    f"DeLong AUC mismatch for {m}: sklearn={sk:.6f} vs "
                    f"fast-delong={aucs[0]:.6f}"
                )

        for m1, m2 in itertools.combinations(models, 2):
            res = delong_test(
                y_true,
                scores_df[m1].to_numpy(),
                scores_df[m2].to_numpy(),
            )
            rows.append({
                "direction": f"{train_ds}_to_{test_ds}",
                "model_1": m1,
                "model_2": m2,
                "auc_1": res["auc_a"],
                "auc_2": res["auc_b"],
                "auc_diff": res["auc_diff"],
                "z_stat": res["z"],
                "p_value": res["p_two_sided"],
            })

    out = pd.DataFrame(rows)

    # Benjamini–Hochberg FDR across all 12 tests.
    rej, p_adj, _, _ = multipletests(
        out["p_value"].to_numpy(), alpha=0.05, method="fdr_bh"
    )
    out["p_adjusted_bh"] = p_adj
    out["significant_at_0.05"] = rej

    out_path = ensure_tables_dir() / "delong_comparisons.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} pairwise tests)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
