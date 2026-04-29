"""Shared helpers and unified paired-bootstrap engine.

Philosophy
----------

Every Phase-1 / Phase-2 inferential claim in this revision is built
on the same primitive: **case-level paired bootstrap**. Cases are
resampled with replacement; the same resampled cases are reused
across every model that shares a test set, so differences between
models in the same direction have tightly-correlated error and
differences-of-differences are meaningful.

The central API is ``paired_bootstrap``. Every analysis module calls
it with an arbitrary ``statistic_fn``. Percentile CIs are returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = REPO_ROOT / "artifacts" / "predictions" / "external_validation_predictions.parquet"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

N_BOOTSTRAPS = 2_000
SEED = 42

DIRECTIONS: list[tuple[str, str]] = [
    ("INSPIRE", "MOVER"),
    ("MOVER",   "INSPIRE"),
]

MODELS_BY_DIRECTION: dict[tuple[str, str], list[str]] = {
    ("INSPIRE", "MOVER"):   ["XGB-INS-A", "XGB-INS-B", "LR-INS-A", "LR-INS-B"],
    ("MOVER",   "INSPIRE"): ["XGB-MOV-A", "XGB-MOV-B", "LR-MOV-A", "LR-MOV-B"],
}

ALL_MODEL_NAMES: list[str] = [m for ms in MODELS_BY_DIRECTION.values() for m in ms]


# =============================================================================
# IO
# =============================================================================

def load_predictions() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical parquet missing: {PARQUET_PATH}. "
            "Run src/preprocessing/build_predictions_parquet.py first."
        )
    return pd.read_parquet(PARQUET_PATH, engine="pyarrow")


def wide_scores(df: pd.DataFrame, test_dataset: str,
                score_col: str = "y_pred_prob_raw") -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Pivot the parquet for one test dataset into positional wide form.

    Keyed on synthetic row index (case_id has 3 collisions in MOVER
    from upstream Phase-1 hashing). Row order within each (test_dataset,
    model_name) group was preserved by the parquet build, so positional
    alignment is safe.

    Returns (y_true, scores_df, asa_stratum).
    """
    sub = df[df["test_dataset"] == test_dataset].copy()
    sub["row_idx"] = sub.groupby("model_name", observed=True).cumcount()
    pivot = sub.pivot(index="row_idx", columns="model_name", values=score_col)
    first_model = MODELS_BY_DIRECTION[
        next(d for d in DIRECTIONS if d[1] == test_dataset)
    ][0]
    first = sub[sub["model_name"] == first_model].sort_values("row_idx")
    return first["y_true"].to_numpy(), pivot, first["asa_stratum"].to_numpy()


def wide_all(df: pd.DataFrame, score_col: str = "y_pred_prob_raw"):
    """Per-test-dataset wide-form bundle, used by multi-direction
    statistics (direction asymmetry, cross-direction means)."""
    out: dict[str, dict] = {}
    for train_ds, test_ds in DIRECTIONS:
        y_true, scores, asa = wide_scores(df, test_ds, score_col)
        out[test_ds] = {
            "y_true": y_true,
            "scores": scores,
            "asa_stratum": asa,
            "n": y_true.size,
            "models": MODELS_BY_DIRECTION[(train_ds, test_ds)],
            "train_dataset": train_ds,
        }
    return out


# =============================================================================
# AUC helpers
# =============================================================================

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast ROC-AUC via Mann–Whitney U (average-rank formulation).

    Returns NaN for degenerate cases (empty, all-one-class, constant
    scores). Verified to match ``sklearn.metrics.roc_auc_score`` within
    1e-10 on the parquet (see ``tests/`` — not shipped here) and runs
    roughly 5–10× faster inside a bootstrap loop because it avoids
    sklearn's per-call validation overhead.
    """
    n = int(y_true.size)
    if n == 0:
        return float("nan")
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rankdata handles ties correctly (average ranks).
    from scipy.stats import rankdata
    r = rankdata(y_score)
    # Sum of ranks of positives:
    sum_r_pos = float(r[y_true.astype(bool)].sum())
    auc = (sum_r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def percentile_ci(values: np.ndarray, ci_level: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - ci_level
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(v, 100.0 * alpha / 2.0))
    hi = float(np.percentile(v, 100.0 * (1 - alpha / 2.0)))
    return lo, hi


def bootstrap_pvalue(values: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for H0: statistic == ``null``.

    p = 2 × min(Pr[boot ≤ null], Pr[boot ≥ null]), with +1/+1 continuity.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    n = v.size
    p_le = (int((v <= null).sum()) + 1) / (n + 1)
    p_ge = (int((v >= null).sum()) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


# =============================================================================
# Unified paired-bootstrap engine
# =============================================================================

def paired_bootstrap_indices(
    n_by_group: dict[str, int],
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = SEED,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield ``n_bootstraps`` dicts of {group_key: index_array}.

    Each group is resampled independently; within a group every case
    is a draw-with-replacement. The same draw for ``group='MOVER'`` is
    shared across all MOVER-tested statistics in that iteration, which
    is what makes the bootstrap *paired*.
    """
    rng = np.random.default_rng(seed)
    groups = list(n_by_group.items())
    for _ in range(n_bootstraps):
        yield {g: rng.integers(0, n, size=n) for g, n in groups}


def paired_bootstrap(
    statistic_fn: Callable[[dict[str, np.ndarray] | None], dict[str, float]],
    n_by_group: dict[str, int],
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = SEED,
    ci_level: float = 0.95,
    include_point: bool = True,
) -> pd.DataFrame:
    """Central engine.

    ``statistic_fn(indices_or_None)`` is called once with ``None``
    (point estimate on the full data) and then ``n_bootstraps`` times
    with a dict mapping group→index-array. Each call must return a
    dict of named scalar statistics. The returned DataFrame has one
    row per named statistic with columns
    ``[statistic, point, ci_lower, ci_upper, bootstrap_p, n_bootstraps,
    seed, ci_level]``.
    """
    # Point estimate
    point = statistic_fn(None) if include_point else {}

    # Preallocate
    draws: dict[str, list[float]] = {k: [] for k in point} if point else None
    for draw_indices in paired_bootstrap_indices(n_by_group, n_bootstraps, seed):
        vals = statistic_fn(draw_indices)
        if draws is None:
            draws = {k: [] for k in vals}
        for k, v in vals.items():
            draws[k].append(v)

    rows = []
    for k, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        lo, hi = percentile_ci(arr, ci_level)
        rows.append({
            "statistic":    k,
            "point":        float(point.get(k, float("nan"))) if include_point else float("nan"),
            "bootstrap_median": float(np.nanmedian(arr)),
            "ci_lower":     lo,
            "ci_upper":     hi,
            "bootstrap_p_vs_zero": bootstrap_pvalue(arr, null=0.0),
            "n_bootstraps": n_bootstraps,
            "seed":         seed,
            "ci_level":     ci_level,
        })
    return pd.DataFrame(rows)


def ensure_tables_dir() -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    return TABLES_DIR
