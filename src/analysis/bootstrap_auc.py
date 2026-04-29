"""Case-level paired-bootstrap 95% CIs for each of the 8 external AUCs.

Uses the unified engine in ``_bootstrap_utils.paired_bootstrap``.
One shared bootstrap draw per test-dataset group per iteration means
the 4 same-direction AUCs come from the identical case resample; this
is the "paired" part of paired bootstrap, required for tight CIs on
downstream AUC *differences*.

Raw and recalibrated columns are both reported; they should match to
machine precision because ROC-AUC is invariant under monotonic
Platt scaling.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION, N_BOOTSTRAPS, SEED,
    ensure_tables_dir, load_predictions, paired_bootstrap, safe_auc, wide_all,
)


def _make_stat_fn(bundle: dict, score_col: str):
    """Return a function that computes per-model AUCs on a given draw."""
    def _fn(idx_by_group: dict[str, np.ndarray] | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for test_ds, b in bundle.items():
            idx = idx_by_group[test_ds] if idx_by_group is not None else None
            y = b["y_true"] if idx is None else b["y_true"][idx]
            for m in b["models"]:
                s = b["scores"][m].to_numpy()
                s = s if idx is None else s[idx]
                out[f"{m}|{test_ds}|{score_col}"] = safe_auc(y, s)
        return out
    return _fn


def run() -> pd.DataFrame:
    df = load_predictions()
    bundle_raw = wide_all(df, "y_pred_prob_raw")
    bundle_rec = wide_all(df, "y_pred_prob_recal")

    n_by_group = {ts: b["n"] for ts, b in bundle_raw.items()}

    boot_raw = paired_bootstrap(
        _make_stat_fn(bundle_raw, "y_pred_prob_raw"), n_by_group,
        n_bootstraps=N_BOOTSTRAPS, seed=SEED,
    )
    boot_rec = paired_bootstrap(
        _make_stat_fn(bundle_rec, "y_pred_prob_recal"), n_by_group,
        n_bootstraps=N_BOOTSTRAPS, seed=SEED,  # same seed → paired draws
    )

    # Parse the statistic key back into (model, test, score_type) columns.
    def _parse(df_: pd.DataFrame, tag: str) -> pd.DataFrame:
        parts = df_["statistic"].str.split("|", expand=True)
        df_ = df_.copy()
        df_["model_name"]   = parts[0]
        df_["test_dataset"] = parts[1]
        df_["score_type"]   = tag
        return df_

    combined = pd.concat([
        _parse(boot_raw, "raw"),
        _parse(boot_rec, "recal"),
    ], ignore_index=True)

    # Wide form: one row per (model, test_dataset) with raw + recal CIs
    raw = combined[combined["score_type"] == "raw"].set_index(["model_name", "test_dataset"])
    rec = combined[combined["score_type"] == "recal"].set_index(["model_name", "test_dataset"])

    rows = []
    for (m, ts), r in raw.iterrows():
        c = rec.loc[(m, ts)]
        rows.append({
            "model_name":        m,
            "train_dataset":     next(d[0] for d in DIRECTIONS if d[1] == ts),
            "test_dataset":      ts,
            "auc_point_raw":     r["point"],
            "ci_lower_025_raw":  r["ci_lower"],
            "ci_upper_975_raw":  r["ci_upper"],
            "auc_point_recal":    c["point"],
            "ci_lower_025_recal": c["ci_lower"],
            "ci_upper_975_recal": c["ci_upper"],
            "auc_raw_minus_recal": r["point"] - c["point"],
            "method":        "paired-case-bootstrap-percentile",
            "n_bootstraps":  N_BOOTSTRAPS,
            "seed":          SEED,
        })
    out = pd.DataFrame(rows)

    out_path = ensure_tables_dir() / "bootstrap_auc_cis.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} models)")

    # Table-2 formatted copy
    table2 = out[[
        "model_name", "train_dataset", "test_dataset",
        "auc_point_raw", "ci_lower_025_raw", "ci_upper_975_raw",
    ]].rename(columns={
        "auc_point_raw":    "auc",
        "ci_lower_025_raw": "ci_lower",
        "ci_upper_975_raw": "ci_upper",
    })
    table2["auc_formatted"] = table2.apply(
        lambda r: f"{r['auc']:.3f} (95% CI: {r['ci_lower']:.3f}–{r['ci_upper']:.3f})",
        axis=1,
    )
    table2_path = ensure_tables_dir() / "table2_external_validation.csv"
    table2.to_csv(table2_path, index=False)
    print(f"  wrote {table2_path}")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
