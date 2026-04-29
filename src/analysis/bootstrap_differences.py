"""Paired-bootstrap CIs for AUC differences.

Two families of comparisons driven by the unified engine:

(a) Within-direction intraop − preop for each (algorithm, direction).
(b) Cross-direction mean_intraop − mean_preop across all 8 models.

Bootstrap p-values are two-sided and computed directly from the
bootstrap distribution of each difference (``bootstrap_p_vs_zero`` in
the engine output).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION, N_BOOTSTRAPS, SEED,
    ensure_tables_dir, load_predictions, paired_bootstrap, safe_auc, wide_all,
)


PAIRED_COMPARISONS: list[tuple[str, str, str, str]] = [
    ("INSPIRE", "MOVER",   "XGB-INS-A", "XGB-INS-B"),
    ("INSPIRE", "MOVER",   "LR-INS-A",  "LR-INS-B"),
    ("MOVER",   "INSPIRE", "XGB-MOV-A", "XGB-MOV-B"),
    ("MOVER",   "INSPIRE", "LR-MOV-A",  "LR-MOV-B"),
]

A_MODELS = ["XGB-INS-A", "LR-INS-A", "XGB-MOV-A", "LR-MOV-A"]
B_MODELS = ["XGB-INS-B", "LR-INS-B", "XGB-MOV-B", "LR-MOV-B"]

# Maps model → its test dataset (group key for resampling).
MODEL_TEST: dict[str, str] = {}
for (train_ds, test_ds), models in MODELS_BY_DIRECTION.items():
    for m in models:
        MODEL_TEST[m] = test_ds


def run() -> pd.DataFrame:
    df = load_predictions()
    bundle = wide_all(df, "y_pred_prob_raw")
    n_by_group = {ts: b["n"] for ts, b in bundle.items()}

    def _auc(model: str, idx: dict[str, np.ndarray] | None) -> float:
        ts = MODEL_TEST[model]
        b = bundle[ts]
        y = b["y_true"]
        s = b["scores"][model].to_numpy()
        if idx is not None:
            ii = idx[ts]
            y, s = y[ii], s[ii]
        return safe_auc(y, s)

    def stat_fn(idx: dict[str, np.ndarray] | None) -> dict[str, float]:
        stats: dict[str, float] = {}
        # Within-direction intraop − preop
        for _, _, a, b in PAIRED_COMPARISONS:
            stats[f"diff|{b}_minus_{a}"] = _auc(b, idx) - _auc(a, idx)
        # Cross-direction mean(B) − mean(A)
        a_aucs = [_auc(m, idx) for m in A_MODELS]
        b_aucs = [_auc(m, idx) for m in B_MODELS]
        stats["mean_aucs|mean_A"] = float(np.nanmean(a_aucs))
        stats["mean_aucs|mean_B"] = float(np.nanmean(b_aucs))
        stats["mean_diff|B_minus_A"] = float(np.nanmean(b_aucs) - np.nanmean(a_aucs))
        return stats

    boot = paired_bootstrap(stat_fn, n_by_group,
                            n_bootstraps=N_BOOTSTRAPS, seed=SEED + 1)

    # Shape the output like the old CSV.
    rows: list[dict] = []
    for train_ds, test_ds, model_a, model_b in PAIRED_COMPARISONS:
        row = boot[boot["statistic"] == f"diff|{model_b}_minus_{model_a}"].iloc[0]
        rows.append({
            "comparison": "intraop_minus_preop",
            "direction": f"{train_ds}_to_{test_ds}",
            "model_preop":  model_a,
            "model_intraop": model_b,
            "auc_preop":   _auc(model_a, None),
            "auc_intraop": _auc(model_b, None),
            "diff_point":  row["point"],
            "diff_median_bootstrap": row["bootstrap_median"],
            "ci_lower_025": row["ci_lower"],
            "ci_upper_975": row["ci_upper"],
            "bootstrap_p_vs_zero": row["bootstrap_p_vs_zero"],
            "n_bootstraps": N_BOOTSTRAPS,
            "seed": SEED + 1,
            "mean_auc_preop": float("nan"),
            "mean_auc_intraop": float("nan"),
        })
    cross = boot[boot["statistic"] == "mean_diff|B_minus_A"].iloc[0]
    rows.append({
        "comparison": "mean_intraop_minus_mean_preop_all8",
        "direction": "cross_direction",
        "model_preop":  None,
        "model_intraop": None,
        "auc_preop":   float("nan"),
        "auc_intraop": float("nan"),
        "diff_point":  cross["point"],
        "diff_median_bootstrap": cross["bootstrap_median"],
        "ci_lower_025": cross["ci_lower"],
        "ci_upper_975": cross["ci_upper"],
        "bootstrap_p_vs_zero": cross["bootstrap_p_vs_zero"],
        "n_bootstraps": N_BOOTSTRAPS,
        "seed": SEED + 1,
        "mean_auc_preop":   boot[boot["statistic"] == "mean_aucs|mean_A"].iloc[0]["point"],
        "mean_auc_intraop": boot[boot["statistic"] == "mean_aucs|mean_B"].iloc[0]["point"],
    })
    out = pd.DataFrame(rows)

    out_path = ensure_tables_dir() / "bootstrap_auc_differences.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} difference rows)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
