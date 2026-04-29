"""Sex-stratified external AUC with paired-bootstrap 95% CIs.

For each of the 8 model × test-dataset combinations, compute AUC
within {Male, Female} subgroups. 2,000 paired-case bootstraps share
the parent direction's resample, so Male-subgroup AUCs across the 4
same-direction models come from an internally-consistent draw.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION, N_BOOTSTRAPS, SEED,
    ensure_tables_dir, load_predictions, paired_bootstrap, safe_auc,
)


def run() -> pd.DataFrame:
    df = load_predictions()

    # Build per-test-dataset bundles that also carry a sex vector.
    bundles: dict[str, dict] = {}
    for train_ds, test_ds in DIRECTIONS:
        sub = df[df["test_dataset"] == test_ds].copy()
        sub["row_idx"] = sub.groupby("model_name", observed=True).cumcount()
        pivot = sub.pivot(index="row_idx", columns="model_name", values="y_pred_prob_raw")
        first = sub[sub["model_name"] == MODELS_BY_DIRECTION[(train_ds, test_ds)][0]].sort_values("row_idx")
        bundles[test_ds] = {
            "y_true": first["y_true"].to_numpy(),
            "sex":    first["sex"].astype(str).to_numpy(),
            "scores": pivot,
            "models": MODELS_BY_DIRECTION[(train_ds, test_ds)],
            "train_dataset": train_ds,
            "n":      len(first),
        }

    def stat_fn(idx: dict[str, np.ndarray] | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for ts, b in bundles.items():
            y = b["y_true"]
            sex = b["sex"]
            if idx is not None:
                ii = idx[ts]
                y, sex = y[ii], sex[ii]
            for m in b["models"]:
                s = b["scores"][m].to_numpy()
                s = s if idx is None else s[idx[ts]]
                for val in ("Female", "Male"):
                    mask = sex == val
                    out[f"{m}|{ts}|{val}"] = safe_auc(y[mask], s[mask])
        return out

    n_by_group = {ts: b["n"] for ts, b in bundles.items()}
    boot = paired_bootstrap(stat_fn, n_by_group, n_bootstraps=N_BOOTSTRAPS, seed=SEED + 4)

    rows = []
    for _, r in boot.iterrows():
        m, ts, sex_val = r["statistic"].split("|")
        # Population counts in the subgroup (full-data)
        b = bundles[ts]
        mask = b["sex"] == sex_val
        n_cases = int(mask.sum())
        n_events = int((b["y_true"][mask] == 1).sum())
        rows.append({
            "model_name":    m,
            "train_dataset": b["train_dataset"],
            "test_dataset":  ts,
            "sex":           sex_val,
            "n_cases":       n_cases,
            "n_events":      n_events,
            "auc":           r["point"],
            "ci_lower":      r["ci_lower"],
            "ci_upper":      r["ci_upper"],
            "bootstrap_median": r["bootstrap_median"],
            "n_bootstraps":  N_BOOTSTRAPS,
            "seed":          SEED + 4,
        })
    out = pd.DataFrame(rows).sort_values(["test_dataset", "model_name", "sex"]).reset_index(drop=True)

    out_path = ensure_tables_dir() / "sex_stratified_auc.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} rows: 8 models × 2 sexes)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
