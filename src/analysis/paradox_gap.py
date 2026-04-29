"""Paired-bootstrap CIs for the Simpson's paradox gap.

Uses the unified engine. Fallback rule for strata with fewer than
``MIN_EVENTS_STRATUM`` = 10 events: drop the unstable stratum and
compute ``gap = overall − ASA_3_plus`` instead of
``gap = overall − mean(ASA_1_2, ASA_3_plus)``. This matches the
manuscript's published handling of the 9-event MOVER ASA_1_2 stratum.

The LR-INS-B ASA_1_2 AUC of 0.858 is a real computation on 9 events but is statistically meaningless — 95% CI
(0.74, 0.97). It is reported in the CSV for completeness with an
``asa_1_2_reliability`` column flagging the small-n issue; it is
**not** used in the gap for that row (used_fallback=True).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ._bootstrap_utils import (
    DIRECTIONS, MODELS_BY_DIRECTION, N_BOOTSTRAPS, SEED,
    ensure_tables_dir, load_predictions, paired_bootstrap, safe_auc, wide_all,
)


MIN_EVENTS_STRATUM = 10


def run() -> pd.DataFrame:
    df = load_predictions()
    bundle = wide_all(df, "y_pred_prob_raw")

    # Fallback flag per direction — fixed once, not bootstrapped
    per_direction_fallback: dict[str, bool] = {}
    per_direction_events:   dict[str, tuple[int, int]] = {}
    for ts, b in bundle.items():
        asa = b["asa_stratum"]
        y = b["y_true"]
        n12 = int(((asa == "ASA_1_2") & (y == 1)).sum())
        n3  = int(((asa == "ASA_3_plus") & (y == 1)).sum())
        per_direction_fallback[ts] = n12 < MIN_EVENTS_STRATUM
        per_direction_events[ts] = (n12, n3)

    n_by_group = {ts: b["n"] for ts, b in bundle.items()}

    def stat_fn(idx: dict[str, np.ndarray] | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for ts, b in bundle.items():
            y = b["y_true"]
            asa = b["asa_stratum"]
            if idx is not None:
                ii = idx[ts]
                y_s, asa_s = y[ii], asa[ii]
            else:
                y_s, asa_s = y, asa
            for m in b["models"]:
                s = b["scores"][m].to_numpy()
                s_s = s if idx is None else s[idx[ts]]
                overall = safe_auc(y_s, s_s)
                mask12 = asa_s == "ASA_1_2"
                mask3  = asa_s == "ASA_3_plus"
                a12 = safe_auc(y_s[mask12], s_s[mask12])
                a3  = safe_auc(y_s[mask3],  s_s[mask3])
                if per_direction_fallback[ts]:
                    gap = overall - a3 if not (np.isnan(overall) or np.isnan(a3)) else float("nan")
                else:
                    mean_within = (a12 + a3) / 2 if not (np.isnan(a12) or np.isnan(a3)) else float("nan")
                    gap = overall - mean_within if not np.isnan(mean_within) else float("nan")
                out[f"overall|{m}|{ts}"] = overall
                out[f"asa12|{m}|{ts}"]   = a12
                out[f"asa3|{m}|{ts}"]    = a3
                out[f"gap|{m}|{ts}"]     = gap
        return out

    boot = paired_bootstrap(stat_fn, n_by_group,
                            n_bootstraps=N_BOOTSTRAPS, seed=SEED + 2)

    # Reshape per (model, test_dataset) row
    rows: list[dict] = []
    for ts, b in bundle.items():
        n12_events, n3_events = per_direction_events[ts]
        use_fallback = per_direction_fallback[ts]
        for m in b["models"]:
            overall_row = boot[boot["statistic"] == f"overall|{m}|{ts}"].iloc[0]
            a12_row     = boot[boot["statistic"] == f"asa12|{m}|{ts}"].iloc[0]
            a3_row      = boot[boot["statistic"] == f"asa3|{m}|{ts}"].iloc[0]
            gap_row     = boot[boot["statistic"] == f"gap|{m}|{ts}"].iloc[0]

            feature_set = "intraop" if m.endswith("-B") else "preop"
            definition = (
                "overall_minus_asa_3_plus_only (ASA_1_2 events<10)"
                if use_fallback else "overall_minus_mean(asa_1_2,asa_3_plus)"
            )
            gap = gap_row["point"]
            interp = (
                "Simpson's paradox (wide gap)" if not np.isnan(gap) and gap >= 0.10
                else "mild gap" if not np.isnan(gap) and gap >= 0.05
                else "no meaningful paradox"
            ) + (" (fallback)" if use_fallback else "")

            asa12_reliability = (
                "unreliable_small_n" if n12_events < MIN_EVENTS_STRATUM
                else "ok"
            )

            rows.append({
                "model_name":    m,
                "train_dataset": next(d[0] for d in DIRECTIONS if d[1] == ts),
                "test_dataset":  ts,
                "feature_set":   feature_set,
                "overall_auc":   overall_row["point"],
                "auc_asa_1_2":   a12_row["point"],
                "auc_asa_3_plus": a3_row["point"],
                "mean_within_stratum_auc":
                    (a3_row["point"] if use_fallback
                     else (a12_row["point"] + a3_row["point"]) / 2),
                "paradox_gap":   gap,
                "gap_ci_lower": gap_row["ci_lower"],
                "gap_ci_upper": gap_row["ci_upper"],
                "gap_bootstrap_p_vs_zero": gap_row["bootstrap_p_vs_zero"],
                "paradox_definition":  definition,
                "n_events_asa_1_2":    n12_events,
                "n_events_asa_3_plus": n3_events,
                "asa_1_2_reliability": asa12_reliability,
                "used_fallback":       use_fallback,
                "notes": (
                    "auc_asa_1_2 reported for completeness but not used in "
                    "gap; unreliable at n_events<10; unreliable at n_events < 10; reported for completeness only"
                    if asa12_reliability == "unreliable_small_n" else ""
                ),
                "interpretation": interp,
                "n_bootstraps":   N_BOOTSTRAPS,
                "seed":           SEED + 2,
            })

    out = pd.DataFrame(rows)
    out_path = ensure_tables_dir() / "paradox_gaps.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} models)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
