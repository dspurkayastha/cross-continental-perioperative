"""
Second-level integrity check for the canonical predictions parquet.

Reproduces three sets of numbers from the built parquet and checks them
against values reported in the manuscript:

(a) Overall external AUC for each of 8 (model_name, test_dataset) pairs.
    Target values come from manuscript Table 2 / medRxiv preprint.
    Tolerance: 0.001.

(b) Within-stratum AUC for XGB-MOV-A on INSPIRE — the canonical Simpson's
    paradox case from Table 3:
        ASA_1_2    = 0.597
        ASA_3_plus = 0.584
    Tolerance: 0.001.

(c) Sex distribution in each test dataset (INSPIRE, MOVER).

Fails loudly if any hard-coded expectation is missed by >0.001.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET = REPO_ROOT / "artifacts" / "predictions" / "external_validation_predictions.parquet"

TOL = 0.001

# Reference values from manuscript Table 2.
EXPECTED_OVERALL_AUC: dict[tuple[str, str], float] = {
    ("XGB-INS-A", "MOVER"):   0.785,
    ("XGB-INS-B", "MOVER"):   0.895,
    ("LR-INS-A",  "MOVER"):   0.796,
    ("LR-INS-B",  "MOVER"):   0.741,
    ("XGB-MOV-A", "INSPIRE"): 0.756,
    ("XGB-MOV-B", "INSPIRE"): 0.812,
    ("LR-MOV-A",  "INSPIRE"): 0.806,
    ("LR-MOV-B",  "INSPIRE"): 0.839,
}

# From manuscript Table 3 (Simpson's paradox case).
EXPECTED_SIMPSON = {
    "ASA_1_2":    0.597,
    "ASA_3_plus": 0.584,
}


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def check_overall(df: pd.DataFrame) -> bool:
    print("\n(a) Overall external AUC (raw y_pred_prob_raw)\n" + "-" * 70)
    print(f"{'model':<11} {'test':<8} {'computed':>10} {'expected':>10} "
          f"{'delta':>10} {'status':>7}")
    ok_all = True
    for (model, test), expected in EXPECTED_OVERALL_AUC.items():
        sub = df[(df["model_name"] == model) & (df["test_dataset"] == test)]
        computed = roc_auc_score(sub["y_true"], sub["y_pred_prob_raw"])
        delta = computed - expected
        ok = abs(delta) <= TOL
        ok_all &= ok
        print(f"{model:<11} {test:<8} {_fmt(computed):>10} "
              f"{expected:>10.3f} {delta:>+10.4f} {'PASS' if ok else 'FAIL':>7}")
    return ok_all


def check_simpson(df: pd.DataFrame) -> bool:
    print("\n(b) XGB-MOV-A on INSPIRE — within-stratum AUC "
          "(Simpson's paradox reproduction)\n" + "-" * 70)
    sub = df[(df["model_name"] == "XGB-MOV-A") & (df["test_dataset"] == "INSPIRE")]
    ok_all = True
    print(f"{'stratum':<12} {'n':>10} {'events':>8} "
          f"{'computed':>10} {'expected':>10} {'delta':>10} {'status':>7}")
    for stratum, expected in EXPECTED_SIMPSON.items():
        s = sub[sub["asa_stratum"] == stratum]
        n = len(s)
        events = int(s["y_true"].sum())
        if events == 0 or events == n:
            print(f"{stratum:<12} {n:>10,} {events:>8,} "
                  f"{'degenerate':>10} — cannot compute AUC")
            ok_all = False
            continue
        computed = roc_auc_score(s["y_true"], s["y_pred_prob_raw"])
        delta = computed - expected
        ok = abs(delta) <= TOL
        ok_all &= ok
        print(f"{stratum:<12} {n:>10,} {events:>8,} {_fmt(computed):>10} "
              f"{expected:>10.3f} {delta:>+10.4f} {'PASS' if ok else 'FAIL':>7}")
    return ok_all


def check_sex(df: pd.DataFrame) -> None:
    print("\n(c) Sex distribution in each test dataset\n" + "-" * 70)
    # Deduplicate — the parquet has 4 predictions per case, so sex values
    # are repeated 4×. Use one model_name per direction.
    one_per_direction = df[df["model_name"].isin(["XGB-INS-A", "XGB-MOV-A"])]
    for test in ("INSPIRE", "MOVER"):
        sub = one_per_direction[one_per_direction["test_dataset"] == test]
        vc = sub["sex"].value_counts(dropna=False)
        pct = (vc / vc.sum() * 100).round(1)
        print(f"\n  {test} (n={len(sub):,}):")
        for val, cnt in vc.items():
            print(f"    {str(val):<10} {cnt:>8,}  ({pct[val]:>4.1f}%)")


def main() -> int:
    if not PARQUET.exists():
        print(f"ERROR: parquet not found at {PARQUET}", file=sys.stderr)
        return 2

    df = pd.read_parquet(PARQUET, engine="pyarrow")
    print(f"loaded {PARQUET} — {len(df):,} rows, "
          f"{df['model_name'].nunique()} models")

    passed_a = check_overall(df)
    passed_b = check_simpson(df)
    check_sex(df)

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if (passed_a and passed_b) else 'FAIL'}")
    print("=" * 70)
    return 0 if (passed_a and passed_b) else 1


if __name__ == "__main__":
    sys.exit(main())
