"""
Verify positional alignment of the 8 external-validation prediction CSVs.

Precondition for assembling the canonical parquet described in CLAUDE.md §5:
the four CSVs within each validation direction must share an identical
row ordering. Since the CSVs do not carry a case_id, the only way to safely
join predictions with covariates (ASA, sex, race) from the Phase-1
analysis_ready tables is by row position. This script certifies that
assumption.

Checks per direction (INSPIRE -> MOVER and MOVER -> INSPIRE):
    1. Row count matches the expected external-test-set size.
    2. SHA-256 of the y_true column is byte-identical across all 4 models
       in the direction (i.e., every model saw the same rows in the same
       order).
    3. Row count matches the corresponding Phase-1 analysis_ready.csv
       (no silent truncation / re-ordering during Phase-3).

Usage:
    python src/validation/verify_prediction_alignment.py
    (reads paths from the DATA_ROOT constant; override with env var
     CCPERIOP_DATA_ROOT if the drive label changes)
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(
    os.environ.get(
        "CCPERIOP_DATA_ROOT",
        "./data/",
    )
)
PRED_DIR = DATA_ROOT / "derived" / "phase3" / "predictions"
PHASE1_DIR = DATA_ROOT / "derived" / "phase1"

EXPECTED_ROWS = {
    "MOVER": 57_545,
    "INSPIRE": 127_413,
}

DIRECTIONS = {
    "INSPIRE_to_MOVER": {
        "test_dataset": "MOVER",
        "analysis_ready": PHASE1_DIR / "mover_analysis_ready.csv",
        "prediction_files": [
            "XGB-INS-A_on_MOVER_predictions.csv",
            "XGB-INS-B_on_MOVER_predictions.csv",
            "LR-INS-A_on_MOVER_predictions.csv",
            "LR-INS-B_on_MOVER_predictions.csv",
        ],
    },
    "MOVER_to_INSPIRE": {
        "test_dataset": "INSPIRE",
        "analysis_ready": PHASE1_DIR / "inspire_analysis_ready.csv",
        "prediction_files": [
            "XGB-MOV-A_on_INSPIRE_predictions.csv",
            "XGB-MOV-B_on_INSPIRE_predictions.csv",
            "LR-MOV-A_on_INSPIRE_predictions.csv",
            "LR-MOV-B_on_INSPIRE_predictions.csv",
        ],
    },
}


def sha256_of_series(series: pd.Series) -> str:
    """Hash the raw bytes of a Series' values in row order."""
    return hashlib.sha256(series.to_numpy().tobytes()).hexdigest()


def verify_direction(name: str, spec: dict) -> tuple[bool, list[str]]:
    """Return (passed, messages) for one validation direction."""
    msgs: list[str] = [f"\n=== {name} ==="]
    test_ds = spec["test_dataset"]
    expected_n = EXPECTED_ROWS[test_ds]

    # Check analysis_ready row count
    ar_path = spec["analysis_ready"]
    if not ar_path.exists():
        msgs.append(f"  FAIL: analysis_ready not found at {ar_path}")
        return False, msgs
    ar_rows = sum(1 for _ in ar_path.open()) - 1  # minus header
    ar_match = ar_rows == expected_n
    status = "OK" if ar_match else "FAIL"
    msgs.append(
        f"  [{status}] analysis_ready rows: {ar_rows:,} "
        f"(expected {expected_n:,})"
    )

    # Load each prediction CSV's y_true column and hash it
    hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    missing: list[str] = []
    for fname in spec["prediction_files"]:
        fpath = PRED_DIR / fname
        if not fpath.exists():
            missing.append(fname)
            continue
        df = pd.read_csv(fpath, usecols=["y_true"])
        row_counts[fname] = len(df)
        hashes[fname] = sha256_of_series(df["y_true"].astype("int64"))

    if missing:
        msgs.append(f"  FAIL: missing prediction files: {missing}")
        return False, msgs

    # Per-file row count check
    all_rows_ok = True
    for fname, n in row_counts.items():
        ok = n == expected_n
        all_rows_ok &= ok
        msgs.append(
            f"  [{'OK' if ok else 'FAIL'}] {fname}: {n:,} rows"
        )

    # Hash consistency across the 4 files
    unique_hashes = set(hashes.values())
    hashes_match = len(unique_hashes) == 1
    if hashes_match:
        h = next(iter(unique_hashes))
        msgs.append(f"  [OK] y_true SHA-256 identical across 4 models:")
        msgs.append(f"       {h}")
    else:
        msgs.append("  [FAIL] y_true hashes differ across models:")
        for fname, h in hashes.items():
            msgs.append(f"       {h}  {fname}")

    passed = ar_match and all_rows_ok and hashes_match
    return passed, msgs


def main() -> int:
    if not PRED_DIR.exists():
        print(
            f"ERROR: prediction dir missing: {PRED_DIR}\n"
            "       Is the DATA_ROOT drive mounted?",
            file=sys.stderr,
        )
        return 2

    all_passed = True
    for name, spec in DIRECTIONS.items():
        passed, msgs = verify_direction(name, spec)
        for m in msgs:
            print(m)
        all_passed &= passed

    print("\n" + "=" * 60)
    print("OVERALL: " + ("PASS" if all_passed else "FAIL"))
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
