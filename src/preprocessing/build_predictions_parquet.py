"""
Build the canonical external-validation predictions parquet.

Produces ``artifacts/predictions/external_validation_predictions.parquet``
with the schema defined in ``CLAUDE.md`` §5. This file is the single
source of truth for every Phase-1 statistical analysis in the JAMIA
revision (DeLong, bootstrap CIs, paradox-gap CIs, sex-stratified AUC,
etc.).

Inputs (all read from ``${CCPERIOP_DATA_ROOT}``; same env-var pattern
used by ``src/validation/verify_prediction_alignment.py``):

* Eight per-model prediction CSVs at
  ``derived/phase3/predictions/{model}_on_{target}_predictions.csv``
  with minimal schema ``y_true, y_prob_original, y_prob_recalibrated``.
* Two Phase-1 analysis-ready tables at
  ``derived/phase1/{inspire,mover}_analysis_ready.csv`` from which
  covariates (``surgery_id``, ``asa``, ``high_asa``, ``emergency``,
  ``age``, ``sex``) are joined by row position. Positional alignment is
  certified by ``verify_prediction_alignment.py``; this script re-asserts
  it via a redundant ``y_true`` equality check.

Precondition: run ``verify_prediction_alignment.py`` first. This script
will abort if the positional ``y_true`` cross-check fails.

``case_id`` derivation
----------------------

``case_id = sha256(str(surgery_id) + CASE_ID_SALT).hexdigest()[:16]``

The salt (``CASE_ID_SALT`` below) is **public** and intentionally
committed to the repo. Its purpose is to resist rainbow-table attacks
against known ``surgery_id`` values if someone with access to the raw
INSPIRE/MOVER dataset later obtained the parquet. A public salt still
accomplishes this because knowing the salt gives no advantage in
reversing a SHA-256 hash. Keeping it fixed in source guarantees that
any future re-run produces bit-identical ``case_id`` values, so the
Zenodo-published parquet stays referenceable across revisions.

``race_ethnicity``
------------------
Deliberately **not** emitted. Phase-1 did not extract ``PATIENT_RACE_C``
/ ``PATIENT_ETHNIC_C`` from MOVER ``patient_information.csv``, so the
column would be all-NULL and misleading. Race-stratified analysis needs
a small Phase-1 addendum (see CLAUDE.md §6 / JAMIA brief §2.2); the
parquet schema will be extended when that data is added.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = Path(
    os.environ.get(
        "CCPERIOP_DATA_ROOT",
        "./data/",
    )
)
PRED_DIR = DATA_ROOT / "derived" / "phase3" / "predictions"
PHASE1_DIR = DATA_ROOT / "derived" / "phase1"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "artifacts" / "predictions"
OUTPUT_PATH = OUTPUT_DIR / "external_validation_predictions.parquet"

CASE_ID_SALT = "ccperiop_jamia_2026"
CASE_ID_HEX_CHARS = 16  # 64 bits of namespace; ample for ~185K rows

EXPECTED_ROWS = {
    "INSPIRE": 127_413,
    "MOVER": 57_545,
}

COVARIATE_COLS = [
    "surgery_id",
    "mortality",
    "asa",
    "high_asa",
    "emergency",
    "age",
    "sex",
]


@dataclass(frozen=True)
class ModelSpec:
    model_name: str  # e.g. "XGB-INS-B"
    algorithm: Literal["XGBoost", "LogReg"]
    train_dataset: Literal["INSPIRE", "MOVER"]
    test_dataset: Literal["INSPIRE", "MOVER"]
    feature_set: Literal["preop", "intraop"]
    prediction_file: str


def _ms(algo_tag: str, train: str, test: str, variant: str) -> ModelSpec:
    algo_map = {"XGB": "XGBoost", "LR": "LogReg"}
    train_full = {"INS": "INSPIRE", "MOV": "MOVER"}
    feature_map = {"A": "preop", "B": "intraop"}
    name = f"{algo_tag}-{train}-{variant}"
    return ModelSpec(
        model_name=name,
        algorithm=algo_map[algo_tag],  # type: ignore[arg-type]
        train_dataset=train_full[train],  # type: ignore[arg-type]
        test_dataset=test,  # type: ignore[arg-type]
        feature_set=feature_map[variant],  # type: ignore[arg-type]
        prediction_file=f"{name}_on_{test}_predictions.csv",
    )


MODELS: list[ModelSpec] = [
    # INSPIRE-trained, tested on MOVER
    _ms("XGB", "INS", "MOVER", "A"),
    _ms("XGB", "INS", "MOVER", "B"),
    _ms("LR", "INS", "MOVER", "A"),
    _ms("LR", "INS", "MOVER", "B"),
    # MOVER-trained, tested on INSPIRE
    _ms("XGB", "MOV", "INSPIRE", "A"),
    _ms("XGB", "MOV", "INSPIRE", "B"),
    _ms("LR", "MOV", "INSPIRE", "A"),
    _ms("LR", "MOV", "INSPIRE", "B"),
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_parquet")


# =============================================================================
# Helpers
# =============================================================================

def hash_case_id(surgery_id: object) -> str:
    """SHA-256 of ``str(surgery_id) + CASE_ID_SALT``, truncated."""
    payload = (str(surgery_id) + CASE_ID_SALT).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:CASE_ID_HEX_CHARS]


def derive_asa_stratum(asa: pd.Series) -> pd.Series:
    """Map ASA (1–5/6) → {'ASA_1_2', 'ASA_3_plus'}. NaN → NaN."""
    out = pd.Series(pd.NA, index=asa.index, dtype="object")
    out[asa.isin([1, 2])] = "ASA_1_2"
    out[asa >= 3] = "ASA_3_plus"
    return out


def load_analysis_ready(dataset: str) -> pd.DataFrame:
    """Load the Phase-1 analysis-ready CSV for one dataset, pruned to
    the covariates we need and with ``case_id`` and ``asa_stratum``
    derived."""
    fname = f"{dataset.lower()}_analysis_ready.csv"
    path = PHASE1_DIR / fname
    log.info("loading %s", path)
    df = pd.read_csv(path, usecols=COVARIATE_COLS, low_memory=False)

    if len(df) != EXPECTED_ROWS[dataset]:
        raise AssertionError(
            f"{dataset} analysis_ready row count {len(df):,} != "
            f"expected {EXPECTED_ROWS[dataset]:,}"
        )

    # Derive case_id
    df["case_id"] = df["surgery_id"].map(hash_case_id)

    # Derive asa_stratum and cross-check against high_asa
    df["asa_stratum"] = derive_asa_stratum(df["asa"])

    # Cross-check: where asa is not null, high_asa == 1 iff asa >= 3.
    defined = df["asa"].notna()
    expected_high = (df.loc[defined, "asa"] >= 3).astype(int)
    actual_high = df.loc[defined, "high_asa"].astype(int)
    mismatch = (expected_high != actual_high).sum()
    if mismatch > 0:
        raise AssertionError(
            f"{dataset}: asa_stratum derivation disagrees with "
            f"high_asa for {mismatch:,} rows"
        )
    log.info(
        "%s: asa_stratum cross-check OK (n_defined=%d)",
        dataset, int(defined.sum()),
    )

    # Coerce `emergency` to bool where possible
    df["emergency"] = df["emergency"].astype("boolean")

    return df.rename(columns={"mortality": "y_true_ref"})


def load_predictions(spec: ModelSpec) -> pd.DataFrame:
    path = PRED_DIR / spec.prediction_file
    log.info("loading %s", path)
    df = pd.read_csv(path)
    missing = {"y_true", "y_prob_original", "y_prob_recalibrated"} - set(df.columns)
    if missing:
        raise AssertionError(f"{spec.prediction_file} missing columns: {missing}")
    return df


def assemble_one(spec: ModelSpec, covars: pd.DataFrame) -> pd.DataFrame:
    """Join one model's predictions to the test-dataset covariates by
    row position, assert y_true agreement, return the long-format
    slice."""
    preds = load_predictions(spec)

    if len(preds) != len(covars):
        raise AssertionError(
            f"{spec.model_name}: prediction rows {len(preds):,} != "
            f"covariate rows {len(covars):,}"
        )

    # Redundant positional-alignment assertion (belt & braces).
    n_mismatch = int(
        (preds["y_true"].to_numpy() != covars["y_true_ref"].to_numpy()).sum()
    )
    if n_mismatch > 0:
        raise AssertionError(
            f"{spec.model_name}: y_true mismatch between prediction CSV "
            f"and analysis_ready for {n_mismatch:,} rows"
        )

    out = pd.DataFrame({
        "case_id":           covars["case_id"].to_numpy(),
        "y_true":            preds["y_true"].astype("int8").to_numpy(),
        "y_pred_prob_raw":   preds["y_prob_original"].astype("float64").to_numpy(),
        "y_pred_prob_recal": preds["y_prob_recalibrated"].astype("float64").to_numpy(),
        "model_name":        spec.model_name,
        "train_dataset":     spec.train_dataset,
        "test_dataset":      spec.test_dataset,
        "feature_set":       spec.feature_set,
        "algorithm":         spec.algorithm,
        "asa_stratum":       covars["asa_stratum"].to_numpy(),
        "emergency":         covars["emergency"].to_numpy(),
        "age":               covars["age"].astype("float64").to_numpy(),
        "sex":               covars["sex"].to_numpy(),
    })
    return out


def summarise(df: pd.DataFrame) -> None:
    log.info("\n" + "=" * 64)
    log.info("PARQUET SUMMARY")
    log.info("=" * 64)

    log.info("Total rows: %s", f"{len(df):,}")
    log.info("dtypes:\n%s", df.dtypes.to_string())

    log.info("\nRow count per (model_name, test_dataset):")
    grp = df.groupby(["model_name", "test_dataset"], observed=True).size()
    log.info("\n%s", grp.to_string())

    log.info("\nRow count per asa_stratum:")
    strat = df["asa_stratum"].value_counts(dropna=False)
    log.info("\n%s", strat.to_string())

    log.info("\nMissing values per column:")
    miss = df.isna().sum()
    log.info("\n%s", miss.to_string())


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if not PRED_DIR.exists():
        log.error("prediction dir missing: %s — is DATA_ROOT mounted?", PRED_DIR)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    covars_by_test = {
        "MOVER":   load_analysis_ready("MOVER"),
        "INSPIRE": load_analysis_ready("INSPIRE"),
    }

    frames: list[pd.DataFrame] = []
    for spec in MODELS:
        covars = covars_by_test[spec.test_dataset]
        frames.append(assemble_one(spec, covars))

    full = pd.concat(frames, ignore_index=True, copy=False)

    # Cast string-ish columns to 'category' for on-disk compactness
    for col in ("model_name", "train_dataset", "test_dataset",
                "feature_set", "algorithm", "asa_stratum", "sex"):
        full[col] = full[col].astype("category")

    # Expected row count: 4 models × 57,545 (MOVER) + 4 × 127,413 (INSPIRE)
    expected_total = 4 * 57_545 + 4 * 127_413  # 739,832
    if len(full) != expected_total:
        raise AssertionError(
            f"Assembled row count {len(full):,} != expected {expected_total:,}"
        )

    log.info("writing %s", OUTPUT_PATH)
    full.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="zstd")

    summarise(full)

    log.info("\nPreview — first 3 rows:")
    log.info("\n%s", full.head(3).to_string())

    size_bytes = OUTPUT_PATH.stat().st_size
    log.info(
        "\nOK: wrote %s (%.2f MB, %s rows)",
        OUTPUT_PATH, size_bytes / 1_048_576, f"{len(full):,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
