"""Race/ethnicity stratification for the MOVER direction.

**Current status: BLOCKED by data availability.**

The MOVER EPIC EMR release distributed on UCI does not include
``PATIENT_RACE_C`` or ``PATIENT_ETHNIC_C`` in any of its 9
per-table CSVs. Column headers:

    patient_information.csv  → LOG_ID, MRN, DISCH_DISP_C, DISCH_DISP,
      HOSP_ADMSN_TIME, HOSP_DISCH_TIME, LOS, ICU_ADMIN_FLAG,
      SURGERY_DATE, BIRTH_DATE, HEIGHT, WEIGHT, SEX, PRIMARY_ANES_TYPE_NM,
      ASA_RATING_C, ASA_RATING, PATIENT_CLASS_GROUP, PATIENT_CLASS_NM,
      PRIMARY_PROCEDURE_NM, IN_OR_DTTM, OUT_OR_DTTM, AN_START_DATETIME,
      AN_STOP_DATETIME

No race column, no ethnicity column, in any file.

This is not an extraction gap; it is a deliberate omission in the UCI
release, consistent with its de-identification posture. Race-stratified
external validation is therefore not possible without an updated MOVER
release that includes self-reported race and ethnicity.

This stub emits ``results/tables/race_stratified_auc_mover_only.csv``
with a single row documenting the block so the row appears in the
aggregate paper-numbers review.
"""

from __future__ import annotations

import sys

import pandas as pd

from ._bootstrap_utils import ensure_tables_dir


def run() -> pd.DataFrame:
    row = {
        "status": "BLOCKED_data_unavailable",
        "blocker": (
            "PATIENT_RACE_C and PATIENT_ETHNIC_C are not present in the "
            "MOVER EPIC EMR release used for this study"
        ),
        "remediation": (
            "race-stratified analysis is not possible without an updated MOVER release"
        ),
        "n_cases": 0,
        "n_events": 0,
        "auc": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
    }
    out = pd.DataFrame([row])
    out_path = ensure_tables_dir() / "race_stratified_auc_mover_only.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  (BLOCKED — see file for reason)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
