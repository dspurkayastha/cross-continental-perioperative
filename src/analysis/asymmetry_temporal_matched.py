"""Temporal-overlap matched sensitivity for the direction asymmetry.

**Status: BLOCKED by INSPIRE data anonymisation.**

The INSPIRE v1.3 release anonymises all timestamps as *Relative Time*
minutes from a reference shifted per-dataset (confirmed via
``schema.csv`` and the ``opdate`` value distribution: range
[-1,440, 5,184,000] minutes ≈ 10.07 years, all at day-level
granularity). No mapping to calendar dates is published. Consequently,
individual surgeries cannot be placed in calendar year 2015–2020 or
any other absolute interval.

MOVER does carry calendar ``SURGERY_DATE`` values. A temporal-overlap
analysis would require *both* datasets to expose calendar dates. One
direction cannot carry the analysis alone.

Options
-------

1. *Write to the INSPIRE curators* (Lim et al., Scientific Data 2024)
   to request a dataset-level reference date so published opdate values
   can be mapped to calendar time.
2. *Use subject-relative surgery order* as a proxy (e.g. pre-2018 vs
   post-2018 by ranking opdate within the dataset). This is *not*
   a temporal overlap and should not be reported as such.
3. *Accept the block*. Temporal-overlap sensitivity is precluded by INSPIRE's timestamp anonymisation scheme.

This stub emits a one-row CSV noting the block so the combined summary
table (``asymmetry_all_dimensions_summary.csv``) has a complete record.
"""

from __future__ import annotations

import sys

import pandas as pd

from ._bootstrap_utils import ensure_tables_dir


def run() -> pd.DataFrame:
    row = {
        "dimension":            "temporal_overlap_2015_2020",
        "framing":              "temporal_overlap",
        "framing_label":        "Restrict both cohorts to 2015–2020 inclusive",
        "status":               "BLOCKED_inspire_timestamps_anonymised",
        "blocker":              (
            "INSPIRE v1.3 distributes all timestamps as 'Relative Time' "
            "minutes from a shifted reference (schema.csv). Calendar "
            "dates cannot be recovered for INSPIRE surgeries; temporal "
            "overlap matching requires both datasets to expose absolute "
            "dates."
        ),
        "matched_asymmetry_pp": float("nan"),
        "ci_lower":             float("nan"),
        "ci_upper":             float("nan"),
        "bootstrap_p":          float("nan"),
        "n_inspire_test":       0,
        "n_mover_test":         0,
        "interpretation":       "not_applicable",
        "notes":                (
            "Not runnable: INSPIRE v1.3 distributes timestamps as 'Relative "
            "Time' minutes from a shifted reference, so calendar-date "
            "overlap matching is not possible with the current INSPIRE release."
        ),
    }
    out = pd.DataFrame([row])
    out_path = ensure_tables_dir() / "asymmetry_temporal_matched.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  (BLOCKED stub)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
