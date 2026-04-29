"""60-min intraoperative window sensitivity analysis.

**Current status: BLOCKED by upstream feature extraction.**

The canonical intraoperative feature files at
``<DATA_ROOT>/derived/phase1/`` are

    inspire_intraop_features.csv
    mover_intraop_features.csv
    mover_intraop_features_BACKUP.csv
    the intraoperative-feature summary

No analogous *_30min or *_120min variants exist. Reproducing this
analysis requires re-running ``the intraoperative-feature extraction step`` with
two additional window widths, which reads the raw 50 M-row INSPIRE
``vitals.csv`` and all 19 MOVER ``flowsheet_part*.csv`` partitions.
Estimated compute: **4–6 hours** plus ~60 GB of scratch disk.

Two paths are available:

1. *Rerun Phase-1 for 30- and 120-min windows, then retrain XGB-INS-B
   and XGB-MOV-B for each (additional ~1 hour per model × 2 windows
   × 2 models = ~4 hours). Total: ~8–10 hours of compute. Produces a
   defensible sensitivity table for the revision.

2. *Argue the 60-min choice via the literature* — cite Fritz et al.
   and the MySurgeryRisk tradition (PMID 29381489) and move the
   sensitivity analysis to a Limitation paragraph.

Option 1 produces a defensible sensitivity table; option 2 cites the literature precedent.

This stub emits ``results/tables/window_sensitivity.csv`` with a single
row documenting the block so it is visible in ``paper_numbers.csv``.
"""

from __future__ import annotations

import sys

import pandas as pd

from ._bootstrap_utils import ensure_tables_dir


def run() -> pd.DataFrame:
    row = {
        "status": "BLOCKED_requires_phase1_rerun",
        "blocker": (
            "No 30-min or 120-min intraop feature files exist; "
            "only 60-min variant is cached at derived/phase1/"
        ),
        "remediation_option_1": (
            "Rerun the intraoperative-feature extraction step at 30 and 120 min "
            "(~4–6 h compute) then retrain XGB-INS-B and XGB-MOV-B "
            "for each (~2 additional hours); total ~8–10 h"
        ),
        "remediation_option_2": (
            "Argue 60-min choice via literature (Fritz 2019, "
            "MySurgeryRisk tradition); move sensitivity discussion "
            "to Limitations"
        ),
        "window_min": "n/a",
        "auc_xgb_ins_b_on_mover": float("nan"),
        "auc_xgb_mov_b_on_inspire": float("nan"),
    }
    out = pd.DataFrame([row])
    out_path = ensure_tables_dir() / "window_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  (BLOCKED — see file for options)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
