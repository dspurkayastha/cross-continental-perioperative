"""Aggregate every matched-asymmetry framing into one summary table.

Reads:
* ``asymmetry_decomposition.csv``         (ASA case-mix, produced by
  case_mix_matched_asymmetry.py).
* ``_asymmetry_emergency_summary.csv``    (emergency, this turn).
* ``_asymmetry_comorbidity_summary.csv``  (comorbidity, if derivation
  succeeded this turn).
* ``asymmetry_temporal_matched.csv``      (temporal stub).

Writes ``results/tables/asymmetry_all_dimensions_summary.csv``.
"""

from __future__ import annotations

import sys

import pandas as pd

from ._bootstrap_utils import TABLES_DIR, ensure_tables_dir


SCHEMA = [
    "dimension", "framing", "framing_label",
    "matched_asymmetry_pp", "ci_lower", "ci_upper", "bootstrap_p",
    "pct_of_baseline", "interpretation",
    "n_inspire_test", "n_mover_test", "notes",
]


def _baseline() -> dict:
    return {
        "dimension":            "baseline",
        "framing":              "unmatched",
        "framing_label":        "Original direction asymmetry (full cohorts, unmatched)",
        "matched_asymmetry_pp": 0.08529513967311195,
        "ci_lower":             0.06906403,
        "ci_upper":             0.10239782,
        "bootstrap_p":          0.001,
        "pct_of_baseline":      100.0,
        "interpretation":       "baseline",
        "n_inspire_test":       127413,
        "n_mover_test":         57545,
        "notes":                "Reference point for all matched framings",
    }


def _asa_rows() -> pd.DataFrame:
    path = TABLES_DIR / "asymmetry_decomposition.csv"
    if not path.exists():
        return pd.DataFrame()
    asa = pd.read_csv(path)
    out = asa[asa["row"] != "original_unmatched"].copy()
    out = pd.DataFrame({
        "dimension":            "asa_case_mix",
        "framing":              out["row"].str.replace("framing_", "", regex=False),
        "framing_label":        out["description"],
        "matched_asymmetry_pp": out["asymmetry"],
        "ci_lower":             out["asymmetry_ci_lower"],
        "ci_upper":             out["asymmetry_ci_upper"],
        "bootstrap_p":          out["bootstrap_p"],
        "pct_of_baseline":      out["asymmetry"].abs() / abs(0.08529513967311195) * 100,
        "interpretation":       out["interpretation"],
        "n_inspire_test":       out["n_inspire"],
        "n_mover_test":         out["n_mover"],
        "notes":                "",
    })
    return out


def _load_or_empty(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run() -> pd.DataFrame:
    rows: list[pd.DataFrame] = [pd.DataFrame([_baseline()])]

    asa = _asa_rows()
    if not asa.empty:
        rows.append(asa[SCHEMA])

    for fname in ("_asymmetry_emergency_summary.csv",
                  "_asymmetry_comorbidity_summary.csv"):
        df_ = _load_or_empty(fname)
        if df_.empty:
            continue
        missing = [c for c in SCHEMA if c not in df_.columns]
        for m in missing:
            df_[m] = ""
        rows.append(df_[SCHEMA])

    tmp = _load_or_empty("asymmetry_temporal_matched.csv")
    if not tmp.empty:
        tmp["pct_of_baseline"] = float("nan")
        tmp["notes"] = tmp.get("notes", "").astype(str) + " | status=" + \
                       tmp.get("status", "").astype(str)
        rows.append(tmp[SCHEMA])

    out = pd.concat(rows, ignore_index=True)
    out_path = ensure_tables_dir() / "asymmetry_all_dimensions_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} rows)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
