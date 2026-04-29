"""Regenerate Table 1 (baseline characteristics) from the canonical parquet.

Two outputs:
* ``results/tables/table1_regenerated.csv``  — raw numbers + test statistics.
* ``results/tables/table1_regenerated.tex`` — booktabs LaTeX for the
  manuscript table.

The canonical predictions parquet is the single source of truth for every
downstream number, so regenerating Table 1 here guarantees that the
manuscript's Table 1 and the Results narrative (which pulls from
``paper_numbers.csv``) share an identical denominator.

Deduplication
-------------

Each case appears 4× in the parquet (once per same-direction model).
We restrict to ``model_name == {first-of-direction}`` to get exactly n
unique cases per test dataset (57,545 MOVER, 127,413 INSPIRE), avoiding
the 3-collision issue with ``case_id`` in MOVER.

BMI
---

Not in the parquet. Joined positionally from the cohort analysis-ready
CSV — row order matches, cross-checked by ``verify_prediction_alignment.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from ._bootstrap_utils import (
    ensure_tables_dir, load_predictions, MODELS_BY_DIRECTION, DIRECTIONS,
)


DATA_ROOT = Path(os.environ.get(
    "CCPERIOP_DATA_ROOT", "./data/",
))
PHASE1_DIR = DATA_ROOT / "derived" / "phase1"

EXPECTED_N = {"INSPIRE": 127_413, "MOVER": 57_545}


def _one_row_per_case(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for train_ds, test_ds in DIRECTIONS:
        first_model = MODELS_BY_DIRECTION[(train_ds, test_ds)][0]
        sub = df[(df["test_dataset"] == test_ds) &
                 (df["model_name"] == first_model)].copy()
        sub["row_idx"] = np.arange(len(sub))
        if len(sub) != EXPECTED_N[test_ds]:
            raise AssertionError(
                f"{test_ds}: {len(sub):,} rows, expected {EXPECTED_N[test_ds]:,}"
            )
        out[test_ds] = sub
    return out


def _attach_bmi(case_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    fname = f"{dataset.lower()}_analysis_ready.csv"
    ar = pd.read_csv(PHASE1_DIR / fname, usecols=["bmi"], low_memory=False)
    if len(ar) != len(case_df):
        raise AssertionError(
            f"{dataset}: analysis_ready rows {len(ar):,} != parquet rows "
            f"{len(case_df):,}; positional join unsafe"
        )
    out = case_df.reset_index(drop=True).copy()
    out["bmi"] = ar["bmi"].to_numpy()
    return out


def _fmt_mean_sd(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.mean():.1f} ± {x.std():.1f}"


def _fmt_median_iqr(x: pd.Series) -> str:
    """Match the preprint's preferred summary form for age."""
    x = x.dropna()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.0f} ({q1:.0f}–{q3:.0f})"


def _fmt_n_pct(n: int, total: int) -> str:
    return f"{n:,} ({n / max(total, 1) * 100:.1f}%)"


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a = a.dropna(); b = b.dropna()
    stat, p = ttest_ind(a, b, equal_var=False)
    return float(stat), float(p)


def _chi2(table: list[list[int]]) -> tuple[float, float]:
    stat, p, _, _ = chi2_contingency(table)
    return float(stat), float(p)


def build_table1(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ins = bundle["INSPIRE"]
    mov = bundle["MOVER"]
    rows: list[dict] = []

    def _add(label: str, ins_val, mov_val, test: str = "",
             stat: float | None = None, p: float | None = None,
             indent: bool = False):
        rows.append({
            "characteristic": ("  " if indent else "") + label,
            "INSPIRE (n=127,413)": ins_val,
            "MOVER (n=57,545)":    mov_val,
            "test":                test,
            "statistic": f"{stat:.3f}" if stat is not None else "",
            "p_value":   f"{p:.2e}" if p is not None else "",
        })

    # ---- n ----------------------------------------------------------------
    _add("n (total cases)", f"{len(ins):,}", f"{len(mov):,}")

    # ---- Age --------------------------------------------------------------
    stat, p = _welch(ins["age"], mov["age"])
    _add("Age, years, median (IQR)",
         _fmt_median_iqr(ins["age"]), _fmt_median_iqr(mov["age"]),
         "Welch t", stat, p)
    _add("Age, years, mean ± SD",
         _fmt_mean_sd(ins["age"]), _fmt_mean_sd(mov["age"]),
         "Welch t", stat, p)

    # ---- Sex --------------------------------------------------------------
    n_ins_f = int((ins["sex"] == "Female").sum())
    n_ins_m = int((ins["sex"] == "Male").sum())
    n_mov_f = int((mov["sex"] == "Female").sum())
    n_mov_m = int((mov["sex"] == "Male").sum())
    stat, p = _chi2([[n_ins_f, n_ins_m], [n_mov_f, n_mov_m]])
    _add("Sex", "", "", "χ²", stat, p)
    _add("Female",
         _fmt_n_pct(n_ins_f, len(ins)),
         _fmt_n_pct(n_mov_f, len(mov)),
         indent=True)
    _add("Male",
         _fmt_n_pct(n_ins_m, len(ins)),
         _fmt_n_pct(n_mov_m, len(mov)),
         indent=True)

    # ---- BMI --------------------------------------------------------------
    stat, p = _welch(ins["bmi"], mov["bmi"])
    _add("BMI, kg/m² (mean ± SD)",
         _fmt_mean_sd(ins["bmi"]), _fmt_mean_sd(mov["bmi"]),
         "Welch t", stat, p)

    # ---- ASA --------------------------------------------------------------
    n_ins_low  = int((ins["asa_stratum"] == "ASA_1_2").sum())
    n_ins_high = int((ins["asa_stratum"] == "ASA_3_plus").sum())
    n_mov_low  = int((mov["asa_stratum"] == "ASA_1_2").sum())
    n_mov_high = int((mov["asa_stratum"] == "ASA_3_plus").sum())
    stat, p = _chi2([[n_ins_low, n_ins_high], [n_mov_low, n_mov_high]])
    _add("ASA physical status", "", "", "χ²", stat, p)
    _add("ASA 1–2",
         _fmt_n_pct(n_ins_low, len(ins)),
         _fmt_n_pct(n_mov_low, len(mov)),
         indent=True)
    _add("ASA ≥3",
         _fmt_n_pct(n_ins_high, len(ins)),
         _fmt_n_pct(n_mov_high, len(mov)),
         indent=True)

    # ---- Emergency --------------------------------------------------------
    n_ins_em = int((ins["emergency"] == True).sum())
    n_mov_em = int((mov["emergency"] == True).sum())
    stat, p = _chi2([
        [n_ins_em, len(ins) - n_ins_em],
        [n_mov_em, len(mov) - n_mov_em],
    ])
    _add("Surgery", "", "", "χ²", stat, p)
    _add("Emergency",
         _fmt_n_pct(n_ins_em, len(ins)),
         _fmt_n_pct(n_mov_em, len(mov)),
         indent=True)
    _add("Elective",
         _fmt_n_pct(len(ins) - n_ins_em, len(ins)),
         _fmt_n_pct(len(mov) - n_mov_em, len(mov)),
         indent=True)

    # ---- Mortality --------------------------------------------------------
    n_ins_d = int(ins["y_true"].sum())
    n_mov_d = int(mov["y_true"].sum())
    stat, p = _chi2([
        [n_ins_d, len(ins) - n_ins_d],
        [n_mov_d, len(mov) - n_mov_d],
    ])
    _add("In-hospital mortality",
         _fmt_n_pct(n_ins_d, len(ins)),
         _fmt_n_pct(n_mov_d, len(mov)),
         "χ²", stat, p)

    # Deaths within each ASA stratum (% of deaths in the stratum)
    ins_d_low  = int(ins[ins["asa_stratum"] == "ASA_1_2"]["y_true"].sum())
    ins_d_high = int(ins[ins["asa_stratum"] == "ASA_3_plus"]["y_true"].sum())
    mov_d_low  = int(mov[mov["asa_stratum"] == "ASA_1_2"]["y_true"].sum())
    mov_d_high = int(mov[mov["asa_stratum"] == "ASA_3_plus"]["y_true"].sum())
    _add("Deaths by ASA stratum", "", "", "", None, None)
    _add("ASA 1–2",
         f"{ins_d_low:,} ({ins_d_low / max(n_ins_d, 1) * 100:.1f}% of deaths; "
         f"{ins_d_low / max(n_ins_low, 1) * 100:.2f}% mortality)",
         f"{mov_d_low:,} ({mov_d_low / max(n_mov_d, 1) * 100:.1f}% of deaths; "
         f"{mov_d_low / max(n_mov_low, 1) * 100:.2f}% mortality)",
         indent=True)
    _add("ASA ≥3",
         f"{ins_d_high:,} ({ins_d_high / max(n_ins_d, 1) * 100:.1f}% of deaths; "
         f"{ins_d_high / max(n_ins_high, 1) * 100:.2f}% mortality)",
         f"{mov_d_high:,} ({mov_d_high / max(n_mov_d, 1) * 100:.1f}% of deaths; "
         f"{mov_d_high / max(n_mov_high, 1) * 100:.2f}% mortality)",
         indent=True)

    return pd.DataFrame(rows)


# =============================================================================
# LaTeX emission
# =============================================================================

def to_latex_booktabs(df: pd.DataFrame) -> str:
    """Emit a booktabs-style table. Special characters are LaTeX-safe."""
    def esc(s: str) -> str:
        return (s.replace("%", r"\%")
                 .replace("±", r"$\pm$")
                 .replace("²", r"$^2$")
                 .replace("–", "--")
                 .replace("≥", r"$\geq$")
                 .replace("χ²", r"$\chi^2$"))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Baseline characteristics of the INSPIRE and MOVER external-"
        r"validation cohorts. Between-dataset comparisons use Welch's $t$-test "
        r"for continuous variables and $\chi^2$ for categorical variables. "
        r"Regenerated from the canonical predictions parquet; see "
        r"\texttt{results/tables/table1\_regenerated.csv} for raw numbers.}",
        r"\label{tab:table1}",
        r"\begin{tabular}{lccrrr}",
        r"\toprule",
        r"Characteristic & INSPIRE & MOVER & Test & Statistic & $p$-value \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        cells = [
            esc(str(r["characteristic"])),
            esc(str(r["INSPIRE (n=127,413)"])),
            esc(str(r["MOVER (n=57,545)"])),
            esc(str(r["test"])),
            esc(str(r["statistic"])),
            esc(str(r["p_value"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def diff_vs_preprint(table: pd.DataFrame) -> list[str]:
    """Hard-coded preprint Table-1 values for a discrepancy report.

    Numbers taken directly from ``manuscript/jamia_intro_results.tex``
    ``\\label{tab:cohort}`` (the canonical Table 1).
    """
    preprint = {
        # label : (INSPIRE %, MOVER %)
        "Female":    (49.0, 52.3),   # preprint: 62,471 / 30,113
        "ASA 1–2":   (90.4, 36.5),   # preprint: 115,139 / 21,028
        "ASA ≥3":    (9.6,  63.5),   # preprint: 12,274 / 36,517
        "Emergency": (7.9,  15.6),   # preprint: 10,033 / 8,987
    }
    msgs: list[str] = []
    for _, r in table.iterrows():
        key = r["characteristic"].strip()
        if key in preprint:
            ins_cell = str(r["INSPIRE (n=127,413)"])
            mov_cell = str(r["MOVER (n=57,545)"])
            # pull leading pct from "n (x.y%)"
            def _pct(s: str) -> float | None:
                import re
                m = re.search(r"\(([\d.]+)%\)", s)
                return float(m.group(1)) if m else None
            ins_pct = _pct(ins_cell); mov_pct = _pct(mov_cell)
            exp_ins, exp_mov = preprint[key]
            for label, cur, exp in (("INSPIRE", ins_pct, exp_ins),
                                    ("MOVER",   mov_pct, exp_mov)):
                if cur is None:
                    continue
                delta = cur - exp
                if abs(delta) > 2.0:
                    msgs.append(f"  STOP: {label} {key}: preprint {exp}%, "
                                f"regen {cur}%, Δ={delta:+.1f}pp (>2pp)")
                elif abs(delta) > 0.5:
                    msgs.append(f"  flag: {label} {key}: preprint {exp}%, "
                                f"regen {cur}%, Δ={delta:+.1f}pp")
    return msgs


def main() -> int:
    df = load_predictions()
    bundle = _one_row_per_case(df)
    bundle["INSPIRE"] = _attach_bmi(bundle["INSPIRE"], "INSPIRE")
    bundle["MOVER"]   = _attach_bmi(bundle["MOVER"],   "MOVER")

    tbl = build_table1(bundle)
    out_csv = ensure_tables_dir() / "table1_regenerated.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    out_tex = ensure_tables_dir() / "table1_regenerated.tex"
    out_tex.write_text(to_latex_booktabs(tbl))
    print(f"  wrote {out_tex}")

    msgs = diff_vs_preprint(tbl)
    if msgs:
        print("\n  DISCREPANCIES vs preprint:")
        for m in msgs:
            print(m)
        if any("STOP" in m for m in msgs):
            print("\n  >2pp discrepancy — review before accepting Table 1")
            return 2
    else:
        print("\n  All tracked values within 0.5pp of preprint.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
