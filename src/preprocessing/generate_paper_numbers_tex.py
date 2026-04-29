"""Auto-generate ``results/paper_numbers.tex`` from ``paper_numbers.csv``.

One ``\\newcommand`` per numeric row plus a pre-formatted ``*Full`` macro
wherever the CSV already carries a composed display string (e.g. "0.895
(95% CI: 0.886–0.904)").

Naming convention
-----------------

Macros are prefixed ``\\pn`` (for *paper number*). The CSV ``id`` is
converted to CamelCase by splitting on ``_`` and ``-`` and capitalising
each segment. Examples::

    auc_XGB-INS-B_on_MOVER             -> \\pnAucXgbInsBOnMover
    paradox_gap_XGB-MOV-A_on_INSPIRE   -> \\pnParadoxGapXgbMovAOnInspire
    shap_spearman_rho_XGB-INS-B        -> \\pnShapSpearmanRhoXgbInsB

Four headline direction-asymmetry macros additionally emit the exact
names the manuscript references (matching the manuscript's ``\\pn`` macro convention)::

    \\pnDirectionAsymmetryDiff   = "+8.53 percentage points"
    \\pnDirectionAsymmetryCi     = "95% CI: +6.91 to +10.24 percentage points"
    \\pnDirectionAsymmetryP      = "p = 0.001"
    \\pnDirectionAsymmetryFull   = full sentence fragment

Table-1 regenerated values are emitted via ``\\pnTable1*`` macros.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from src.analysis._bootstrap_utils import RESULTS_DIR, TABLES_DIR


CSV_PATH = RESULTS_DIR / "paper_numbers.csv"
TEX_PATH = RESULTS_DIR / "paper_numbers.tex"

LATEX_ESCAPES = {
    "%": r"\%",
    "#": r"\#",
    "&": r"\&",
    "_": r"\_",
    "$": r"\$",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "–": "--",
    "—": "---",
    "±": r"$\pm$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "×": r"$\times$",
    "χ²": r"$\chi^2$",
    "ρ": r"$\rho$",
    "Δ": r"$\Delta$",
    "²": r"$^2$",
}


def latex_escape(s: str) -> str:
    """Escape for use inside a ``\\newcommand{...}`` body."""
    # Do symbols first (non-special). % etc. handled explicitly.
    for src, dst in LATEX_ESCAPES.items():
        s = s.replace(src, dst)
    return s


_DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def digits_to_words(s: str) -> str:
    """LaTeX macro names must be letters only (no digits). Replace each
    digit with its Title-case English word in place."""
    return "".join(_DIGIT_WORDS[c] if c.isdigit() else c for c in s)


def id_to_macro(id_: str) -> str:
    """Convert ``auc_XGB-INS-B_on_MOVER`` → ``pnAucXgbInsBOnMover``.

    Also replaces any digits with their word forms (LaTeX does not
    accept digits in ``\\newcommand`` macro names).
    """
    parts = re.split(r"[_\-]", id_)
    camel = "".join(p[:1].upper() + p[1:].lower() for p in parts if p)
    return "pn" + digits_to_words(camel)


def emit_value(macro: str, value: str) -> str:
    value = latex_escape(str(value))
    return f"\\newcommand{{\\{macro}}}{{{value}}}\n"


def emit_row(row: pd.Series) -> list[str]:
    """Return one or more ``\\newcommand`` lines for a CSV row."""
    macro = id_to_macro(str(row["id"]))
    lines: list[str] = []
    # Skip emission for do_not_cite / limitations_note rows (emit a
    # comment-line only).
    role = row.get("paper_role", "primary")
    if role == "do_not_cite":
        lines.append(
            f"% unreliable at n_events < 10; reported for completeness only — {row['id']}: {row['formatted']}\n"
        )
        return lines
    if role == "limitations_note":
        lines.append(
            f"% limitations-note only — {row['id']}: {row['formatted']}\n"
        )
        return lines

    # Raw value
    val = row["value"]
    if pd.notna(val):
        try:
            num = float(val)
            # Integer-ish values print without decimals
            if abs(num - round(num)) < 1e-9:
                lines.append(emit_value(macro, f"{int(round(num))}"))
            else:
                lines.append(emit_value(macro, f"{num:.4g}"))
        except (TypeError, ValueError):
            lines.append(emit_value(macro, str(val)))

    # Pre-formatted version
    if pd.notna(row["formatted"]) and str(row["formatted"]).strip():
        lines.append(emit_value(macro + "Full", str(row["formatted"])))

    return lines


# =============================================================================
# Table 1 macros
# =============================================================================

def emit_table1_macros() -> list[str]:
    """Pull the regenerated Table-1 numbers into stable \\pnTable1* macros."""
    path = TABLES_DIR / "table1_regenerated.csv"
    if not path.exists():
        return ["% Table 1 not regenerated yet — run src.analysis.table1_regen first\n"]
    t1 = pd.read_csv(path)
    lines: list[str] = ["% --- Table 1 (regenerated) -------------------------\n"]
    # Map characteristic → macro suffix. The ASA 1-2 / ≥3 rows occur
    # twice in Table 1 (once as the case-mix block, once as the deaths-
    # by-stratum block); state-based differentiation below.
    mapping = {
        "n (total cases)":              "N",
        "Age, years, median (IQR)":     "AgeMedianIqr",
        "Age, years, mean ± SD":        "AgeMeanSd",
        "Female":                       "Female",
        "Male":                         "Male",
        "BMI, kg/m² (mean ± SD)":       "Bmi",
        "ASA 1–2":                      "Asa12",
        "ASA ≥3":                       "Asa3Plus",
        "Emergency":                    "Emergency",
        "Elective":                     "Elective",
        "In-hospital mortality":        "Mortality",
    }
    in_deaths_block = False
    for _, row in t1.iterrows():
        key = row["characteristic"].strip()
        if key == "Deaths by ASA stratum":
            in_deaths_block = True
            continue
        if key not in mapping:
            continue
        suffix = mapping[key]
        if in_deaths_block and key in ("ASA 1–2", "ASA ≥3"):
            suffix = "Deaths" + suffix
        # digits in suffix (Asa12, Asa3Plus, DeathsAsa12, Table1) → words
        suffix = digits_to_words(suffix)
        ins = latex_escape(str(row["INSPIRE (n=127,413)"]))
        mov = latex_escape(str(row["MOVER (n=57,545)"]))
        lines.append(f"\\newcommand{{\\pnTableOne{suffix}Inspire}}{{{ins}}}\n")
        lines.append(f"\\newcommand{{\\pnTableOne{suffix}Mover}}{{{mov}}}\n")
    return lines


# =============================================================================
# Headline macros (exact names matching the manuscript's `\pn` macro convention)
# =============================================================================

def emit_direction_asymmetry_macros(pn: pd.DataFrame) -> list[str]:
    row = pn[pn["id"] == "direction_asymmetry_diff_pp"]
    if row.empty:
        return []
    row = row.iloc[0]
    formatted = str(row["formatted"])
    # e.g. "+8.53pp (95% CI: +6.91–+10.24pp; bootstrap p=0.0010)"
    m = re.search(
        r"([+-]?[\d.]+)pp\s+\(95%?\s*CI:\s*([+-]?[\d.]+)(?:pp)?\s*[–-]\s*"
        r"([+-]?[\d.]+)pp\s*;\s*bootstrap\s*p\s*=\s*([\d.]+)\s*\)",
        formatted,
    )
    if not m:
        return [f"% WARNING: could not parse direction_asymmetry string: "
                f"{formatted}\n"]
    diff, lo, hi, p = m.groups()
    diff_val = float(diff)
    lo_val, hi_val = float(lo), float(hi)
    p_val = float(p)
    # Format p-value: show as "0.001" if exactly at MC floor
    p_str = f"p < 0.001" if p_val <= 0.001 + 1e-9 else f"p = {p_val:.3f}"

    lines = [
        "\n% --- Headline direction-asymmetry macros ----------------------\n",
        f"\\newcommand{{\\pnDirectionAsymmetryDiff}}"
        f"{{{diff_val:+.2f} percentage points}}\n",
        f"\\newcommand{{\\pnDirectionAsymmetryCi}}"
        f"{{95\\% CI: {lo_val:+.2f} to {hi_val:+.2f} percentage points}}\n",
        f"\\newcommand{{\\pnDirectionAsymmetryP}}{{{p_str}}}\n",
        f"\\newcommand{{\\pnDirectionAsymmetryFull}}"
        f"{{{diff_val:+.2f} percentage points "
        f"(95\\% CI: {lo_val:+.2f} to {hi_val:+.2f}; bootstrap {p_str})}}\n",
    ]
    return lines


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} missing", file=sys.stderr)
        return 2
    pn = pd.read_csv(CSV_PATH)

    out_lines: list[str] = [
        "% Auto-generated by src/preprocessing/generate_paper_numbers_tex.py\n",
        "% DO NOT EDIT BY HAND — regenerate from results/paper_numbers.csv\n",
        f"% Row count: {len(pn)}\n\n",
    ]

    out_lines.append("% --- Auto-generated from paper_numbers.csv ---------\n")
    for _, row in pn.iterrows():
        out_lines.extend(emit_row(row))

    out_lines.extend(emit_direction_asymmetry_macros(pn))
    out_lines.extend(["\n"])
    out_lines.extend(emit_table1_macros())

    TEX_PATH.write_text("".join(out_lines))
    print(f"  wrote {TEX_PATH}  ({sum(1 for _ in open(TEX_PATH))} lines)")

    # Sanity check — confirm the 4 headline macros exist, and that
    # no macro name contains a digit (LaTeX would silently truncate).
    content = TEX_PATH.read_text()
    expected = [
        r"\pnDirectionAsymmetryDiffPp",
        r"\pnParadoxGapXgbMovAOnInspire",
        r"\pnShapSpearmanRhoXgbInsB",
        r"\pnMeanIntraopVsPreopAllEight",
    ]
    missing = [m for m in expected if m not in content]
    if missing:
        print(f"  WARNING: headline macros missing: {missing}")
    else:
        print("  all 4 headline macros present")

    bad = re.findall(r"\\newcommand\{\\(pn[A-Za-z]*\d+[A-Za-z]*)\}", content)
    if bad:
        print(f"  WARNING: {len(bad)} macro names still contain digits "
              f"(e.g. {bad[:3]}) — LaTeX will choke")
    else:
        print("  all macro names are digit-free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
