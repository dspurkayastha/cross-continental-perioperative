#!/usr/bin/env python3
"""
Figure S1 --- CONSORT-style cohort flow diagram.

Builds a side-by-side cohort attrition figure for the JAMIA supplementary
(§S1.3). Counts traced to:
- derived/phase1/
    the cohort-summary report  (Phase-1.4 attrition tables)
- Code/Phase1_Data_Extraction/the preoperative-feature extraction step
    (MOVER MRN+LOG_ID dedup at Phase-1.5a)

No raw patient data is read; all values are constants traced to the cohort
summary. The script is committed for reproducibility/auditability and
matches the \\pn macros in results/paper_numbers.tex ().

Usage:
    python src/figures/figure_S1_consort.py
    -> manuscript/figures/figure_S1_consort.pdf

, 2026-04-26.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PDF = REPO_ROOT / "Manuscript" / "Current" / "figures" / "figure_S1_consort.pdf"


# Per-cohort attrition. Counts must match the manuscript cohort summary in §S1.
INSPIRE_FLOW = [
    ("INSPIRE source release (PhysioNet v1.3)\n(Seoul National Univ. Hospital, 2011--2020)", 130_960, None),
    ("Excluded: missing ASA classification", None, 3_547),
    ("Inclusion: ASA 1--6 recorded;\nage and sex non-null", 127_413, None),
    ("Final analysis cohort", 127_413, None),
]

MOVER_FLOW = [
    ("MOVER source release (UC Irvine EPIC subset)\n(UC Irvine Med. Ctr., 2015--2022)", 65_728, None),
    ("Excluded: missing ASA (cath-lab/IR;\nno anesthesiologist assessment)", None, 6_970),
    ("Inclusion (post-ASA): ASA 1--6 recorded", 58_758, None),
    ("Excluded: missing discharge\ndisposition", None, 3),
    ("Inclusion: outcome ascertainable", 58_755, None),
    ("Excluded: duplicate (MRN, LOG\\_ID)\npairs after emergency-flag merge\n(deduplicated on first occurrence)", None, 1_210),
    ("Final analysis cohort", 57_545, None),
]


def draw_flow(ax, flow, title, color, exclusion_side="right"):
    """Render a single CONSORT column.

    : connectors now run inclusion -> next inclusion (skipping any
    exclusion-side branches), terminating on the actual box top edge so they
    visibly touch the boxes rather than ending in midair. Exclusion
    boxes branch outward via a short horizontal connector from the central
    flow line, and the column's `exclusion_side` controls which side the
    branch sits on (left for INSPIRE, right for MOVER) so each cohort's
    exclusions sit in its own outer margin.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(flow) * 2 + 1)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    box_x_left, box_x_right = 1.0, 9.0
    box_x_center = (box_x_left + box_x_right) / 2  # 5.0
    box_height = 1.4

    # Pre-compute box top y-coordinate per index; needed for connectors that
    # skip exclusion branches.
    y_top_at = {i: i * 2 + 0.4 for i in range(len(flow))}

    inclusion_indices = [i for i, (_, n, _) in enumerate(flow) if n is not None]

    # Exclusion branch geometry depends on which side of the column the
    # exclusions sit. INSPIRE = left margin; MOVER = right margin.
    # Side connector originates at the central flow line (box_x_center) so it
    # visually T-intersects with the inclusion-to-inclusion vertical arrow,
    # following standard CONSORT convention.
    if exclusion_side == "left":
        excl_x = -2.6
        excl_ha = "right"
        connector_from_x = box_x_center
        connector_to_x = excl_x
    else:  # "right"
        excl_x = 12.6
        excl_ha = "left"
        connector_from_x = box_x_center
        connector_to_x = excl_x

    for i, (label, n, excluded) in enumerate(flow):
        y_top = y_top_at[i]
        if n is not None:
            # Inclusion / final box (boxed)
            box = patches.FancyBboxPatch(
                (box_x_left, y_top), box_x_right - box_x_left, box_height,
                boxstyle="round,pad=0.08",
                linewidth=1.4,
                edgecolor="black",
                facecolor=color if i in (0, len(flow) - 1) else "white",
                alpha=0.18 if i in (0, len(flow) - 1) else 1.0,
            )
            ax.add_patch(box)
            ax.text(
                box_x_center, y_top + 0.55,
                f"{label}",
                ha="center", va="center", fontsize=10,
            )
            ax.text(
                box_x_center, y_top + 1.10,
                f"$n$ = {n:,}",
                ha="center", va="center", fontsize=10.5, fontweight="bold",
            )
        else:
            # Exclusion in outer margin; short horizontal connector from the
            # central flow line into the box anchor.
            ax.text(
                excl_x, y_top + 0.7,
                f"{label}\n($n$ = {excluded:,})",
                ha=excl_ha, va="center", fontsize=9, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f1d9",
                          edgecolor="#b59f55"),
            )
            ax.annotate(
                "",
                xy=(connector_to_x, y_top + 0.7), xycoords="data",
                xytext=(connector_from_x, y_top + 0.7), textcoords="data",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black",
                                shrinkA=0, shrinkB=2),
            )

    # Inclusion-to-inclusion vertical connectors. We draw one arrow per
    # consecutive pair of inclusion indices, terminating on the actual box
    # top y so the arrow is visibly anchored to the destination box.
    for src_i, dst_i in zip(inclusion_indices, inclusion_indices[1:]):
        src_y_bottom = y_top_at[src_i] + box_height
        dst_y_top = y_top_at[dst_i]
        ax.annotate(
            "",
            xy=(box_x_center, dst_y_top), xycoords="data",
            xytext=(box_x_center, src_y_bottom), textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="black",
                            shrinkA=0, shrinkB=0),
        )


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), gridspec_kw={"wspace": 0.35})

    draw_flow(axes[0], INSPIRE_FLOW, "INSPIRE (Korea)", "#3a6e9a",
              exclusion_side="left")
    draw_flow(axes[1], MOVER_FLOW, "MOVER (USA)", "#a3553e",
              exclusion_side="right")

    fig.suptitle(
        "Figure S1. CONSORT-style cohort flow",
        fontsize=14, fontweight="bold", y=0.995,
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    print(f"wrote {OUTPUT_PDF}  ({OUTPUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
