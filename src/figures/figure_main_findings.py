#!/usr/bin/env python3
"""
main-text findings figures.

Produces:
- figure_paradox.pdf       (per-model overall vs within-stratum AUC; paradox gap)
- figure_asymmetry.pdf     (per-model degradation by training direction;
                             group-level asymmetry visible)

Source data:
- results/tables/paradox_gaps.csv
- results/tables/direction_asymmetry_per_model.csv
- results/tables/bootstrap_auc_differences.csv (for cross-direction CI)

Usage:
    python src/figures/figure_main_findings.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "Manuscript" / "Current" / "figures"
TABLES = REPO_ROOT / "results" / "tables"


# =========================================================================
# Figure fig:paradox — per-model overall vs within-stratum AUC
# =========================================================================
def fig_paradox():
    df = pd.read_csv(TABLES / "paradox_gaps.csv")
    df = df.sort_values(["train_dataset", "model_name"]).reset_index(drop=True)

    n = len(df)
    width = 0.36
    x = np.arange(n)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

    overall = df["overall_auc"].to_numpy()
    within = df["mean_within_stratum_auc"].to_numpy()
    gap = df["paradox_gap"].to_numpy()
    gap_lo = df["gap_ci_lower"].to_numpy()
    gap_hi = df["gap_ci_upper"].to_numpy()

    colors = ["#3a6e9a" if t == "INSPIRE" else "#a3553e" for t in df["train_dataset"]]
    fallback = df["used_fallback"].to_numpy()

    ax_top.bar(x - width/2, overall, width, label="Overall AUC",
               color=colors, alpha=0.55, edgecolor="black", linewidth=0.6)
    ax_top.bar(x + width/2, within, width, label="Within-stratum AUC (mean)",
               color=colors, alpha=0.95, edgecolor="black", linewidth=0.6,
               hatch="//")
    ax_top.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.55)
    ax_top.text(0.01, 0.51, "Random discrimination", transform=ax_top.get_yaxis_transform(),
                fontsize=8, color="#444",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
    ax_top.set_ylabel("AUC")
    ax_top.set_ylim(0.4, 1.0)
    ax_top.legend(loc="upper right", fontsize=10, frameon=False)
    ax_top.grid(axis="y", alpha=0.25, linewidth=0.5)

    # Mark fallback (small-n ASA 1-2 stratum) with a marker above
    for i, fb in enumerate(fallback):
        if fb:
            ax_top.text(i, 0.97, "*", ha="center", va="center", fontsize=14, color="#a3553e",
                        fontweight="bold")

    # Bottom: paradox gap with CIs
    err = [gap - gap_lo, gap_hi - gap]
    ax_bot.bar(x, gap * 100, width=0.6, color=colors, alpha=0.85, edgecolor="black",
               linewidth=0.6, yerr=np.array(err) * 100, capsize=3)
    ax_bot.set_ylabel("Paradox gap\n(pp)", fontsize=10)
    ax_bot.set_ylim(0, max(gap_hi * 100) + 2)
    ax_bot.grid(axis="y", alpha=0.25, linewidth=0.5)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(df["model_name"], rotation=22, ha="right", fontsize=10)

    # Group annotations
    for ax, ymax in [(ax_top, 0.42), (ax_bot, 0.5)]:
        pass

    fig.suptitle("Simpson's paradox gap by model (overall AUC vs mean within-stratum AUC)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = FIGURES_DIR / "figure_paradox.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}: gap range {gap.min()*100:.1f}-{gap.max()*100:.1f} pp")


# =========================================================================
# Figure fig:asymmetry — per-model degradation by training direction
# =========================================================================
def fig_asymmetry():
    df = pd.read_csv(TABLES / "direction_asymmetry_per_model.csv")
    # Order: 4 INSPIRE-trained on top; 4 MOVER-trained on bottom
    df["order"] = df["group"].map({"INSPIRE-trained": 0, "MOVER-trained": 1})
    df = df.sort_values(["order", "model"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))

    n = len(df)
    y = np.arange(n)
    deg = df["degradation"].to_numpy() * 100  # convert to percentage
    colors = ["#3a6e9a" if g == "INSPIRE-trained" else "#a3553e" for g in df["group"]]

    ax.barh(y, deg, color=colors, alpha=0.92, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.7)

    # Group means
    inspire_mean = df[df["group"] == "INSPIRE-trained"]["degradation"].mean() * 100
    mover_mean = df[df["group"] == "MOVER-trained"]["degradation"].mean() * 100
    ax.axvline(inspire_mean, color="#3a6e9a", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"INSPIRE-trained mean ({inspire_mean:.1f} pp)")
    ax.axvline(mover_mean, color="#a3553e", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"MOVER-trained mean ({mover_mean:.1f} pp)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("External AUC degradation\n(internal − external; pp)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=10, frameon=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)

    # Group separator
    ax.axhline(3.5, color="#888", linewidth=0.6, alpha=0.55)
    # : in-plot group labels were colliding with bar fills
    # at the right edge of the plot. Moved to right margin, rotated 270°
    # (read bottom-to-top), in axes-fraction coordinates so they sit cleanly
    # outside the plot area regardless of x-axis range.
    ax.annotate("INSPIRE-trained", xy=(1.02, 0.75), xycoords="axes fraction",
                rotation=270, va="center", ha="center",
                fontsize=10, color="#3a6e9a", fontweight="bold")
    ax.annotate("MOVER-trained", xy=(1.02, 0.25), xycoords="axes fraction",
                rotation=270, va="center", ha="center",
                fontsize=10, color="#a3553e", fontweight="bold")

    ax.set_title("Direction asymmetry: external AUC degradation by training cohort", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(right=0.92)

    out = FIGURES_DIR / "figure_asymmetry.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}: INSPIRE-mean {inspire_mean:.1f}pp, MOVER-mean {mover_mean:.1f}pp, ratio {mover_mean/inspire_mean:.2f}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_paradox()
    fig_asymmetry()


if __name__ == "__main__":
    main()
