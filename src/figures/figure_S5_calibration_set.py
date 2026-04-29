#!/usr/bin/env python3
"""
§S5 calibration cluster figures + main fig:dca.

Produces:
- figure_S5a_calib_slope.pdf  (per-model calibration slope pre/post Platt)
- figure_S5b_oe_ratio.pdf      (per-model O:E ratio pre/post Platt; log-y)
- figure_S5c_calib_curves.pdf  (8-panel reliability diagrams pre-recal)
- figure_S5d_dca_per_direction.pdf (8-panel DCA at 2-10% threshold)
- figure_dca_main.pdf          (main fig:dca: aggregated net-benefit)

Source data:
- <DATA_ROOT>/derived/phase3/phase3_results.json
  (per-model original + recalibrated calibration metrics, including CIs)
- <DATA_ROOT>/derived/phase3/predictions/*.csv
  (per-model y_true + y_prob_original + y_prob_recalibrated)
- results/tables/dca_2_10_threshold.csv

All caption-relevant numerical values derive from these canonical outputs;
where main-text uses a \\pn macro, this script computes the same value
from the same source.

Usage:
    python src/figures/figure_S5_calibration_set.py
"""

from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "Manuscript" / "Current" / "figures"
DATA_ROOT = Path(os.environ.get("CCPERIOP_DATA_ROOT", "./data/"))
PHASE3 = DATA_ROOT / "derived" / "phase3"
RESULTS_TABLES = REPO_ROOT / "results" / "tables"

MODEL_ORDER = [
    "XGB-INS-A_on_MOVER",
    "XGB-INS-B_on_MOVER",
    "LR-INS-A_on_MOVER",
    "LR-INS-B_on_MOVER",
    "XGB-MOV-A_on_INSPIRE",
    "XGB-MOV-B_on_INSPIRE",
    "LR-MOV-A_on_INSPIRE",
    "LR-MOV-B_on_INSPIRE",
]

# Figure S5d (DCA per-direction) panel order: LR-INS-B placed top-left to
# bring the focal deployment-failure panel into the F-pattern reading zone.
MODEL_ORDER_DCA = [
    "LR-INS-B_on_MOVER",
    "XGB-INS-B_on_MOVER",
    "XGB-INS-A_on_MOVER",
    "LR-INS-A_on_MOVER",
    "XGB-MOV-A_on_INSPIRE",
    "XGB-MOV-B_on_INSPIRE",
    "LR-MOV-A_on_INSPIRE",
    "LR-MOV-B_on_INSPIRE",
]
SHORT_NAMES = {k: k.split("_on_")[0] for k in MODEL_ORDER}
DIRECTION_COLOR = {True: "#3a6e9a", False: "#a3553e"}  # True = INSPIRE-trained


def is_inspire_trained(model_id):
    return "INS" in model_id.split("_on_")[0]


def load_phase3():
    with (PHASE3 / "phase3_results.json").open() as f:
        return json.load(f)


# =========================================================================
# Figure S5a — calibration slope pre/post Platt
# =========================================================================
def fig_s5a_slope(results):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    n = len(MODEL_ORDER)
    width = 0.38
    x = np.arange(n)

    pre = [results[k]["original"]["calibration_slope"] for k in MODEL_ORDER]
    post = [results[k]["recalibrated"]["calibration_slope"] for k in MODEL_ORDER]

    pre_ci = [results[k]["original"]["calibration_slope_ci"] for k in MODEL_ORDER]
    post_ci = [results[k]["recalibrated"]["calibration_slope_ci"] for k in MODEL_ORDER]

    pre_err = [[v - lo for v, (lo, hi) in zip(pre, pre_ci)],
               [hi - v for v, (lo, hi) in zip(pre, pre_ci)]]
    post_err = [[v - lo for v, (lo, hi) in zip(post, post_ci)],
                [hi - v for v, (lo, hi) in zip(post, post_ci)]]

    ax.bar(x - width/2, pre, width, label="Pre-recalibration",
           yerr=pre_err, capsize=2.5, color="#c2a87a", alpha=0.92,
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, post, width, label="Post Platt scaling (5-fold CV)",
           yerr=post_err, capsize=2.5, color="#3a6e9a", alpha=0.92,
           edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.02, 1.04, "Ideal slope = 1.0", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[k] for k in MODEL_ORDER], rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("Calibration slope (logit-link regression coefficient)")
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out = FIGURES_DIR / "figure_S5a_calib_slope.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}: pre range [{min(pre):.3f}, {max(pre):.3f}]; post range [{min(post):.3f}, {max(post):.3f}]")


# =========================================================================
# Figure S5b — O:E ratio pre/post Platt (log-y)
# =========================================================================
def fig_s5b_oe(results):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    n = len(MODEL_ORDER)
    width = 0.38
    x = np.arange(n)

    pre = [results[k]["original"]["oe_ratio"] for k in MODEL_ORDER]
    post = [results[k]["recalibrated"]["oe_ratio"] for k in MODEL_ORDER]
    pre_ci = [results[k]["original"]["oe_ratio_ci"] for k in MODEL_ORDER]
    post_ci = [results[k]["recalibrated"]["oe_ratio_ci"] for k in MODEL_ORDER]

    pre_err = [[v - lo for v, (lo, hi) in zip(pre, pre_ci)],
               [hi - v for v, (lo, hi) in zip(pre, pre_ci)]]
    post_err = [[v - lo for v, (lo, hi) in zip(post, post_ci)],
                [hi - v for v, (lo, hi) in zip(post, post_ci)]]

    ax.bar(x - width/2, pre, width, label="Pre-recalibration",
           yerr=pre_err, capsize=2.5, color="#c2a87a", alpha=0.92,
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, post, width, label="Post Platt scaling (5-fold CV)",
           yerr=post_err, capsize=2.5, color="#3a6e9a", alpha=0.92,
           edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[k] for k in MODEL_ORDER], rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("O:E ratio (log scale; observed / expected events)")
    ax.set_ylim(0.01, 5)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(axis="y", which="both", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out = FIGURES_DIR / "figure_S5b_oe_ratio.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}: pre range [{min(pre):.3f}, {max(pre):.3f}]; post range [{min(post):.3f}, {max(post):.3f}]")


# =========================================================================
# Figure S5c — pre-recalibration reliability diagrams (8 panels)
# =========================================================================
def fig_s5c_curves(results):
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.5), sharex=True, sharey=True)

    for ax, k in zip(axes.flat, MODEL_ORDER):
        # Load per-model predictions CSV
        short = SHORT_NAMES[k]
        # File format: <MODEL>_on_<DATASET>_predictions.csv
        # Where DATASET is the test_dataset (extracted from key)
        test_ds = k.split("_on_")[1]
        pred_csv = PHASE3 / "predictions" / f"{short}_on_{test_ds}_predictions.csv"
        if not pred_csv.exists():
            ax.text(0.5, 0.5, f"missing\n{pred_csv.name}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
            continue
        df = pd.read_csv(pred_csv)
        y = df["y_true"].to_numpy()
        p = df["y_prob_original"].to_numpy()

        # Decile binning
        deciles = pd.qcut(p, q=10, labels=False, duplicates="drop")
        dfb = pd.DataFrame({"y": y, "p": p, "bin": deciles})
        bin_summary = dfb.groupby("bin").agg(
            mean_pred=("p", "mean"),
            obs_rate=("y", "mean"),
            n=("y", "size"),
        ).reset_index()

        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.scatter(bin_summary["mean_pred"], bin_summary["obs_rate"],
                   s=18 + bin_summary["n"] / max(bin_summary["n"]) * 30,
                   color=DIRECTION_COLOR[is_inspire_trained(k)],
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.plot(bin_summary["mean_pred"], bin_summary["obs_rate"],
                color=DIRECTION_COLOR[is_inspire_trained(k)], linewidth=0.8, alpha=0.75)

        ax.set_title(short, fontsize=10.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.2, linewidth=0.4)

    for ax in axes[1, :]:
        ax.set_xlabel("Mean predicted probability (decile)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed event rate", fontsize=10)

    fig.suptitle("Pre-recalibration reliability diagrams (8 external validation runs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = FIGURES_DIR / "figure_S5c_calib_curves.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# =========================================================================
# Figure S5d — DCA per-direction at 2-10% threshold (8 panels)
# Figure fig:dca (main) — aggregate net benefit
# =========================================================================
def fig_dca(results):
    csv_path = RESULTS_TABLES / "dca_2_10_threshold.csv"
    df = pd.read_csv(csv_path)

    # ---- supp 8-panel ----
    # rebuild: same recipe as main fig_dca () propagated
    # across all 8 panels — treat-all dropped (was forcing y-axis to span
    # 0..-0.10 and squashing model curves to indistinguishable y~0); per-panel
    # legend removed in favor of a single shared figure-level legend below
    # the grid (was overlying the model curve in the LR-INS-B focal panel).
    # Panel order: LR-INS-B placed top-left (focal deployment-failure panel;
    # F-pattern attention zone). See MODEL_ORDER_DCA above.
    fig_s, axes = plt.subplots(2, 4, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, model_key in zip(axes.flat, MODEL_ORDER_DCA):
        model_short = SHORT_NAMES[model_key]
        sub = df[df["model_id"] == model_short]
        if sub.empty:
            ax.text(0.5, 0.5, f"missing\n{model_short}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
            continue
        sub = sub.sort_values("threshold")
        thr = sub["threshold"].to_numpy() * 100  # percent
        nb_model = sub["net_benefit_model"].to_numpy()

        is_ins = "INS" in model_short
        ax.plot(thr, nb_model, color=DIRECTION_COLOR[is_ins], linewidth=1.6,
                label="Model (recalibrated)")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        # Highlight LR-INS-B as the focal deployment-failure panel.
        is_focal = (model_short == "LR-INS-B")
        if is_focal:
            for spine in ax.spines.values():
                spine.set_edgecolor("#a3553e")
                spine.set_linewidth(2.0)
            ax.set_title(model_short + "  (focal: net-benefit loss above 2%)",
                         fontsize=10.5, fontweight="bold", color="#a3553e")
        else:
            ax.set_title(model_short, fontsize=10.5)
        ax.set_xlim(2, 10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.2, linewidth=0.4)

    for ax in axes[1, :]:
        ax.set_xlabel("Threshold probability (%)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Net benefit", fontsize=10)
    fig_s.legend(["Model (recalibrated)"], loc="lower center",
                 bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=10,
                 frameon=False)
    fig_s.suptitle("Decision curve analysis at 2–10% threshold (per-direction breakdown)",
                   fontsize=12)
    fig_s.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_s = FIGURES_DIR / "figure_S5d_dca_per_direction.pdf"
    fig_s.savefig(out_s, format="pdf", bbox_inches="tight")
    plt.close(fig_s)
    print(f"wrote {out_s.name}")

    # ---- main aggregated ----
    # rebuild: treat-all dropped (visible in supp
    # Figure S5d / S7); auto y-scale lets model curves separate visibly;
    # legend moved below x-axis label to avoid overlying model curves.
    fig_m, ax = plt.subplots(figsize=(7, 3.6))
    intraop_models = [m for m in df["model_id"].unique() if m.endswith("-B")]
    preop_models = [m for m in df["model_id"].unique() if m.endswith("-A")]

    for label, models, color in [
        ("Intraoperative models", intraop_models, "#3a6e9a"),
        ("Preoperative-only models", preop_models, "#a3553e"),
    ]:
        sub = df[df["model_id"].isin(models)]
        agg = sub.groupby("threshold")["net_benefit_model"].mean().reset_index()
        agg = agg.sort_values("threshold")
        ax.plot(agg["threshold"] * 100, agg["net_benefit_model"], color=color,
                linewidth=2.0, label=label)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_xlim(2, 10)
    ax.set_xlabel("Threshold probability (%)", fontsize=10)
    ax.set_ylabel("Net benefit", fontsize=10)
    ax.set_title("Aggregated decision curve analysis (2–10% threshold)", fontsize=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=2, fontsize=10, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig_m.tight_layout()
    out_m = FIGURES_DIR / "figure_dca_main.pdf"
    fig_m.savefig(out_m, format="pdf", bbox_inches="tight")
    plt.close(fig_m)
    print(f"wrote {out_m.name}")


# =========================================================================
# Figures S5e / S5f — copy SHAP cached PDFs
# =========================================================================
def copy_shap():
    import shutil
    src_e = REPO_ROOT / "results" / "figures" / "shap_summary_XGB-INS-B_on_MOVER.pdf"
    src_f = REPO_ROOT / "results" / "figures" / "shap_summary_XGB-MOV-B_on_INSPIRE.pdf"
    dst_e = FIGURES_DIR / "figure_S5e_shap_xgb_ins_b.pdf"
    dst_f = FIGURES_DIR / "figure_S5f_shap_xgb_mov_b.pdf"
    shutil.copy2(src_e, dst_e)
    shutil.copy2(src_f, dst_f)
    print(f"copied SHAP {src_e.name} -> {dst_e.name}")
    print(f"copied SHAP {src_f.name} -> {dst_f.name}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_phase3()
    fig_s5a_slope(results)
    fig_s5b_oe(results)
    fig_s5c_curves(results)
    fig_dca(results)
    copy_shap()


if __name__ == "__main__":
    main()
