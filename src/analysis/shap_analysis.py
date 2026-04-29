"""SHAP analysis for the two best XGBoost models.

Scope:

* ``XGB-INS-B`` — trained on INSPIRE, best cross-continental performer
  (external AUC 0.895 on MOVER). SHAP on internal INSPIRE and external
  MOVER feature matrices.
* ``XGB-MOV-B`` — trained on MOVER, best MOVER-direction model
  (external AUC 0.812 on INSPIRE). SHAP on internal MOVER and external
  INSPIRE feature matrices.

LR models are skipped; linear-model SHAP is ``coef × feature_value``
and reduces to the model's own coefficients.

Outputs
-------

* ``artifacts/shap_values/{model}_on_{dataset}.npy`` — raw arrays
  (gitignored).
* ``results/figures/shap_summary_{model}_on_{dataset}.pdf`` + ``.png``
  — SHAP summary/beeswarm plots.
* ``results/tables/shap_importance_comparison.csv`` — per (model,
  feature) table of internal and external mean |SHAP|, with Spearman
  rank correlation between the two columns per model.
* ``results/tables/universally_transferable_features.csv`` — top-10
  features whose importance rank is preserved across internal and
  external evaluation for each model (criterion: in top 20 of both).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

from ._bootstrap_utils import REPO_ROOT, ensure_tables_dir


DATA_ROOT = Path(os.environ.get(
    "CCPERIOP_DATA_ROOT", "./data/",
))
MODELS_DIR = DATA_ROOT / "derived" / "phase2" / "models"
FEATURES_DIR = DATA_ROOT / "derived" / "phase2" / "features"

SHAP_DIR = REPO_ROOT / "artifacts" / "shap_values"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

SEX_ENCODING = {"Male": 0, "M": 0, "Female": 1, "F": 1}

# (model_name, train_features_csv, external_features_csv)
MODELS = [
    ("XGB-INS-B", "inspire_train_full.csv", "mover_train_full.csv",  "INSPIRE", "MOVER"),
    ("XGB-MOV-B", "mover_train_full.csv",   "inspire_train_full.csv", "MOVER",   "INSPIRE"),
]


def _apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    if "sex" in df.columns:
        df["sex"] = df["sex"].map(SEX_ENCODING).fillna(0).astype(int)
    for col in ("anesthesia_type", "department"):
        if col in df.columns and col in encoders:
            enc = encoders[col]
            unknown = enc.get("__unknown__", -99)
            df[col] = df[col].apply(
                lambda x, e=enc, u=unknown: e.get(x, u) if pd.notna(x) else -1
            ).astype(int)
    return df


def _load_model_and_data(model_name: str, features_csv: str):
    model_dir = MODELS_DIR / model_name
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "final_model.json"))
    with (model_dir / "feature_names.json").open() as f:
        feature_names = json.load(f)
    encoders: dict = {}
    enc_path = model_dir / "categorical_encoders.json"
    if enc_path.exists():
        with enc_path.open() as f:
            encoders = json.load(f)

    df = pd.read_csv(FEATURES_DIR / features_csv, low_memory=False)
    df = _apply_encoders(df, encoders)
    # Missing columns → NaN (XGBoost handles natively)
    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["mortality"].to_numpy(dtype=int) if "mortality" in df.columns else None
    return booster, feature_names, X, y


def _compute_shap(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(booster)
    return explainer.shap_values(X)


def _save_summary_plot(shap_vals: np.ndarray, X: np.ndarray,
                       feature_names: list[str],
                       model_name: str, dataset: str, top_k: int = 20):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig = plt.figure(figsize=(8, 9))
        shap.summary_plot(shap_vals, X, feature_names=feature_names,
                          max_display=top_k, show=False, plot_size=None)
        plt.title(f"SHAP summary — {model_name} on {dataset}\n"
                  f"(top {top_k} features; n={len(X):,})", fontsize=11)
        plt.tight_layout()
        out = FIGURES_DIR / f"shap_summary_{model_name}_on_{dataset}.{suffix}"
        plt.savefig(out, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out}")


def run() -> pd.DataFrame:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importance_rows: list[dict] = []
    transfer_rows: list[dict] = []

    for model_name, train_csv, ext_csv, train_ds, ext_ds in MODELS:
        print(f"\n--- {model_name} ---")

        booster, feat, X_int, _ = _load_model_and_data(model_name, train_csv)
        _,       _,    X_ext, _ = _load_model_and_data(model_name, ext_csv)

        print(f"  computing SHAP on internal ({train_ds}) n={len(X_int):,} ...")
        sv_int = _compute_shap(booster, X_int)
        print(f"  computing SHAP on external ({ext_ds}) n={len(X_ext):,} ...")
        sv_ext = _compute_shap(booster, X_ext)

        np.save(SHAP_DIR / f"{model_name}_on_{train_ds}.npy", sv_int)
        np.save(SHAP_DIR / f"{model_name}_on_{ext_ds}.npy",   sv_ext)

        _save_summary_plot(sv_int, X_int, feat, model_name, train_ds)
        _save_summary_plot(sv_ext, X_ext, feat, model_name, ext_ds)

        # Mean |SHAP| per feature
        mean_abs_int = np.mean(np.abs(sv_int), axis=0)
        mean_abs_ext = np.mean(np.abs(sv_ext), axis=0)

        imp = pd.DataFrame({
            "feature": feat,
            "mean_abs_shap_internal": mean_abs_int,
            "mean_abs_shap_external": mean_abs_ext,
        })
        imp["rank_internal"] = imp["mean_abs_shap_internal"].rank(ascending=False, method="min")
        imp["rank_external"] = imp["mean_abs_shap_external"].rank(ascending=False, method="min")
        imp["rank_delta"] = imp["rank_external"] - imp["rank_internal"]
        imp.insert(0, "model_name", model_name)
        imp.insert(1, "internal_dataset", train_ds)
        imp.insert(2, "external_dataset", ext_ds)

        rho, _ = spearmanr(mean_abs_int, mean_abs_ext)
        imp["spearman_rho_internal_vs_external"] = float(rho)
        importance_rows.append(imp)

        # Universally transferable: in top-20 by mean |SHAP| on BOTH
        top20_int = set(imp.nsmallest(20, "rank_internal")["feature"].tolist())
        top20_ext = set(imp.nsmallest(20, "rank_external")["feature"].tolist())
        stable = sorted(top20_int & top20_ext,
                        key=lambda f: imp.set_index("feature").loc[f, "rank_internal"])
        for f in stable[:10]:
            row = imp.set_index("feature").loc[f]
            transfer_rows.append({
                "model_name": model_name,
                "feature": f,
                "rank_internal": int(row["rank_internal"]),
                "rank_external": int(row["rank_external"]),
                "mean_abs_shap_internal": float(row["mean_abs_shap_internal"]),
                "mean_abs_shap_external": float(row["mean_abs_shap_external"]),
                "criterion": "top-20 in both internal and external",
            })

        print(f"  Spearman ρ(internal vs external mean |SHAP|) = {rho:.3f}")
        print(f"  {len(stable)} features are top-20 in both; reporting top 10")

    importance = pd.concat(importance_rows, ignore_index=True)
    imp_path = ensure_tables_dir() / "shap_importance_comparison.csv"
    importance.to_csv(imp_path, index=False)
    print(f"\n  wrote {imp_path}  ({len(importance)} rows)")

    transfer = pd.DataFrame(transfer_rows)
    tr_path = ensure_tables_dir() / "universally_transferable_features.csv"
    transfer.to_csv(tr_path, index=False)
    print(f"  wrote {tr_path}  ({len(transfer)} rows)")

    return importance


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
