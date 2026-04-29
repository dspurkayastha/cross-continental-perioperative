"""Class imbalance sensitivity for XGB-INS-B.

Retrains the best-performing model (XGB-INS-B) with three class-
imbalance handling strategies, holding median hyperparameters fixed
so the comparison isolates the imbalance treatment:

(i)   **No weighting** — ``scale_pos_weight = 1``.
(ii)  **Class weighting (main analysis equivalent)** —
      ``scale_pos_weight = 9.734`` (median of the 10 per-fold tuned
      values for XGB-INS-B, per ``hyperparameters.json``).
(iii) **SMOTE oversampling** — ``scale_pos_weight = 1``, minority
      class oversampled to parity via SMOTE before training.

Each config produces a single final model trained on the full INSPIRE
cohort (no held-out split — identical to the main-analysis final-model
protocol) and is externally validated on MOVER. Discrimination (AUC)
and calibration (Brier, slope, intercept) are reported.

Expected result (per Carriero et al. 2025, Methods §2.3): class
weighting (ii) retains best calibration; SMOTE degrades calibration
most. Actual numbers reported honestly regardless.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ._bootstrap_utils import REPO_ROOT, ensure_tables_dir


DATA_ROOT = Path(os.environ.get(
    "CCPERIOP_DATA_ROOT", "./data/",
))
MODELS_DIR = DATA_ROOT / "derived" / "phase2" / "models" / "XGB-INS-B"
FEATURES_DIR = DATA_ROOT / "derived" / "phase2" / "features"

SEX_ENCODING = {"Male": 0, "M": 0, "Female": 1, "F": 1}
SEED = 42


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


def _load_matrix(features_csv: str, feature_names: list[str], encoders: dict):
    df = pd.read_csv(FEATURES_DIR / features_csv, low_memory=False)
    df = _apply_encoders(df, encoders)
    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["mortality"].to_numpy(dtype=int)
    return X, y


def _calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Fit logistic regression on logit(y_pred) vs y_true for slope/intercept
    eps = 1e-7
    p = np.clip(y_pred, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y_true)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return {
        "brier": float(brier_score_loss(y_true, y_pred)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "oe_ratio": float(y_pred.mean() / max(y_true.mean(), 1e-9)),
    }


def _train_xgb(X_train: np.ndarray, y_train: np.ndarray,
               median_hp: dict, n_estimators: int,
               scale_pos_weight: float) -> xgb.Booster:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth":         int(median_hp["max_depth"]),
        "learning_rate":     float(median_hp["learning_rate"]),
        "min_child_weight":  float(median_hp["min_child_weight"]),
        "subsample":         float(median_hp["subsample"]),
        "colsample_bytree":  float(median_hp["colsample_bytree"]),
        "gamma":             float(median_hp["gamma"]),
        "reg_alpha":         float(median_hp["reg_alpha"]),
        "reg_lambda":        float(median_hp["reg_lambda"]),
        "scale_pos_weight":  float(scale_pos_weight),
        "seed": SEED,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    return xgb.train(params, dtrain, num_boost_round=n_estimators, verbose_eval=False)


def run() -> pd.DataFrame:
    # Load model metadata
    with (MODELS_DIR / "hyperparameters.json").open() as f:
        hp_data = json.load(f)
    with (MODELS_DIR / "feature_names.json").open() as f:
        feature_names = json.load(f)
    with (MODELS_DIR / "categorical_encoders.json").open() as f:
        encoders = json.load(f)

    median_hp = hp_data["median_hyperparams"]
    n_estimators = int(hp_data["median_n_estimators"])
    weighted_spw = float(median_hp["scale_pos_weight"])

    print("  loading INSPIRE training matrix ...")
    X_train, y_train = _load_matrix("inspire_train_full.csv", feature_names, encoders)
    print(f"    n={X_train.shape[0]:,}, events={y_train.sum():,}")

    print("  loading MOVER external test matrix ...")
    X_test, y_test = _load_matrix("mover_train_full.csv", feature_names, encoders)
    print(f"    n={X_test.shape[0]:,}, events={y_test.sum():,}")

    configs = [
        ("no_weighting", 1.0, False),
        ("class_weighting_main", weighted_spw, False),
        ("smote", 1.0, True),
    ]

    rows = []
    for name, spw, use_smote in configs:
        print(f"\n  === {name} (spw={spw:g}, smote={use_smote}) ===")
        t0 = time.time()
        if use_smote:
            # SMOTE requires dense, finite inputs. Median-impute NaNs;
            # ``keep_empty_features=True`` so all-NaN columns are kept
            # (as zeros) — the trained booster expects the full 140-
            # column schema at predict time.
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            X_imputed = imputer.fit_transform(X_train)
            smote = SMOTE(random_state=SEED, k_neighbors=5)
            X_tr, y_tr = smote.fit_resample(X_imputed, y_train)
            print(f"    SMOTE resampled: {len(X_tr):,} rows, "
                  f"{int(y_tr.sum()):,} events; NaNs median-imputed pre-SMOTE")
        else:
            X_tr, y_tr = X_train, y_train

        booster = _train_xgb(X_tr, y_tr, median_hp, n_estimators, spw)
        dtest = xgb.DMatrix(X_test, missing=np.nan)
        y_pred = booster.predict(dtest)
        duration_min = (time.time() - t0) / 60.0

        auc = float(roc_auc_score(y_test, y_pred))
        cal = _calibration_metrics(y_test, y_pred)
        rows.append({
            "config":           name,
            "scale_pos_weight": spw,
            "smote":            use_smote,
            "n_train":          len(X_tr),
            "n_train_events":   int(y_tr.sum()),
            "external_auc":     auc,
            "brier":            cal["brier"],
            "calibration_slope":     cal["calibration_slope"],
            "calibration_intercept": cal["calibration_intercept"],
            "oe_ratio":         cal["oe_ratio"],
            "duration_min":     round(duration_min, 2),
            "seed":             SEED,
        })
        print(f"    ext_AUC={auc:.4f}  brier={cal['brier']:.4f}  "
              f"slope={cal['calibration_slope']:.3f}  "
              f"intercept={cal['calibration_intercept']:.3f}  "
              f"O/E={cal['oe_ratio']:.3f}  ({duration_min:.2f} min)")

    out = pd.DataFrame(rows)
    out_path = ensure_tables_dir() / "class_imbalance_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  wrote {out_path}")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
