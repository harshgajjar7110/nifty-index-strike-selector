"""
Module 4 — LightGBM Quantile Regression Models
Trains P10 and P90 quantile models to predict the weekly log-range of Nifty 50.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from loguru import logger

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def train_models() -> dict:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run module3_garch.py first")

    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Dataset shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Feature / target split
    # ------------------------------------------------------------------
    if "log_range" not in df.columns:
        raise ValueError("'log_range' column not found in the dataset.")

    df = df.sort_index()  # ensure chronological order

    y = df["log_range"].values
    X = df.drop(columns=["log_range"])
    feature_columns: list[str] = list(X.columns)
    logger.info(f"Features: {len(feature_columns)}  |  Target rows: {len(y)}")

    # ------------------------------------------------------------------
    # 3. Time-series split — no shuffle
    # ------------------------------------------------------------------
    n = len(df)
    split = int(n * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    logger.info(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ------------------------------------------------------------------
    # 4. Train P10 model
    # ------------------------------------------------------------------
    logger.info("Training P10 (alpha=0.10) model …")
    lgbm_p10 = LGBMRegressor(
        objective="quantile", alpha=0.10,
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=10,
        random_state=42,
    )
    lgbm_p10.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(50), log_evaluation(100)],
    )

    # ------------------------------------------------------------------
    # 5. Train P90 model
    # ------------------------------------------------------------------
    logger.info("Training P90 (alpha=0.90) model …")
    lgbm_p90 = LGBMRegressor(
        objective="quantile", alpha=0.90,
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=10,
        random_state=42,
    )
    lgbm_p90.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(50), log_evaluation(100)],
    )

    # ------------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------------
    y_pred_p10 = lgbm_p10.predict(X_test)
    y_pred_p90 = lgbm_p90.predict(X_test)

    pb_p10 = pinball_loss(y_test, y_pred_p10, alpha=0.10)
    pb_p90 = pinball_loss(y_test, y_pred_p90, alpha=0.90)

    coverage_mask = (y_pred_p10 <= y_test) & (y_test <= y_pred_p90)
    coverage_rate = float(coverage_mask.mean())

    coverage_warning = coverage_rate < 0.80
    if coverage_warning:
        logger.warning(
            f"Coverage rate {coverage_rate:.2%} is below the 80% target!"
        )
    else:
        logger.info(f"Coverage rate: {coverage_rate:.2%}  (target >= 80%)")

    logger.info(f"Pinball P10: {pb_p10:.6f}  |  Pinball P90: {pb_p90:.6f}")

    # ------------------------------------------------------------------
    # 7. SHAP feature importance (based on P90 model)
    # ------------------------------------------------------------------
    logger.info("Computing SHAP values for P90 model …")
    explainer = shap.TreeExplainer(lgbm_p90)
    shap_values = explainer.shap_values(X_test)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    shap_out = OUTPUTS_DIR / "shap_importance.png"
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(shap_out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP importance plot saved to {shap_out}")

    # ------------------------------------------------------------------
    # 8. Save models
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(lgbm_p10, MODELS_DIR / "lgbm_p10.pkl")
    joblib.dump(lgbm_p90, MODELS_DIR / "lgbm_p90.pkl")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.pkl")
    logger.info("Models saved to models/")

    # ------------------------------------------------------------------
    # 9. Save evaluation JSON
    # ------------------------------------------------------------------
    eval_dict = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "coverage_rate": round(coverage_rate, 6),
        "pinball_p10": round(pb_p10, 6),
        "pinball_p90": round(pb_p90, 6),
        "coverage_warning": coverage_warning,
    }

    eval_path = OUTPUTS_DIR / "model_evaluation.json"
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_dict, f, indent=2)
    logger.info(f"Evaluation saved to {eval_path}")

    return eval_dict


if __name__ == "__main__":
    results = train_models()
    print(results)
