"""
Module 4 — LightGBM Quantile Regime Models
Trains per-VIX-regime LightGBM models with quantile regression (P10, P90).
Regimes: low (<15), mid (15–20), high (≥20).
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
import lightgbm as lgb
from loguru import logger

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

REGIME_THRESHOLDS = (15.0, 20.0)
MIN_DATA_PER_REGIME = 10  # Lower threshold to use NGBoost for more regimes
NGBOOST_N_ESTIMATORS = 250
NGBOOST_LEARNING_RATE = 0.05


def assign_regime(vix_series: pd.Series) -> pd.Series:
    """Assign VIX regimes: low (<15), mid (15–20), high (≥20)."""
    return pd.cut(vix_series, bins=[-np.inf, 15.0, 20.0, np.inf], labels=["low", "mid", "high"])


def train_models() -> dict:
    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run module3_garch.py first")
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Dataset shape: {df.shape}")

    if "log_range" not in df.columns or "vix_level" not in df.columns:
        raise ValueError("'log_range' and 'vix_level' columns required.")

    df = df.sort_index()
    y = df["log_range"].values
    X = df.drop(columns=["log_range"])
    feature_columns = list(X.columns)
    logger.info(f"Features: {len(feature_columns)} | Columns: {feature_columns} | Target rows: {len(y)}")

    # Time-series split
    n = len(df)
    split = int(n * 0.80)
    X_train_df, X_test_df = X.iloc[:split], X.iloc[split:]
    X_train, X_test = X_train_df.values, X_test_df.values
    y_train, y_test = y[:split], y[split:]
    logger.info(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # Assign regimes on train set
    train_regimes = assign_regime(X_train_df["vix_level"])
    mask_array = (train_regimes == train_regimes).values
    logger.info(f"Train regimes: low={sum(train_regimes=='low')}, mid={sum(train_regimes=='mid')}, high={sum(train_regimes=='high')}")

    # Train per-regime LightGBM quantile models (P10 and P90)
    lgb_models = {}
    regime_meta = {}

    for regime in ["low", "mid", "high"]:
        mask = (train_regimes == regime).values
        X_regime = X_train[mask]
        y_regime = y_train[mask]
        n_regime = len(X_regime)

        logger.info(f"Training {regime} regime ({n_regime} rows)...")

        if n_regime >= MIN_DATA_PER_REGIME:
            # LightGBM quantile regression for P10 and P90
            train_data = lgb.Dataset(X_regime, label=y_regime)
            params = {
                "objective": "quantile",
                "metric": "quantile",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "verbose": -1,
            }

            # Train P10 model
            p10_params = params.copy()
            p10_params["alpha"] = 0.10
            lgb_p10 = lgb.train(p10_params, train_data, num_boost_round=100)

            # Train P90 model
            p90_params = params.copy()
            p90_params["alpha"] = 0.90
            lgb_p90 = lgb.train(p90_params, train_data, num_boost_round=100)

            lgb_models[regime] = {"p10": lgb_p10, "p90": lgb_p90}
            regime_meta[regime] = {
                "model_file": f"lgb_{regime}.pkl",
                "source": "lightgbm_quantile",
                "n_train": int(n_regime),
            }
            logger.info(f"{regime}: LightGBM quantile trained ({n_regime} rows)")
        else:
            logger.warning(f"{regime}: Insufficient data ({n_regime} rows < {MIN_DATA_PER_REGIME}), skipping")

    # Predict on test set per-regime
    test_regimes = assign_regime(X_test_df["vix_level"])
    y_pred_p10_list = []
    y_pred_p90_list = []

    for idx, row_x in enumerate(X_test):
        regime = test_regimes.iloc[idx]
        if regime not in lgb_models:
            logger.warning(f"Regime {regime} has no trained model, using global mean")
            p10_pred = float(np.percentile(y_train, 10))
            p90_pred = float(np.percentile(y_train, 90))
        else:
            models = lgb_models[regime]
            p10_pred = float(models["p10"].predict(row_x.reshape(1, -1))[0])
            p90_pred = float(models["p90"].predict(row_x.reshape(1, -1))[0])

        y_pred_p10_list.append(p10_pred)
        y_pred_p90_list.append(p90_pred)

    y_pred_p10 = np.array(y_pred_p10_list)
    y_pred_p90 = np.array(y_pred_p90_list)

    # Evaluate overall coverage
    coverage_mask = (y_pred_p10 <= y_test) & (y_test <= y_pred_p90)
    coverage_rate = float(coverage_mask.mean())
    logger.info(f"Overall coverage rate: {coverage_rate:.2%}")

    # Per-regime coverage
    regime_coverage = {}
    for regime in ["low", "mid", "high"]:
        mask = test_regimes == regime
        if mask.sum() > 0:
            cov = float(coverage_mask[mask].mean())
            regime_coverage[regime] = round(cov, 4)
            logger.info(f"{regime} coverage: {cov:.2%} ({mask.sum()} rows)")
        else:
            regime_coverage[regime] = None

    # SHAP for LightGBM models
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for regime in ["low", "mid", "high"]:
        if regime not in lgb_models:
            logger.info(f"{regime}: No trained model, skipping SHAP")
            continue

        # Get test rows in this regime
        mask = test_regimes == regime
        if mask.sum() == 0:
            continue

        X_regime_test = X_test_df[mask].values
        try:
            # Use P10 model for SHAP
            model = lgb_models[regime]["p10"]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_regime_test)

            fig = plt.figure()
            if isinstance(shap_values, list):
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            shap.summary_plot(shap_vals, X_regime_test, plot_type="bar", show=False)
            plt.tight_layout()
            shap_path = OUTPUTS_DIR / f"shap_importance_{regime}.png"
            fig.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"SHAP {regime} saved to {shap_path}")
        except Exception as e:
            logger.warning(f"SHAP {regime} failed: {e}")

    # Save models
    MODELS_DIR.mkdir(exist_ok=True)
    for regime in ["low", "mid", "high"]:
        if regime in lgb_models:
            model_path = MODELS_DIR / f"lgb_{regime}.pkl"
            joblib.dump(lgb_models[regime], model_path)
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.pkl")
    logger.info("Models saved to models/")

    # Save regime metadata
    meta_path = MODELS_DIR / "regime_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(regime_meta, f, indent=2)
    logger.info(f"Regime metadata saved to {meta_path}")

    # Save evaluation JSON
    eval_dict = {
        "n_train": int(len(X_train_df)),
        "n_test": int(len(X_test_df)),
        "coverage_rate": round(coverage_rate, 6),
        "regime_coverage": regime_coverage,
        "regime_n_train": {r: v["n_train"] for r, v in regime_meta.items()},
        "regime_model_type": {r: v["source"] for r, v in regime_meta.items()},
    }

    eval_path = OUTPUTS_DIR / "model_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_dict, f, indent=2)
    logger.info(f"Evaluation saved to {eval_path}")

    return eval_dict


if __name__ == "__main__":
    results = train_models()
    print(results)
