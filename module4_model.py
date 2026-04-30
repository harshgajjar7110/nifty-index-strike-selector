"""
Module 4 — LightGBM Quantile Regime Models
Trains per-VIX-regime LightGBM models with quantile regression (P10, P90).
Regimes: low (<15), mid (15–20), high (≥20).
"""

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

REGIME_THRESHOLDS = (15.0, 20.0)
MIN_DATA_PER_REGIME = 10  # Lower threshold to use NGBoost for more regimes
NGBOOST_N_ESTIMATORS = 250
NGBOOST_LEARNING_RATE = 0.05


def _tune_lgb_regime(X_regime: np.ndarray, y_regime: np.ndarray, alpha: float = 0.90) -> dict:
    """Grid search LightGBM hyperparams using 3-fold TimeSeriesSplit. Returns best params."""
    if len(X_regime) < 20:
        logger.warning("Too few samples for tuning, using defaults")
        return {"num_leaves": 31, "learning_rate": 0.1, "num_boost_round": 100}

    best_score = float("inf")
    best_params = {"num_leaves": 31, "learning_rate": 0.1, "num_boost_round": 100}
    tscv = TimeSeriesSplit(n_splits=3)

    for num_leaves in [15, 31, 63]:
        for lr in [0.05, 0.1, 0.15]:
            for n_rounds in [100, 200, 300]:
                scores = []
                for train_idx, val_idx in tscv.split(X_regime):
                    X_tr, X_val = X_regime[train_idx], X_regime[val_idx]
                    y_tr, y_val = y_regime[train_idx], y_regime[val_idx]
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    params = {
                        "objective": "quantile",
                        "metric": "quantile",
                        "alpha": alpha,
                        "num_leaves": num_leaves,
                        "learning_rate": lr,
                        "verbose": -1,
                    }
                    m = lgb.train(params, train_data, num_boost_round=n_rounds)
                    preds = m.predict(X_val)
                    pinball = float(np.mean(
                        np.where(y_val >= preds, alpha * (y_val - preds), (alpha - 1) * (y_val - preds))
                    ))
                    scores.append(pinball)
                avg = np.mean(scores)
                if avg < best_score:
                    best_score = avg
                    best_params = {"num_leaves": num_leaves, "learning_rate": lr, "num_boost_round": n_rounds}

    logger.info(f"Best params alpha={alpha} (score={best_score:.6f}): {best_params}")
    return best_params


def assign_regime(vix_series: pd.Series) -> pd.Series:
    """Assign VIX regimes: low (<15), mid (15–20), high (≥20)."""
    return pd.cut(vix_series, bins=[-np.inf, 15.0, 20.0, np.inf], labels=["low", "mid", "high"])


def _optimize_thresholds(X_train_df: pd.DataFrame, y_train: np.ndarray) -> tuple:
    """Grid search VIX thresholds to minimize variance of regime sizes (balanced training data)."""
    vix = X_train_df["vix_level"].values
    best_score = float("inf")
    best_low, best_high = 15.0, 20.0

    for low_thresh in np.arange(12.0, 18.5, 0.5):
        for high_thresh in np.arange(18.0, 25.5, 0.5):
            if high_thresh <= low_thresh:
                continue
            n_low  = int(np.sum(vix < low_thresh))
            n_mid  = int(np.sum((vix >= low_thresh) & (vix < high_thresh)))
            n_high = int(np.sum(vix >= high_thresh))
            counts = [n_low, n_mid, n_high]
            if min(counts) < 10:
                continue
            score = float(np.var(counts))
            if score < best_score:
                best_score = score
                best_low, best_high = low_thresh, high_thresh

    logger.info(f"Optimized thresholds: low={best_low}, high={best_high} (variance={best_score:.1f})")
    return best_low, best_high


def assign_regime_dynamic(vix_series: pd.Series, low_thresh: float, high_thresh: float) -> pd.Series:
    return pd.cut(vix_series, bins=[-np.inf, low_thresh, high_thresh, np.inf], labels=["low", "mid", "high"])


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

    # Load per-regime quantile alphas from env (configurable)
    regime_alphas = {
        "low": (
            float(os.getenv("ALPHA_LOW_P10", "0.10")),
            float(os.getenv("ALPHA_LOW_P90", "0.90"))
        ),
        "mid": (
            float(os.getenv("ALPHA_MID_P10", "0.15")),
            float(os.getenv("ALPHA_MID_P90", "0.85"))
        ),
        "high": (
            float(os.getenv("ALPHA_HIGH_P10", "0.10")),
            float(os.getenv("ALPHA_HIGH_P90", "0.90"))
        ),
    }
    logger.info(f"Regime alphas: {regime_alphas}")

    # Time-series split
    n = len(df)
    split = int(n * 0.80)
    X_train_df, X_test_df = X.iloc[:split], X.iloc[split:]
    X_train, X_test = X_train_df.values, X_test_df.values
    y_train, y_test = y[:split], y[split:]
    logger.info(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # Optimize VIX regime thresholds
    low_thresh, high_thresh = _optimize_thresholds(X_train_df, y_train)
    thresh_path = MODELS_DIR / "regime_thresholds.json"
    MODELS_DIR.mkdir(exist_ok=True)
    with open(thresh_path, "w") as f:
        json.dump({"low_thresh": float(low_thresh), "high_thresh": float(high_thresh)}, f, indent=2)
    logger.info(f"Regime thresholds saved to {thresh_path}")

    # Assign regimes on train set
    train_regimes = assign_regime_dynamic(X_train_df["vix_level"], low_thresh, high_thresh)
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
        alpha_lower, alpha_upper = regime_alphas[regime]

        logger.info(f"Training {regime} regime ({n_regime} rows) with alphas {alpha_lower}/{alpha_upper}...")

        if n_regime >= MIN_DATA_PER_REGIME:
            # Tune separately — P10 and P90 can prefer different tree depths
            best_p10 = _tune_lgb_regime(X_regime, y_regime, alpha=alpha_lower)
            best_p90 = _tune_lgb_regime(X_regime, y_regime, alpha=alpha_upper)
            train_data = lgb.Dataset(X_regime, label=y_regime)

            base_params = {"objective": "quantile", "metric": "quantile", "verbose": -1}

            # Train P10 model
            p10_params = {**base_params, "alpha": alpha_lower,
                          "num_leaves": best_p10["num_leaves"],
                          "learning_rate": best_p10["learning_rate"]}
            lgb_p10 = lgb.train(p10_params, train_data, num_boost_round=best_p10["num_boost_round"])

            # Train P90 model
            p90_params = {**base_params, "alpha": alpha_upper,
                          "num_leaves": best_p90["num_leaves"],
                          "learning_rate": best_p90["learning_rate"]}
            lgb_p90 = lgb.train(p90_params, train_data, num_boost_round=best_p90["num_boost_round"])

            lgb_models[regime] = {"p10": lgb_p10, "p90": lgb_p90}
            regime_meta[regime] = {
                "model_file": f"lgb_{regime}.pkl",
                "source": "lightgbm_quantile",
                "n_train": int(n_regime),
                "best_params_p10": best_p10,
                "best_params_p90": best_p90,
            }
            logger.info(f"{regime}: LightGBM quantile trained ({n_regime} rows)")
        else:
            logger.warning(f"{regime}: Insufficient data ({n_regime} rows < {MIN_DATA_PER_REGIME}), skipping")

    # Save best hyperparameters per regime
    best_params_all = {}
    for r in ["low", "mid", "high"]:
        if r in regime_meta:
            best_params_all[r] = {
                "p10": regime_meta[r].get("best_params_p10", {}),
                "p90": regime_meta[r].get("best_params_p90", {}),
            }
    MODELS_DIR.mkdir(exist_ok=True)
    params_path = MODELS_DIR / "lgb_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params_all, f, indent=2)
    logger.info(f"Best params saved to {params_path}")

    # Predict on test set per-regime
    test_regimes = assign_regime_dynamic(X_test_df["vix_level"], low_thresh, high_thresh)
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

    # Inversion guard
    inverted = y_pred_p90 < y_pred_p10
    if inverted.sum() > 0:
        logger.warning(f"{inverted.sum()} rows have P90 < P10 — clamping")
        y_pred_p10 = np.minimum(y_pred_p10, y_pred_p90)

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
