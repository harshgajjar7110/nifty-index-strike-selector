"""
Module 5: Conformal Calibration
Wraps per-regime NGBoost models with MAPIE conformal prediction
to provide guaranteed coverage bounds on predicted weekly range.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"


class RegimeNGBoostWrapper(BaseEstimator, RegressorMixin):
    """Route predictions by VIX regime to per-regime NGBoost models."""

    def __init__(self, ngb_models: dict, feature_columns: list):
        self.ngb_models = ngb_models
        self.feature_columns = feature_columns
        self.vix_col_idx = feature_columns.index("vix_level")

    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = np.zeros(len(X))
        for i, row in enumerate(X):
            vix = row[self.vix_col_idx]
            regime = "low" if vix < 15 else ("mid" if vix < 20 else "high")
            model = self.ngb_models[regime]

            if isinstance(model, dict):  # Linear QR fallback
                X_const = np.concatenate([[1], row])
                preds[i] = float(model["qr_p10"].predict(X_const)[0])
            else:  # NGBoost
                dist = model.pred_dist(row.reshape(1, -1))
                mu = float(dist.params["loc"][0])
                preds[i] = mu

        return preds


def run_calibration() -> dict:
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    target_coverage = float(os.getenv("TARGET_COVERAGE", "0.85"))
    logger.info(f"Loaded TARGET_COVERAGE={target_coverage} from .env")

    # Load regime models
    meta_path = MODELS_DIR / "regime_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run module4_model.py first")

    with open(meta_path) as f:
        regime_meta = json.load(f)

    ngb_models = {}
    for regime in ["low", "mid", "high"]:
        model_file = MODELS_DIR / regime_meta[regime]["model_file"]
        ngb_models[regime] = joblib.load(model_file)

    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    logger.info("Loaded per-regime NGBoost models and feature columns")

    data_path = DATA_DIR / "feature_matrix_with_garch.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Run module3_garch.py first")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded feature matrix: {df.shape}")

    # Prepare data — 80/20 split, then conf/eval
    df = df.sort_index()
    target_col = "log_range"
    available_features = [c for c in feature_columns if c in df.columns]
    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        logger.warning(f"Missing feature columns in data: {missing}")

    df_clean = df[available_features + [target_col]].dropna()
    X = df_clean[available_features].values
    y = df_clean[target_col].values

    split_idx = int(len(X) * 0.80)
    X_calib_all = df_clean[available_features].iloc[split_idx:]
    y_calib_all = df_clean[target_col].iloc[split_idx:]
    n = len(X_calib_all)
    mid = n // 2
    X_conf, y_conf = X_calib_all.iloc[:mid], y_calib_all.iloc[:mid]
    X_eval, y_eval = X_calib_all.iloc[mid:], y_calib_all.iloc[mid:]
    X_conf_arr = X_conf.values
    y_conf_arr = y_conf.values
    X_eval_arr = X_eval.values
    y_eval_arr = y_eval.values
    logger.info(f"Calibration set size: {n} samples (conf={mid}, eval={n - mid})")

    # Build wrapper + fit MAPIE
    wrapper = RegimeNGBoostWrapper(ngb_models, feature_columns)
    from mapie.regression import SplitConformalRegressor

    mapie = SplitConformalRegressor(
        estimator=wrapper,
        confidence_level=target_coverage,
        prefit=True,
    )
    mapie.conformalize(X_conf_arr, y_conf_arr)
    logger.info("Fitted SplitConformalRegressor with RegimeNGBoostWrapper")

    # Compute coverage
    def compute_coverage(confidence_level: float) -> float:
        try:
            m = SplitConformalRegressor(
                estimator=RegimeNGBoostWrapper(ngb_models, feature_columns),
                confidence_level=confidence_level,
                prefit=True,
            )
            m.conformalize(X_conf_arr, y_conf_arr)
            try:
                preds = m.predict_interval(X_eval_arr)
                lower, upper = preds[:, 0], preds[:, 1]
            except AttributeError:
                _, y_pi = m.predict(X_eval_arr, alpha=1 - confidence_level)
                if y_pi.ndim == 3:
                    lower, upper = y_pi[:, 0, 0], y_pi[:, 1, 0]
                else:
                    lower, upper = y_pi[:, 0], y_pi[:, 1]
            covered = np.mean((lower <= y_eval_arr) & (y_eval_arr <= upper))
            return float(covered)
        except Exception as e:
            logger.warning(f"Coverage computation failed at {confidence_level}: {e}")
            return float("nan")

    coverage_80 = compute_coverage(0.80)
    coverage_85 = compute_coverage(0.85)
    coverage_90 = compute_coverage(0.90)

    logger.info(f"Empirical coverage @ 80%: {coverage_80:.4f}")
    logger.info(f"Empirical coverage @ 85%: {coverage_85:.4f}")
    logger.info(f"Empirical coverage @ 90%: {coverage_90:.4f}")

    for target, actual in [(0.80, coverage_80), (0.85, coverage_85), (0.90, coverage_90)]:
        if not np.isnan(actual) and actual < target:
            logger.warning(f"Coverage {actual:.4f} is below target {target:.2f}")

    # Calibration curve
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    nominal_levels = np.arange(0.70, 0.96, 0.05)
    empirical_levels = [compute_coverage(cl) for cl in nominal_levels]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(nominal_levels, empirical_levels, "o-", color="steelblue",
            label="Empirical coverage", linewidth=2, markersize=6)
    ax.plot([0.70, 0.95], [0.70, 0.95], "--", color="gray",
            label="Perfect calibration", linewidth=1.5)
    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title("MAPIE Conformal Calibration Curve\n(Nifty 50 Weekly Log-Range)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0.68, 0.97)
    ax.set_ylim(0.68, 0.97)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = OUTPUTS_DIR / "calibration_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved calibration curve to {plot_path}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mapie_calibrated.pkl"
    joblib.dump(mapie, model_path)
    logger.info(f"Saved calibrated MAPIE model to {model_path}")

    report = {
        "coverage_at_80": round(coverage_80, 4) if not np.isnan(coverage_80) else None,
        "coverage_at_85": round(coverage_85, 4) if not np.isnan(coverage_85) else None,
        "coverage_at_90": round(coverage_90, 4) if not np.isnan(coverage_90) else None,
    }

    report_path = OUTPUTS_DIR / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved calibration report to {report_path}")

    return report


if __name__ == "__main__":
    report = run_calibration()
    print(report)
