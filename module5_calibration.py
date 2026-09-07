"""
Module 5: Conformal Calibration
Wraps per-regime LightGBM models with MAPIE conformal prediction
to provide guaranteed coverage bounds on predicted weekly range.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from config import cfg
from utils.models_utils import RegimeLGBQuantileWrapper
from utils.utils_constants import REGIMES

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"


def _load_regime_thresholds() -> tuple:
    """Load optimized VIX regime thresholds from models/; fall back to defaults (15, 20)."""
    thresh_path = Path(__file__).parent / "models" / "regime_thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            t = json.load(f)
        return t["low_thresh"], t["high_thresh"]
    return 15.0, 20.0


def _coverage_at_target(mid: np.ndarray, half_width: np.ndarray, y_test: np.ndarray, target: float) -> float:
    """Binary search for interval scale that achieves target empirical coverage."""
    lo, hi = 0.5, 5.0
    for _ in range(50):
        scale = (lo + hi) / 2
        lower = mid - scale * half_width
        upper = mid + scale * half_width
        cov = float(np.mean((lower <= y_test) & (y_test <= upper)))
        if cov < target:
            lo = scale
        else:
            hi = scale
    return round(cov, 4)


def run_calibration() -> dict:
    target_coverage = cfg.target_coverage
    logger.info(f"Loaded TARGET_COVERAGE={target_coverage} from config")

    # Load regime models
    meta_path = MODELS_DIR / "regime_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run module4_model.py first")

    with open(meta_path) as f:
        regime_meta = json.load(f)

    lgb_models = {}
    for regime in REGIMES:
        if regime in regime_meta:
            model_file = MODELS_DIR / regime_meta[regime]["model_file"]
            if model_file.exists():
                lgb_models[regime] = joblib.load(model_file)

    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    logger.info("Loaded per-regime LightGBM models and feature columns")

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

    usable_features = [c for c in available_features if df[c].notna().any()]
    dead = sorted(set(available_features) - set(usable_features))
    if dead:
        logger.warning(f"Excluding all-NaN features from calibration: {dead}")
    df_clean = df[available_features + [target_col]].dropna(subset=usable_features + [target_col])
    X = df_clean[available_features].values
    y = df_clean[target_col].values

    split_idx = int(len(X) * 0.80)  # Align with module4_model.py 80/20 split to avoid data leakage
    X_calib_all = df_clean[available_features].iloc[split_idx:]
    y_calib_all = df_clean[target_col].iloc[split_idx:]
    n = len(X_calib_all)
    X_conf_arr = X_calib_all.values
    y_conf_arr = y_calib_all.values
    logger.info(f"Calibration set size: {n} samples (global-only, no per-regime split)")

    low_thresh, high_thresh = _load_regime_thresholds()
    logger.info(f"Loaded regime thresholds: low={low_thresh}, high={high_thresh}")

    from mapie.regression import SplitConformalRegressor

    wrapper_global = RegimeLGBQuantileWrapper(
        lgb_models=lgb_models,
        feature_columns=feature_columns,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    mapie_global = SplitConformalRegressor(
        estimator=wrapper_global,
        confidence_level=target_coverage,
        prefit=True,
    )
    mapie_global.conformalize(X_conf_arr, y_conf_arr)
    logger.info(f"Global MAPIE conformalized on {n} samples")

    vix_col_idx = feature_columns.index("vix_level")
    X_eval_arr = X_conf_arr
    y_eval_arr = y_conf_arr
    coverage_per_regime = {}
    calibration_method = {}
    for regime in REGIMES:
        if regime == "low":
            mask_eval = X_eval_arr[:, vix_col_idx] < low_thresh
        elif regime == "mid":
            mask_eval = (X_eval_arr[:, vix_col_idx] >= low_thresh) & (X_eval_arr[:, vix_col_idx] < high_thresh)
        else:
            mask_eval = X_eval_arr[:, vix_col_idx] >= high_thresh
        X_r_eval, y_r_eval = X_eval_arr[mask_eval], y_eval_arr[mask_eval]
        if len(X_r_eval) >= 3:
            _, y_pis_r = mapie_global.predict_interval(X_r_eval)
            if len(y_pis_r.shape) == 3:
                y_low_r, y_high_r = y_pis_r[:, 0, 0], y_pis_r[:, 1, 0]
            else:
                y_low_r, y_high_r = y_pis_r[:, 0], y_pis_r[:, 1]
            coverage_per_regime[regime] = float(np.mean((y_low_r <= y_r_eval) & (y_r_eval <= y_high_r)))
            logger.info(f"Regime {regime} in-sample coverage: {coverage_per_regime[regime]:.4f} (n={len(X_r_eval)})")
        else:
            coverage_per_regime[regime] = None
        calibration_method[regime] = "global_only"

    for regime in REGIMES:
        stale_path = MODELS_DIR / f"mapie_{regime}.pkl"
        if stale_path.exists():
            stale_path.unlink()
            logger.info(f"Removed stale per-regime {stale_path} (global-only mode)")

    joblib.dump(mapie_global, MODELS_DIR / "mapie_calibrated.pkl")
    logger.info("Saved global MAPIE to mapie_calibrated.pkl")

    # Compute coverage: Use the evaluation set (X_eval_arr) which was NOT used for conformalization
    def compute_coverage_mapie() -> dict:
        """Honest OOS coverage via 3-fold TimeSeriesSplit over the holdout."""
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        covered_all, low_all, high_all, y_all = [], [], [], []
        for tr_idx, te_idx in tscv.split(X_conf_arr):
            m = SplitConformalRegressor(estimator=wrapper_global, confidence_level=target_coverage, prefit=True)
            m.conformalize(X_conf_arr[tr_idx], y_conf_arr[tr_idx])
            _, y_pis_cv = m.predict_interval(X_conf_arr[te_idx])
            if len(y_pis_cv.shape) == 3:
                low_cv, high_cv = y_pis_cv[:, 0, 0], y_pis_cv[:, 1, 0]
            else:
                low_cv, high_cv = y_pis_cv[:, 0], y_pis_cv[:, 1]
            y_te = y_conf_arr[te_idx]
            covered_all.append((low_cv <= y_te) & (y_te <= high_cv))
            low_all.append(low_cv)
            high_all.append(high_cv)
            y_all.append(y_te)
        covered = np.concatenate(covered_all)
        y_low = np.concatenate(low_all)
        y_high = np.concatenate(high_all)
        y_eval_cv = np.concatenate(y_all)
        actual_coverage = float(np.mean(covered))
        half_width = (y_high - y_low) / 2
        mid = (y_low + y_high) / 2

        return {
            "actual_coverage": float(actual_coverage),
            "_mid": mid,
            "_half_width": half_width,
            "_y_test": y_eval_cv,
        }

    coverage_results = compute_coverage_mapie()
    actual_coverage = coverage_results["actual_coverage"]
    _mid = coverage_results.pop("_mid")
    _half_width = coverage_results.pop("_half_width")
    _y_test = coverage_results.pop("_y_test")

    logger.info(f"Empirical OOS coverage @ {target_coverage:.0%}: {actual_coverage:.4f}")

    if actual_coverage < target_coverage:
        logger.warning(f"Coverage {actual_coverage:.4f} is below target {target_coverage:.2f}")

    # Calibration curve: honest CV empirical coverage at each nominal level
    from sklearn.model_selection import TimeSeriesSplit as _TSCV
    nominal_levels = np.arange(0.70, 0.96, 0.05)
    empirical_levels = []
    for lvl in nominal_levels:
        cv_covered = []
        for tr_idx, te_idx in _TSCV(n_splits=3).split(X_conf_arr):
            try:
                m = SplitConformalRegressor(estimator=wrapper_global, confidence_level=float(lvl), prefit=True)
                m.conformalize(X_conf_arr[tr_idx], y_conf_arr[tr_idx])
                _, pis = m.predict_interval(X_conf_arr[te_idx])
            except ValueError as e:
                logger.warning(f"lvl={float(lvl):.2f}: skipped fold ({e})")
                continue
            if len(pis.shape) == 3:
                low, high = pis[:, 0, 0], pis[:, 1, 0]
            else:
                low, high = pis[:, 0], pis[:, 1]
            y_te = y_conf_arr[te_idx]
            cv_covered.append(np.mean((low <= y_te) & (y_te <= high)))
        empirical_levels.append(float(np.mean(cv_covered)) if cv_covered else float("nan"))

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

    # Save wrapper
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_path = MODELS_DIR / "regime_lgb_wrapper.pkl"
    joblib.dump(wrapper_global, wrapper_path)
    logger.info(f"Saved regime wrapper to {wrapper_path}")

    report = {
        "target_coverage": target_coverage,
        "actual_oos_coverage": round(actual_coverage, 4),
        "per_regime_coverage": {k: round(v, 4) if v is not None else None for k, v in coverage_per_regime.items()},
        "per_regime_calibration_method": calibration_method,
        "calibration_data": {
            "nominal": [round(float(l), 2) for l in nominal_levels],
            "empirical": [round(float(l), 4) for l in empirical_levels]
        }
    }

    report_path = OUTPUTS_DIR / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved calibration report to {report_path}")

    return report


if __name__ == "__main__":
    report = run_calibration()
    print(report)
