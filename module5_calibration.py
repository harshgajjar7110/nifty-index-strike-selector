"""
Module 5: Conformal Calibration
Wraps P10/P90 LightGBM quantile models with MAPIE conformal prediction
to provide guaranteed coverage bounds on predicted weekly range.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"


def run_calibration() -> dict:
    # ------------------------------------------------------------------ #
    # 1. Load artifacts
    # ------------------------------------------------------------------ #
    model_files = {
        "lgbm_p10": MODELS_DIR / "lgbm_p10.pkl",
        "lgbm_p90": MODELS_DIR / "lgbm_p90.pkl",
        "feature_columns": MODELS_DIR / "feature_columns.pkl",
    }
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError("Run module4_model.py first")

    lgbm_p10 = joblib.load(model_files["lgbm_p10"])
    lgbm_p90 = joblib.load(model_files["lgbm_p90"])
    feature_columns = joblib.load(model_files["feature_columns"])
    logger.info("Loaded P10/P90 models and feature columns")

    data_path = DATA_DIR / "feature_matrix_with_garch.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Run module3_garch.py first")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded feature matrix: {df.shape}")

    # ------------------------------------------------------------------ #
    # 2. Prepare data — same 80/20 time-series split as M4
    # ------------------------------------------------------------------ #
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
    X_calib = X_conf.values
    y_calib = y_conf.values
    logger.info(f"Calibration set size: {n} samples (conf={mid}, eval={n - mid})")

    # ------------------------------------------------------------------ #
    # 3. Fit MAPIE with quantile conformal method
    # ------------------------------------------------------------------ #
    # MAPIE 1.3.0 API: ConformalizedQuantileRegressor with prefit=True
    # takes [lower_estimator, upper_estimator] and skips fit step,
    # going directly to conformalize on the calibration set.
    try:
        from mapie.regression import ConformalizedQuantileRegressor

        mapie = ConformalizedQuantileRegressor(
            estimator=[lgbm_p10, lgbm_p90],
            confidence_level=0.85,  # target coverage = 1 - alpha (alpha=0.15)
            prefit=True,
        )
        # With prefit=True, _is_fitted is already True; call conformalize directly
        mapie.conformalize(X_calib, y_calib)
        logger.info("Fitted ConformalizedQuantileRegressor (MAPIE 1.3.0 API)")
        use_new_api = True

    except Exception as e:
        logger.warning(f"ConformalizedQuantileRegressor failed ({e}), falling back to SplitConformalRegressor")
        try:
            from mapie.regression import SplitConformalRegressor
            mapie = SplitConformalRegressor(
                estimator=lgbm_p90,
                confidence_level=0.85,
                prefit=True,
            )
            mapie.conformalize(X_calib, y_calib)
            use_new_api = True
        except Exception as e2:
            logger.warning(f"SplitConformalRegressor also failed ({e2}), using legacy MapieRegressor")
            from mapie.regression import MapieRegressor
            mapie = MapieRegressor(estimator=lgbm_p90, method="plus", cv="prefit")
            mapie.fit(X_calib, y_calib)
            use_new_api = False

    # ------------------------------------------------------------------ #
    # 4. Compute empirical coverage at multiple alpha levels
    # ------------------------------------------------------------------ #
    def compute_coverage(confidence_level: float) -> float:
        """Get empirical coverage on eval half of calibration set at given confidence level."""
        try:
            if use_new_api:
                try:
                    from mapie.regression import ConformalizedQuantileRegressor
                    m = ConformalizedQuantileRegressor(
                        estimator=[lgbm_p10, lgbm_p90],
                        confidence_level=confidence_level,
                        prefit=True,
                    )
                    m.conformalize(X_conf, y_conf)
                except Exception:
                    from mapie.regression import SplitConformalRegressor
                    m = SplitConformalRegressor(
                        estimator=lgbm_p90,
                        confidence_level=confidence_level,
                        prefit=True,
                    )
                    m.conformalize(X_conf, y_conf)
                try:
                    preds = m.predict_interval(X_eval)
                    lower, upper = preds[:, 0], preds[:, 1]
                except AttributeError:
                    _, y_pi = m.predict(X_eval, alpha=1 - confidence_level)
                    if y_pi.ndim == 3:
                        lower, upper = y_pi[:, 0, 0], y_pi[:, 1, 0]
                    else:
                        lower, upper = y_pi[:, 0], y_pi[:, 1]
            else:
                try:
                    preds = mapie.predict_interval(X_eval)
                    lower, upper = preds[:, 0], preds[:, 1]
                except AttributeError:
                    _, y_pi = mapie.predict(X_eval, alpha=1 - confidence_level)
                    if y_pi.ndim == 3:
                        lower, upper = y_pi[:, 0, 0], y_pi[:, 1, 0]
                    else:
                        lower, upper = y_pi[:, 0], y_pi[:, 1]
        except Exception as e:
            logger.warning(f"Coverage computation failed at {confidence_level}: {e}")
            return float("nan")

        covered = np.mean((lower <= y_eval.values) & (y_eval.values <= upper))
        return float(covered)

    coverage_80 = compute_coverage(0.80)
    coverage_85 = compute_coverage(0.85)
    coverage_90 = compute_coverage(0.90)

    logger.info(f"Empirical coverage @ 80%: {coverage_80:.4f}")
    logger.info(f"Empirical coverage @ 85%: {coverage_85:.4f}")
    logger.info(f"Empirical coverage @ 90%: {coverage_90:.4f}")

    for target, actual in [(0.80, coverage_80), (0.85, coverage_85), (0.90, coverage_90)]:
        if not np.isnan(actual) and actual < target:
            logger.warning(
                f"Coverage {actual:.4f} is below target {target:.2f} — "
                "model may be under-confident"
            )

    # ------------------------------------------------------------------ #
    # 5. Plot calibration curve
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 6. Save model and report
    # ------------------------------------------------------------------ #
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mapie_calibrated.pkl"
    joblib.dump(mapie, model_path)
    logger.info(f"Saved calibrated MAPIE model to {model_path}")

    report = {
        "coverage_at_80": round(coverage_80, 4),
        "coverage_at_85": round(coverage_85, 4),
        "coverage_at_90": round(coverage_90, 4),
    }

    report = {k: (v if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in report.items()}
    report_path = OUTPUTS_DIR / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved calibration report to {report_path}")

    return report


if __name__ == "__main__":
    report = run_calibration()
    print(report)
