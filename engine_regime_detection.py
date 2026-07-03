"""
Regime Detection Engine: Hidden Markov Model (HMM) Implementation
==================================================================

Identifies market regimes using 4-state HMM:
  State 1: Quiet Bull Market (low vol, rising trend, stable IV)
  State 2: Range Bound Mean Reversion (moderate vol, sideways)
  State 3: High Volatility Expansion (rising VIX, increased ATR, expanding vol)
  State 4: Liquidity Stress / Panic (extreme VIX, strong directional move, correlation spikes)

Returns: Regime probabilities, allocation scores (0-100), regime state class.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HMM_MODEL_PATH = MODELS_DIR / "hmm_regime_detector.pkl"
HMM_METADATA_PATH = MODELS_DIR / "hmm_metadata.json"

# HMM Configuration
NUM_STATES = 4
REGIME_NAMES = {
    0: "Quiet Bull",
    1: "Range Bound",
    2: "High Expansion",
    3: "Panic",
}

# Allocation levels (0-100 score mapping)
ALLOCATION_LEVELS = {
    "Quiet Bull": {"score": 90, "factor": 1.0},
    "Range Bound": {"score": 70, "factor": 1.0},
    "High Expansion": {"score": 40, "factor": 0.5},
    "Panic": {"score": 20, "factor": 0.0},  # No new trades
}

# Feature weights for HMM training
FEATURE_WEIGHTS = {
    "vix_level": 0.3,
    "realized_volatility": 0.25,
    "trend_strength": 0.2,
    "vol_of_vol": 0.15,
    "atr_normalized": 0.1,
}


# ---------------------------------------------------------------------------
# Feature Preparation for Regime Detection
# ---------------------------------------------------------------------------

def prepare_regime_features(
    features_df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Prepare features for HMM training.
    
    Args:
        features_df: DataFrame with date index, must include 'vix', 'close', 'high', 'low', 'atr'
        window: Lookback window for normalized calculations
        
    Returns:
        DataFrame with HMM input features (normalized)
    """
    logger.info(f"Preparing regime features (window={window})")
    
    df = features_df.copy()
    
    # Ensure required columns exist
    required = ["close", "high", "low", "atr"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}. Using defaults.")
    
    # 1. VIX Level (0-100, normalized by baseline ~16)
    if "vix" in df.columns:
        vix_baseline = 16.0
        df["vix_normalized"] = (df["vix"] / vix_baseline).clip(0, 4)  # 0-4 scale
    else:
        df["vix_normalized"] = 1.0  # Neutral if no VIX available
    
    # 2. Realized Volatility (rolling std dev of log returns)
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = df["returns"].rolling(window).std() * np.sqrt(252)
    df["realized_vol_normalized"] = df["realized_vol"] / 0.20  # Normalized by 20% baseline vol
    df["realized_vol_normalized"] = df["realized_vol_normalized"].clip(0, 4)
    
    # 3. Trend Strength (RSI-like momentum)
    df["momentum"] = df["close"].diff(5) / df["close"].shift(5)
    df["trend_strength"] = df["momentum"].rolling(window).mean()
    df["trend_strength_normalized"] = (df["trend_strength"] + 0.02) / 0.04  # 0.02 is baseline, 0.04 is 2x
    df["trend_strength_normalized"] = df["trend_strength_normalized"].clip(-1, 1)
    
    # 4. Volatility of Volatility (vol spike detection)
    df["vol_of_vol"] = df["realized_vol"].rolling(window).std()
    df["vol_of_vol_normalized"] = df["vol_of_vol"] / (df["realized_vol"].mean() * 0.5 + 0.001)
    df["vol_of_vol_normalized"] = df["vol_of_vol_normalized"].clip(0, 3)
    
    # 5. ATR Normalized by Close (volatility regime)
    df["atr_normalized"] = (df["atr"] / df["close"]) * 100
    df["atr_normalized_scale"] = df["atr_normalized"] / (df["atr_normalized"].rolling(window).mean() + 0.001)
    df["atr_normalized_scale"] = df["atr_normalized_scale"].clip(0, 3)
    
    # Drop NaN rows from rolling calculations
    df = df.dropna()
    
    logger.info(f"Regime features ready: {len(df)} rows")
    
    return df


def extract_hmm_features(
    features_df: pd.DataFrame,
    feature_cols: list | None = None,
) -> np.ndarray:
    """
    Extract and stack feature columns for HMM input.
    
    Returns: (N, num_features) array
    """
    if feature_cols is None:
        feature_cols = [
            "vix_normalized",
            "realized_vol_normalized",
            "trend_strength_normalized",
            "vol_of_vol_normalized",
            "atr_normalized_scale",
        ]
    
    missing = [col for col in feature_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    X = features_df[feature_cols].values.astype(np.float64)
    
    # Fill any remaining NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return X


# ---------------------------------------------------------------------------
# HMM Training
# ---------------------------------------------------------------------------

def train_hmm(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    n_iter: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Train Hidden Markov Model on historical regime features.
    
    Args:
        features_df: DataFrame with regime features
        test_size: Fraction for validation (not used in training, for evaluation only)
        n_iter: HMM iterations
        random_state: Reproducibility
        
    Returns:
        dict with model, train/test split info, feature columns
    """
    logger.info(f"Training HMM with {len(features_df)} observations")
    
    # Prepare features
    reg_features = prepare_regime_features(features_df)
    feature_cols = [
        "vix_normalized",
        "realized_vol_normalized",
        "trend_strength_normalized",
        "vol_of_vol_normalized",
        "atr_normalized_scale",
    ]
    X = extract_hmm_features(reg_features, feature_cols)
    
    # Train/test split (chronological)
    n_test = int(len(X) * test_size)
    n_train = len(X) - n_test
    X_train = X[:n_train]
    X_test = X[n_train:]
    
    logger.info(f"Train set: {n_train} rows, Test set: {n_test} rows")
    
    # Initialize and train HMM
    hmm = GaussianHMM(
        n_components=NUM_STATES,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )
    
    logger.info(f"Fitting HMM with {NUM_STATES} states...")
    hmm.fit(X_train)
    
    # Evaluate on test set
    test_likelihood = hmm.score(X_test)
    logger.success(f"HMM fitted. Test likelihood: {test_likelihood:.4f}")
    
    # Classify train and test sets
    train_labels = hmm.predict(X_train)
    test_labels = hmm.predict(X_test)
    
    # Characterize each state
    state_char = _characterize_states(X_train, train_labels)
    
    return {
        "model": hmm,
        "feature_cols": feature_cols,
        "train_size": n_train,
        "test_size": n_test,
        "test_likelihood": float(test_likelihood),
        "state_characteristics": state_char,
    }


def _characterize_states(X: np.ndarray, labels: np.ndarray) -> dict:
    """Characterize each state by mean feature values."""
    char = {}
    for state in range(NUM_STATES):
        mask = labels == state
        if mask.sum() > 0:
            char[state] = {
                "count": int(mask.sum()),
                "pct": float(mask.mean() * 100),
                "mean_vix": float(X[mask, 0].mean()),
                "mean_realized_vol": float(X[mask, 1].mean()),
                "mean_trend": float(X[mask, 2].mean()),
            }
    return char


# ---------------------------------------------------------------------------
# Inference: Predict Current Regime
# ---------------------------------------------------------------------------

def predict_regime_probabilities(
    recent_features: pd.DataFrame,
    hmm_model: GaussianHMM,
    feature_cols: list,
    lookback: int = 1,
) -> dict:
    """
    Predict regime probabilities for the most recent period.
    
    Args:
        recent_features: DataFrame with prepared regime features (at least lookback rows)
        hmm_model: Trained HMM model
        feature_cols: List of feature column names
        lookback: Number of recent rows to use for probability (1 = most recent, higher = smoother)
        
    Returns:
        dict with regime probabilities, most likely regime, allocation score/factor
    """
    if len(recent_features) < lookback:
        logger.warning(f"Not enough data (need {lookback}, have {len(recent_features)})")
        # Return neutral regime if insufficient data
        return {
            "probabilities": {s: 1.0 / NUM_STATES for s in range(NUM_STATES)},
            "most_likely_state": 1,  # Range Bound (neutral)
            "most_likely_regime": "Range Bound",
            "allocation_score": 70,
            "allocation_factor": 1.0,
            "confidence": 0.0,
        }
    
    # Extract features from recent observations
    recent_features_prep = prepare_regime_features(recent_features)
    X_recent = extract_hmm_features(recent_features_prep, feature_cols)
    
    # Use last 'lookback' rows
    X_subset = X_recent[-lookback:]
    
    # Predict state probabilities (posterior prob for final timestep)
    probs = hmm_model.predict_proba(X_subset)[-1]  # Get last timestep probs
    pred_state = hmm_model.predict(X_subset)[-1]   # Get last timestep state
    
    # Map state to regime name and allocation
    regime_name = REGIME_NAMES.get(pred_state, "Unknown")
    alloc = ALLOCATION_LEVELS.get(regime_name, {"score": 50, "factor": 0.5})
    
    # Confidence is the probability of the most likely state
    confidence = float(probs[pred_state])
    
    return {
        "probabilities": {
            REGIME_NAMES[i]: float(probs[i])
            for i in range(NUM_STATES)
        },
        "most_likely_state": int(pred_state),
        "most_likely_regime": regime_name,
        "allocation_score": alloc["score"],
        "allocation_factor": alloc["factor"],
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_hmm_model(model_dict: dict) -> None:
    """Persist HMM model and metadata."""
    import joblib
    
    model = model_dict["model"]
    joblib.dump(model, HMM_MODEL_PATH)
    logger.info(f"HMM model saved → {HMM_MODEL_PATH}")
    
    # Save metadata (non-model info)
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_cols": model_dict["feature_cols"],
        "train_size": model_dict["train_size"],
        "test_size": model_dict["test_size"],
        "test_likelihood": model_dict["test_likelihood"],
        "state_characteristics": model_dict["state_characteristics"],
    }
    with open(HMM_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"HMM metadata saved → {HMM_METADATA_PATH}")


def load_hmm_model() -> tuple[GaussianHMM, dict]:
    """
    Load trained HMM model and metadata.
    
    Returns:
        (model, metadata_dict)
    """
    import joblib
    
    if not HMM_MODEL_PATH.exists():
        raise FileNotFoundError(f"HMM model not found at {HMM_MODEL_PATH}. Train first using train_hmm().")
    
    model = joblib.load(HMM_MODEL_PATH)
    
    if HMM_METADATA_PATH.exists():
        with open(HMM_METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    logger.info(f"HMM model loaded → {HMM_MODEL_PATH}")
    
    return model, metadata


# ---------------------------------------------------------------------------
# Calibration Utility: Map States to Known Market Conditions
# ---------------------------------------------------------------------------

def calibrate_regime_mapping(
    features_df: pd.DataFrame,
    hmm_model: GaussianHMM,
    metadata: dict,
) -> dict:
    """
    Optional: Refine state-to-regime mapping by examining historical correspondence
    between HMM states and known market regimes (VIX spikes, trend periods, etc).
    
    For now, uses a fixed heuristic mapping based on state mean feature values.
    """
    # This is a placeholder for future enhancement
    # Could correlate HMM states with detected VIX spikes, trend shifts, etc.
    return metadata.get("state_characteristics", {})
