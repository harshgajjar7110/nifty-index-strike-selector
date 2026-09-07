"""Shared utilities. Lazy exports to keep lightweight imports cheap."""

__all__ = [
    "retry", "black_scholes_price", "estimate_ic_premium_bs", "bs_price_with_skew",
    "black_scholes_delta", "implied_volatility", "RegimeLGBQuantileWrapper",
    "extract_pis", "compute_breach_probability", "predict_mu_sigma_p90",
    "soft_vix_size_multiplier", "REGIMES", "load_regime_thresholds",
    "assign_regime", "assign_regime_series", "extract_vix", "find_column",
    "chronological_train_test_split",
]

_LAZY = {
    "retry": "utils.retry_utils",
    "black_scholes_price": "utils.black_scholes",
    "estimate_ic_premium_bs": "utils.black_scholes",
    "bs_price_with_skew": "utils.black_scholes",
    "black_scholes_delta": "utils.black_scholes",
    "implied_volatility": "utils.black_scholes",
    "RegimeLGBQuantileWrapper": "utils.models_utils",
    "extract_pis": "utils.models_utils",
    "compute_breach_probability": "utils.models_utils",
    "predict_mu_sigma_p90": "utils.models_utils",
    "soft_vix_size_multiplier": "utils.models_utils",
    "REGIMES": "utils.utils_constants",
    "load_regime_thresholds": "utils.utils_constants",
    "assign_regime": "utils.utils_constants",
    "assign_regime_series": "utils.utils_constants",
    "extract_vix": "utils.utils_constants",
    "find_column": "utils.utils_constants",
    "chronological_train_test_split": "utils.utils_constants",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(name)
