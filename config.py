"""
Centralized configuration using Pydantic BaseSettings.
All environment variables are parsed, validated, and documented in one place.
Usage:
    from config import cfg
    print(cfg.strike_buffer_points)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR


class _Config(BaseSettings):
    # ------------------------------------------------------------------
    # Strike generation
    # ------------------------------------------------------------------
    strike_buffer_points: int = Field(default=50, ge=0, le=300, description="Points beyond P10/P90")
    wing_width_low_vix: int = Field(default=150, ge=50, le=500)
    wing_width_mid_vix: int = Field(default=200, ge=50, le=500)
    wing_width_high_vix: int = Field(default=250, ge=50, le=500)
    target_coverage: float = Field(default=0.85, ge=0.5, le=0.99)
    min_buffer_points: int = Field(default=75, ge=0, le=200)
    put_skew_points: int = Field(default=0, ge=-200, le=200)
    call_skew_points: int = Field(default=0, ge=-200, le=200)

    # ------------------------------------------------------------------
    # Model / calibration
    # ------------------------------------------------------------------
    alpha_low_p10: float = Field(default=0.075, ge=0.01, le=0.49)
    alpha_low_p90: float = Field(default=0.925, ge=0.51, le=0.99)
    alpha_mid_p10: float = Field(default=0.075, ge=0.01, le=0.49)
    alpha_mid_p90: float = Field(default=0.925, ge=0.51, le=0.99)
    alpha_high_p10: float = Field(default=0.075, ge=0.01, le=0.49)
    alpha_high_p90: float = Field(default=0.925, ge=0.51, le=0.99)
    regime_recent_only_years: float | None = Field(default=2.0, ge=0.0, le=10.0, description="If > 0, optimize regime thresholds on the most recent N years of training data. 0 = use full history.")

    # ------------------------------------------------------------------
    # Backtest / costs
    # ------------------------------------------------------------------
    nifty_lot_size: int = Field(default=65, ge=1, le=1000)
    premium_points_base: int = Field(default=80, ge=1, le=500)
    risk_free_rate: float = Field(default=0.065, ge=0.0, le=0.20)
    dividend_yield: float = Field(default=0.015, ge=0.0, le=0.10)
    brokerage_per_trade_inr: float = Field(default=20.0, ge=0.0)

    # ------------------------------------------------------------------
    # Walk-forward
    # ------------------------------------------------------------------
    wf_initial_train_weeks: int = Field(default=120, ge=20, le=500)
    wf_retrain_every_weeks: int = Field(default=4, ge=1, le=52)
    wf_calibration_weeks: int = Field(default=20, ge=5, le=100)
    wf_sl_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    wf_slippage_entry: float = Field(default=1.0, ge=0.0, le=10.0)
    wf_slippage_exit: float = Field(default=0.5, ge=0.0, le=10.0)
    wf_min_cost_cover_mult: float = Field(default=3.0, ge=1.0, le=10.0, description="Skip weeks where premium < mult * round-trip cost (charges + slippage).")

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    monitor_coverage_decay: float = Field(default=0.05, ge=0.0, le=0.30)
    monitor_drift_window_weeks: int = Field(default=12, ge=4, le=52)
    monitor_feature_drift_pval: float = Field(default=0.01, ge=0.001, le=0.10)

    # ------------------------------------------------------------------
    # Credit spread filters
    # ------------------------------------------------------------------
    min_rr_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    min_ev_proxy_pts: float = Field(default=0.0, ge=-100.0, le=500.0, description="Drop any spread whose ev_proxy is below this (points). 0.0 disables.")
    min_premium_low_vix_pts: float = Field(default=15.0, ge=1.0)
    min_premium_mid_vix_pts: float = Field(default=25.0, ge=1.0)
    min_premium_high_vix_pts: float = Field(default=40.0, ge=1.0)
    min_dte_to_trade: int = Field(default=7, ge=0, le=30)
    max_spreads_output: int = Field(default=6, ge=1, le=20)
    min_oi_liquidity: int = Field(default=5000, ge=0)

    # ------------------------------------------------------------------
    # Spread generation (was os.getenv)
    # ------------------------------------------------------------------
    spread_wing_width_low_vix: int = Field(default=100, ge=50, le=500)
    spread_wing_width_mid_vix: int = Field(default=150, ge=50, le=500)
    spread_wing_width_high_vix: int = Field(default=200, ge=50, le=500)
    spread_delta_target: float = Field(default=0.674, ge=0.1, le=2.0)

    # ------------------------------------------------------------------
    # Walk-forward (was os.getenv)
    # ------------------------------------------------------------------
    wf_min_premium_pts: float = Field(default=20.0, ge=0.0, le=500.0)
    wf_max_vix_trade: float = Field(default=30.0, ge=10.0, le=100.0)
    wf_soft_vix_lower: float = Field(default=15.0, ge=10.0, le=40.0, description="Soft-VIX band lower edge: between this and wf_max_vix_trade, reduce size to wf_soft_size_mult.")
    wf_soft_size_mult: float = Field(default=0.75, ge=0.1, le=1.0, description="Size multiplier for weeks in the soft-VIX band [wf_soft_vix_lower, wf_max_vix_trade].")

    # ------------------------------------------------------------------
    # Fallback / calibrated defaults
    # ------------------------------------------------------------------
    fallback_log_range_p10: float = Field(default=0.015, ge=0.001, le=0.5)
    fallback_log_range_p90: float = Field(default=0.035, ge=0.001, le=0.5)
    fallback_log_range_mu: float = Field(default=0.025, ge=0.001, le=0.5)
    fallback_log_range_sigma: float = Field(default=0.008, ge=0.001, le=0.5)

    # ------------------------------------------------------------------
    # Direction / skew
    # ------------------------------------------------------------------
    pcr_bear_threshold: float = Field(default=0.80, ge=0.0, le=2.0)
    pcr_bull_threshold: float = Field(default=1.20, ge=0.0, le=2.0)
    pcr_put_tighten_pts: int = Field(default=30, ge=0, le=200)
    pcr_call_tighten_pts: int = Field(default=30, ge=0, le=200)
    direction_confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_roc: float = Field(default=0.45, ge=0.0, le=1.0)
    weight_vix: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_garch: float = Field(default=0.20, ge=0.0, le=1.0)
    call_skew_factor: float = Field(default=0.01, ge=0.0, le=0.50)

    # ------------------------------------------------------------------
    # Capital sizing
    # ------------------------------------------------------------------
    ic_capital_inr: float = Field(default=500_000, ge=10_000)
    ic_capital_at_risk_pct: float = Field(default=0.20, ge=0.01, le=1.0)
    ic_target_return_monthly_pct: float = Field(default=0.025, ge=0.0, le=0.50)
    ic_target_return_weekly_pct: float = Field(default=0.008, ge=0.0, le=0.50)
    ic_expiry_mode: str = Field(default="weekly")
    ic_max_extra_buffer_pts: int = Field(default=500, ge=0, le=2000)
    ic_min_pop: float = Field(default=0.75, ge=0.0, le=1.0)
    ic_max_breach_prob_per_leg: float = Field(default=0.15, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def wing_width_by_regime(self) -> dict:
        return {
            "low": self.wing_width_low_vix,
            "mid": self.wing_width_mid_vix,
            "high": self.wing_width_high_vix,
        }

    @property
    def regime_alphas(self) -> dict:
        return {
            "low": (self.alpha_low_p10, self.alpha_low_p90),
            "mid": (self.alpha_mid_p10, self.alpha_mid_p90),
            "high": (self.alpha_high_p10, self.alpha_high_p90),
        }

    @property
    def min_premium_by_regime(self) -> dict:
        return {
            "low": self.min_premium_low_vix_pts,
            "mid": self.min_premium_mid_vix_pts,
            "high": self.min_premium_high_vix_pts,
        }


@lru_cache()
def get_config() -> _Config:
    return _Config()


cfg = get_config()

if __name__ == "__main__":
    import json
    print(json.dumps(cfg.model_dump(), indent=2))
