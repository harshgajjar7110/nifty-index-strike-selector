"""
Strike Selection Engine (Phase 2.5)
==================================

Systematically identifies high-probability, positive-expected-value option-selling
opportunities by combining:
  - Probability distributions (from engine_probability_distribution)
  - Expected value analysis (from engine_expected_value)
  - Market regime & volatility context
  - Filtering constraints (DTE, Delta, IV rank)
  - Dynamic Greeks calculation

Output: Ranked strike recommendations for immediate execution or paper trading.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger

from engine_probability_distribution import (
    LognormalDistribution,
    generate_probability_table,
)
from engine_expected_value import EVScorer, calculate_single_leg_ev, calculate_strangle_ev

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STRIKES_RECOMMENDATION_PATH = DATA_DIR / "strikes_recommendation_latest.json"

# DTE Constraints
DTE_CONSTRAINTS = {
    "weekly": {"min": 5, "max": 9},
    "swing": {"min": 15, "max": 45},
    "monthly": {"min": 25, "max": 60},
}

# Delta Constraints (10-20 delta preferred for selling)
DELTA_CONSTRAINTS = {
    "put": {"min": 0.08, "max": 0.22},   # 8-22 delta range
    "call": {"min": 0.08, "max": 0.22},
}

# IV Rank Thresholds (0-100, higher = more extreme IV)
IV_RANK_THRESHOLDS = {
    "minimum": 30,       # Below this: too low IV, skip
    "comfortable": 40,   # Above this: acceptable to sell
    "ideal": 60,         # Above this: aggressive selling OK
}


# ---------------------------------------------------------------------------
# Greeks Calculation
# ---------------------------------------------------------------------------

def calculate_greeks(
    spot: float,
    strike: float,
    volatility: float,
    dte: float,
    side: str,
    rate: float = 0.0,
) -> dict:
    """
    Calculate Greeks for short option position.
    
    Args:
        spot: Current spot price
        strike: Strike price
        volatility: Annual volatility
        dte: Days to expiry
        side: 'put' or 'call'
        rate: Risk-free rate (default 0)
        
    Returns:
        dict with delta, gamma, vega, theta
    """
    
    T = dte / 365.0
    
    if T <= 0 or volatility <= 0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }
    
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility ** 2) * T) / (
        volatility * np.sqrt(T)
    )
    d2 = d1 - volatility * np.sqrt(T)
    
    # Greeks for LONG option (then negate for short)
    if side.lower() == "put":
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(T))
        vega = spot * norm.pdf(d1) * np.sqrt(T) / 100.0
        theta = (
            -spot * norm.pdf(d1) * volatility / (2 * np.sqrt(T))
            + rate * strike * np.exp(-rate * T) * norm.cdf(-d2)
        ) / 365.0
    else:  # call
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(T))
        vega = spot * norm.pdf(d1) * np.sqrt(T) / 100.0
        theta = (
            -spot * norm.pdf(d1) * volatility / (2 * np.sqrt(T))
            - rate * strike * np.exp(-rate * T) * norm.cdf(d2)
        ) / 365.0
    
    # Negate for SHORT position
    return {
        "delta": float(-delta),
        "gamma": float(-gamma),
        "vega": float(-vega),
        "theta": float(-theta),
    }


# ---------------------------------------------------------------------------
# Candidate Strike Generation & Filtering
# ---------------------------------------------------------------------------

class CandidateFilter:
    """Generate and filter candidate strikes based on constraints."""
    
    def __init__(
        self,
        dte: float,
        spot: float,
        volatility: float,
        regime_allocation_factor: float = 1.0,
        iv_rank: float = 50.0,
    ):
        """
        Args:
            dte: Days to expiry
            spot: Current spot price
            volatility: Forecast volatility
            regime_allocation_factor: From regime engine (0-1)
            iv_rank: IV rank (0-100)
        """
        self.dte = dte
        self.spot = spot
        self.volatility = volatility
        self.regime_factor = regime_allocation_factor
        self.iv_rank = iv_rank
        
        # Determine trading horizon (weekly/swing/monthly)
        self.horizon = self._classify_horizon()
        self.dte_constraints = DTE_CONSTRAINTS.get(self.horizon, DTE_CONSTRAINTS["weekly"])
    
    def _classify_horizon(self) -> str:
        """Classify DTE into trading horizon."""
        if 5 <= self.dte <= 9:
            return "weekly"
        elif 25 <= self.dte <= 45:
            return "monthly"
        elif 15 <= self.dte < 25:
            return "swing"
        else:
            # Snap to nearest valid horizon
            if self.dte < 15:
                return "weekly"
            else:
                return "monthly"
    
    def check_dte_constraint(self) -> bool:
        """Check if DTE within acceptable range."""
        min_dte = self.dte_constraints["min"]
        max_dte = self.dte_constraints["max"]
        is_valid = min_dte <= self.dte <= max_dte
        
        if not is_valid:
            logger.warning(
                f"DTE {self.dte} outside {self.horizon} range [{min_dte}, {max_dte}]"
            )
        
        return is_valid
    
    def check_iv_constraint(self) -> bool:
        """Check if IV rank above threshold."""
        min_rank = IV_RANK_THRESHOLDS["minimum"]
        
        if self.iv_rank < min_rank:
            logger.warning(f"IV rank {self.iv_rank:.0f} below minimum {min_rank}")
            return False
        
        if self.iv_rank < IV_RANK_THRESHOLDS["comfortable"]:
            logger.info(f"IV rank {self.iv_rank:.0f} in comfort zone (≥{IV_RANK_THRESHOLDS['comfortable']})")
        
        return True
    
    def check_regime_constraint(self) -> bool:
        """Check if regime allows trading."""
        min_allocation = 0.25  # Don't trade if allocation < 25%
        
        if self.regime_factor < min_allocation:
            logger.warning(
                f"Regime allocation {self.regime_factor:.0%} < {min_allocation:.0%} minimum"
            )
            return False
        
        return True
    
    def generate_candidate_strikes(
        self,
        side: str,
        delta_target: float = 0.15,
    ) -> list:
        """
        Generate candidate strikes for a side.
        
        Args:
            side: 'put' or 'call'
            delta_target: Target delta (default 0.15 = 15 delta)
            
        Returns:
            List of candidate strikes, sorted by distance from spot
        """
        # Use lognormal to find strike at target delta
        dist = LognormalDistribution(self.spot, self.volatility, self.dte)
        
        candidates = []
        
        # Generate strikes in ±5% range around spot
        for pct_move in np.arange(-5, 5.5, 0.25):
            strike = self.spot * (1 + pct_move / 100)
            
            # Check delta
            if side.lower() == "put":
                breach_prob = dist.cdf(strike)
                strike_delta = abs(breach_prob)
            else:
                breach_prob = 1 - dist.cdf(strike)
                strike_delta = breach_prob
            
            # Filter by delta range
            if DELTA_CONSTRAINTS[side]["min"] <= strike_delta <= DELTA_CONSTRAINTS[side]["max"]:
                candidates.append({
                    "strike": float(strike),
                    "delta": float(strike_delta),
                    "breach_prob": float(breach_prob if side == "put" else breach_prob),
                })
        
        logger.info(f"Generated {len(candidates)} candidate {side} strikes")
        return candidates
    
    def validate_candidates(self) -> bool:
        """Check all constraints before proceeding."""
        checks = [
            ("DTE constraint", self.check_dte_constraint()),
            ("IV constraint", self.check_iv_constraint()),
            ("Regime constraint", self.check_regime_constraint()),
        ]
        
        all_valid = all(result for _, result in checks)
        
        for check_name, result in checks:
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check_name}")
        
        return all_valid


# ---------------------------------------------------------------------------
# Strike Selection & Ranking
# ---------------------------------------------------------------------------

class StrikeSelector:
    """Select and rank optimal strikes for multi-leg strategies."""
    
    def __init__(
        self,
        spot: float,
        volatility: float,
        dte: float,
        regime_allocation_factor: float = 1.0,
        iv_rank: float = 50.0,
        premium_model: dict | None = None,
    ):
        """
        Args:
            spot: Current spot
            volatility: Forecast volatility
            dte: Days to expiry
            regime_allocation_factor: Risk allocation from regime
            iv_rank: Implied vol rank (0-100)
            premium_model: dict with 'pe', 'ce' premium points
        """
        self.spot = spot
        self.volatility = volatility
        self.dte = dte
        self.regime_factor = regime_allocation_factor
        self.iv_rank = iv_rank
        
        # Premium model
        if premium_model is None:
            # Default: scale by VIX
            vix = volatility * 100
            vix_factor = max(0.6, min(1.5, vix / 16.0))
            base = 80
            premium_model = {
                "pe": int(base * vix_factor),
                "ce": int(base * vix_factor),
            }
        self.premium_model = premium_model
        
        # Initialize filter
        self.filter = CandidateFilter(dte, spot, volatility, regime_allocation_factor, iv_rank)
        
        # Initialize scorer
        self.scorer = EVScorer(
            regime_allocation_factor=regime_allocation_factor,
            tail_risk_threshold=0.15,
        )
    
    def select_strikes(self) -> dict:
        """
        Select optimal strikes for trading.
        
        Returns:
            dict with recommended strikes and strategy
        """
        logger.info(
            f"Strike selection: spot={self.spot:.2f}, vol={self.volatility:.2%}, "
            f"dte={self.dte}, regime={self.regime_factor:.0%}"
        )
        
        # Validate constraints
        if not self.filter.validate_candidates():
            logger.warning("Constraints not met - no trades recommended")
            return {
                "status": "CONSTRAINTS_NOT_MET",
                "regime_allocation": self.regime_factor,
                "reason": "Market regime or IV conditions unfavorable",
            }
        
        # Generate candidate strikes
        put_candidates = self.filter.generate_candidate_strikes("put")
        call_candidates = self.filter.generate_candidate_strikes("call")
        
        if not put_candidates or not call_candidates:
            logger.warning("Insufficient candidates after filtering")
            return {
                "status": "INSUFFICIENT_CANDIDATES",
                "regime_allocation": self.regime_factor,
            }
        
        # Score individual legs
        put_scored = self._score_leg(put_candidates, "put")
        call_scored = self._score_leg(call_candidates, "call")
        
        # Select best legs
        best_put = max(put_scored, key=lambda x: x["opportunity_score"])
        best_call = max(call_scored, key=lambda x: x["opportunity_score"])
        
        # Evaluate multi-leg strategies
        strangle_eval = self._evaluate_strangle(best_put, best_call)
        
        return {
            "status": "SUCCESS",
            "horizon": self.filter.horizon,
            "generated_at": datetime.now().isoformat(),
            "market": {
                "spot": float(self.spot),
                "volatility": float(self.volatility),
                "dte": float(self.dte),
                "iv_rank": float(self.iv_rank),
                "regime_allocation": float(self.regime_factor),
            },
            "put_recommendations": {
                "all_candidates": put_scored[:5],  # Top 5
                "best_strike": best_put,
            },
            "call_recommendations": {
                "all_candidates": call_scored[:5],  # Top 5
                "best_strike": best_call,
            },
            "multi_leg_strategies": {
                "strangle": strangle_eval,
            },
            "execution_instructions": self._generate_instructions(best_put, best_call),
        }
    
    def _score_leg(self, candidates: list, side: str) -> list:
        """Score individual leg candidates by EV."""
        scored = []
        
        for candidate in candidates:
            strike = candidate["strike"]
            
            # Calculate EV
            premium_pts = self.premium_model.get(f"{'pe' if side == 'put' else 'ce'}", 80)
            premium = premium_pts / 100 * self.spot
            
            ev_dict = calculate_single_leg_ev(
                strike, side, premium,
                self.spot, self.volatility, self.dte,
                method="lognormal",
            )
            
            # Score with risk adjustments
            scored_dict = self.scorer.score_single_leg(ev_dict)
            
            # Add Greeks
            greeks = calculate_greeks(
                self.spot, strike, self.volatility, self.dte, side
            )
            scored_dict.update(greeks)
            
            scored.append(scored_dict)
        
        # Sort by opportunity score
        scored.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return scored
    
    def _evaluate_strangle(self, put_strike_dict: dict, call_strike_dict: dict) -> dict:
        """Evaluate strangle strategy combining best PE + CE."""
        
        pe_prem = self.premium_model.get("pe", 80) / 100 * self.spot
        ce_prem = self.premium_model.get("ce", 80) / 100 * self.spot
        
        strangle_dict = calculate_strangle_ev(
            put_strike_dict["strike"],
            call_strike_dict["strike"],
            pe_prem,
            ce_prem,
            self.spot,
            self.volatility,
            self.dte,
            method="lognormal",
        )
        
        # Score
        scored_strangle = self.scorer.score_multi_leg(strangle_dict)
        
        # Add portfolio Greeks
        pe_greeks = calculate_greeks(
            self.spot, put_strike_dict["strike"], self.volatility, self.dte, "put"
        )
        ce_greeks = calculate_greeks(
            self.spot, call_strike_dict["strike"], self.volatility, self.dte, "call"
        )
        
        scored_strangle["portfolio_greeks"] = {
            "delta": pe_greeks["delta"] + ce_greeks["delta"],
            "gamma": pe_greeks["gamma"] + ce_greeks["gamma"],
            "vega": pe_greeks["vega"] + ce_greeks["vega"],
            "theta": pe_greeks["theta"] + ce_greeks["theta"],
        }
        
        return scored_strangle
    
    def _generate_instructions(self, put_dict: dict, call_dict: dict) -> dict:
        """Generate execution instructions."""
        return {
            "strategy": "Short Strangle",
            "expiry_days": int(self.dte),
            "legs": [
                {
                    "type": "Short Put",
                    "strike": float(put_dict["strike"]),
                    "delta": float(put_dict.get("delta", 0.15)),
                    "premium_per_contract": float(
                        self.premium_model.get("pe", 80) / 100 * self.spot
                    ),
                    "allocation_factor": float(self.regime_factor),
                },
                {
                    "type": "Short Call",
                    "strike": float(call_dict["strike"]),
                    "delta": float(call_dict.get("delta", 0.15)),
                    "premium_per_contract": float(
                        self.premium_model.get("ce", 80) / 100 * self.spot
                    ),
                    "allocation_factor": float(self.regime_factor),
                },
            ],
            "total_premium": float(
                (self.premium_model.get("pe", 80) + self.premium_model.get("ce", 80))
                / 100
                * self.spot
            ),
            "risk_reward": "Short premium strategies yield small steady profits with concentrated downside",
            "exit_rules": [
                "Close at 50% profit",
                "Exit on DTE ≤ 1",
                "Stop loss at 2x premium received",
            ],
        }


# ---------------------------------------------------------------------------
# Main Interface
# ---------------------------------------------------------------------------

def generate_strike_recommendations(
    spot: float,
    volatility: float,
    dte: float,
    regime_allocation_factor: float = 1.0,
    iv_rank: float = 50.0,
    premium_model: dict | None = None,
) -> dict:
    """
    Generate strike recommendations for immediate action.
    
    Args:
        spot: Current spot price
        volatility: Forecast volatility
        dte: Days to expiry
        regime_allocation_factor: Risk allocation (0-1)
        iv_rank: IV rank (0-100)
        premium_model: Optional premium override
        
    Returns:
        Complete strike recommendation report
    """
    
    selector = StrikeSelector(
        spot=spot,
        volatility=volatility,
        dte=dte,
        regime_allocation_factor=regime_allocation_factor,
        iv_rank=iv_rank,
        premium_model=premium_model,
    )
    
    recommendations = selector.select_strikes()
    
    logger.success("Strike recommendations generated")
    return recommendations


def save_strike_recommendations(report: dict) -> None:
    """Persist recommendations to JSON."""
    with open(STRIKES_RECOMMENDATION_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Recommendations saved → {STRIKES_RECOMMENDATION_PATH}")


def load_strike_recommendations() -> dict:
    """Load latest recommendations."""
    if not STRIKES_RECOMMENDATION_PATH.exists():
        logger.warning(f"No recommendations found at {STRIKES_RECOMMENDATION_PATH}")
        return {}
    
    with open(STRIKES_RECOMMENDATION_PATH, "r") as f:
        report = json.load(f)
    
    logger.info(f"Recommendations loaded from {STRIKES_RECOMMENDATION_PATH}")
    return report
