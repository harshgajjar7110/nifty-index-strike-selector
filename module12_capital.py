"""
Capital-aware, target-return-driven strike selection for credit spreads.

Finds the safest (most OTM) short strike that still delivers the target return,
given a capital budget and risk limit. Supports both weekly and monthly expiries.
"""

from pathlib import Path
from math import sqrt, log, exp
from scipy.stats import norm
import numpy as np

from config import cfg
from module9_spreads import estimate_spread_premium, NIFTY_LOT_SIZE
from module6_strikes import round_to_strike

BASE_DIR = Path(__file__).parent


def _load_capital_config() -> dict:
    """Load IC_* config from centralized config."""
    return {
        "capital_inr": cfg.ic_capital_inr,
        "capital_at_risk_pct": cfg.ic_capital_at_risk_pct,
        "target_return_monthly": cfg.ic_target_return_monthly_pct,
        "target_return_weekly": cfg.ic_target_return_weekly_pct,
        "expiry_mode": cfg.ic_expiry_mode.lower(),
        "max_extra_buffer_pts": cfg.ic_max_extra_buffer_pts,
        "min_pop": cfg.ic_min_pop,
        "max_breach_prob_per_leg": cfg.ic_max_breach_prob_per_leg,
        "lot_size": cfg.nifty_lot_size,
    }


def compute_margin_per_lot(max_loss_pts: float, lot_size: int = cfg.nifty_lot_size) -> float:
    """Margin per lot = max_loss_pts * lot_size (INR)."""
    return max(max_loss_pts, 0.0) * lot_size


def compute_lots_tradeable(
    capital_inr: float,
    capital_at_risk_pct: float,
    margin_per_lot_inr: float,
) -> int:
    """Max lots within risk budget. Returns 0 if margin is zero."""
    if margin_per_lot_inr <= 0:
        return 0
    return int((capital_inr * capital_at_risk_pct) // margin_per_lot_inr)


def compute_required_premium_pts(
    capital_inr: float,
    target_return_pct: float,
    lots: int,
    lot_size: int = cfg.nifty_lot_size,
) -> float:
    """Premium per lot (in pts) needed to hit target return."""
    if lots == 0:
        return float("inf")
    required_total_inr = capital_inr * target_return_pct
    return (required_total_inr / lots) / lot_size


def resolve_target_return(
    expiry_mode: str,
    target_return_monthly: float,
    target_return_weekly: float,
    dte_days: int,
) -> float:
    """Resolve correct target return for expiry type (no frequency multiplier)."""
    if expiry_mode == "weekly" or dte_days <= 7:
        return target_return_weekly
    return target_return_monthly


def evaluate_spread_config(
    spot: float,
    short_K: float,
    long_K: float,
    T_years: float,
    sigma: float,
    lot_size: int = cfg.nifty_lot_size,
    spread_type: str = "bull_put",
    r: float = cfg.risk_free_rate,
    q: float = cfg.dividend_yield,
) -> dict:
    """
    Evaluate a spread config using BS pricing.
    Returns strike, premium, max_loss, POP.
    """
    result = estimate_spread_premium(spot, short_K, long_K, T_years, sigma, r, spread_type, q)

    # POP from lognormal: P(Nifty stays above/below short_K)
    if T_years > 0 and sigma > 0:
        if spread_type == "bull_put":
            # P(Nifty >= short_K) at expiry
            z = log(spot / short_K) / (sigma * sqrt(T_years))
        else:  # bear_call
            # P(Nifty <= short_K) at expiry
            z = log(short_K / spot) / (sigma * sqrt(T_years))
        pop_pct = float(norm.cdf(z))
    else:
        pop_pct = None

    return {
        "short_strike": short_K,
        "long_strike": long_K,
        "premium_pts": result["premium_pts"],
        "max_loss_pts": result["max_loss_pts"],
        "rr_ratio": result.get("rr_ratio", 0.0),
        "pop_pct": pop_pct,
    }


def find_safest_viable_spread(
    spot: float,
    base_short_K: float,
    wing_width: float,
    T_years: float,
    sigma: float,
    spread_type: str = "bull_put",
    lot_size: int = cfg.nifty_lot_size,
    capital_config: dict | None = None,
    r: float = cfg.risk_free_rate,
    q: float = cfg.dividend_yield,
) -> dict:
    """
    Binary search for max extra OTM offset where premium >= required.
    Returns capital_sizing config with status, strikes, and feasibility reason.
    """
    cfg = capital_config or _load_capital_config()

    capital_inr = cfg["capital_inr"]
    capital_at_risk_pct = cfg["capital_at_risk_pct"]
    lot_size = cfg.get("lot_size", 65)
    max_extra = cfg["max_extra_buffer_pts"]
    min_pop = cfg["min_pop"]
    dte_days = int(max(T_years * 365, 1))

    target_return = resolve_target_return(
        cfg["expiry_mode"],
        cfg["target_return_monthly"],
        cfg["target_return_weekly"],
        dte_days,
    )

    # Step 1: Feasibility check at extra_offset=0
    # Base long_K is determined by spread_type + wing_width
    if spread_type == "bull_put":
        base_long_K = base_short_K - wing_width
    else:  # bear_call
        base_long_K = base_short_K + wing_width

    base_config = evaluate_spread_config(
        spot, base_short_K, base_long_K, T_years, sigma, lot_size, spread_type, r, q,
    )
    base_margin = compute_margin_per_lot(base_config["max_loss_pts"], lot_size)
    base_lots = compute_lots_tradeable(capital_inr, capital_at_risk_pct, base_margin)
    base_req_pts = compute_required_premium_pts(capital_inr, target_return, base_lots, lot_size)

    if base_lots == 0:
        return _infeasible_result(
            cfg, dte_days, target_return,
            "Insufficient capital for even 1 lot at tightest buffer",
        )

    if base_config["premium_pts"] < base_req_pts:
        return _infeasible_result(
            cfg, dte_days, target_return,
            f"Even at base strike, premium {base_config['premium_pts']:.2f} pts "
            f"< required {base_req_pts:.2f} pts. Reduce target return or increase capital.",
        )

    # Step 2: Binary search over extra_offset
    STEP = 50  # Nifty strike interval
    lo, hi = 0, (max_extra // STEP) * STEP
    best_config = base_config
    best_offset = 0
    best_lots = base_lots
    best_margin = base_margin

    while lo <= hi:
        mid_offset = ((lo + hi) // (2 * STEP)) * STEP

        # Shift short_K farther OTM
        if spread_type == "bull_put":
            trial_short_K = base_short_K - mid_offset
            trial_long_K = trial_short_K - wing_width
        else:  # bear_call
            trial_short_K = base_short_K + mid_offset
            trial_long_K = trial_short_K + wing_width

        candidate = evaluate_spread_config(
            spot, trial_short_K, trial_long_K, T_years, sigma,
            lot_size, spread_type, r, q,
        )
        margin = compute_margin_per_lot(candidate["max_loss_pts"], lot_size)
        lots = compute_lots_tradeable(capital_inr, capital_at_risk_pct, margin)
        req_pts = compute_required_premium_pts(capital_inr, target_return, lots, lot_size)

        if lots > 0 and candidate["premium_pts"] >= req_pts:
            # Feasible — push farther OTM
            best_config = candidate
            best_offset = mid_offset
            best_lots = lots
            best_margin = margin
            lo = mid_offset + STEP
        else:
            # Not feasible — pull back
            hi = mid_offset - STEP

    # Step 3: Final lots/margin/return at best_offset
    final_lots = best_lots
    final_margin = best_margin
    final_req = compute_required_premium_pts(capital_inr, target_return, final_lots, lot_size)
    projected_return = (best_config["premium_pts"] * final_lots * lot_size) / capital_inr

    # Step 4: Probability constraint check
    status = "ok"
    warn = None
    pop = best_config.get("pop_pct")

    if pop is not None and pop < min_pop:
        status = "partial"
        warn = f"POP {pop:.1%} < MIN_POP {min_pop:.1%}. Target too aggressive for safe strikes."

    return {
        "status": status,
        "short_strike": best_config["short_strike"],
        "long_strike": best_config["long_strike"],
        "extra_offset_pts": best_offset,
        "premium_pts": best_config["premium_pts"],
        "max_loss_pts": best_config["max_loss_pts"],
        "rr_ratio": best_config["rr_ratio"],
        "pop_pct": best_config["pop_pct"],
        "capital_summary": {
            "lots": final_lots,
            "margin_per_lot_inr": round(final_margin, 2),
            "total_margin_inr": round(final_margin * final_lots, 2),
            "required_premium_pts": round(final_req, 4),
            "achieved_premium_pts": round(best_config["premium_pts"], 4),
            "target_return_pct": round(target_return * 100, 4),
            "projected_return_pct": round(projected_return * 100, 4),
            "expiry_mode": cfg["expiry_mode"],
            "dte_days": dte_days,
        },
        "infeasibility_reason": warn,
    }


def _infeasible_result(cfg, dte_days, target_return, reason) -> dict:
    """Return infeasible result shape."""
    return {
        "status": "infeasible",
        "short_strike": None,
        "long_strike": None,
        "extra_offset_pts": 0,
        "premium_pts": None,
        "max_loss_pts": None,
        "rr_ratio": None,
        "pop_pct": None,
        "capital_summary": {
            "capital_inr": cfg["capital_inr"],
            "capital_at_risk_pct": cfg["capital_at_risk_pct"],
            "target_return_pct": round(target_return * 100, 4),
            "dte_days": dte_days,
            "expiry_mode": cfg["expiry_mode"],
        },
        "infeasibility_reason": reason,
    }


if __name__ == "__main__":
    # Unit test scenarios
    import json

    cfg = _load_capital_config()
    spot = 24171.75
    base_short_put = 24350
    base_short_call = 24750
    wing_width = 400
    T_years = 5 / 365

    sigma_weekly = 0.183

    print("\n=== Scenario 1: Bull Put, Generous Capital ===")
    result1 = find_safest_viable_spread(
        spot=spot,
        base_short_K=base_short_put,
        wing_width=wing_width,
        T_years=T_years,
        sigma=sigma_weekly,
        spread_type="bull_put",
        lot_size=cfg.nifty_lot_size,
        capital_config=cfg,
    )
    print(json.dumps(result1, indent=2, default=str))

    print("\n=== Scenario 2: Bear Call, Generous Capital ===")
    result2 = find_safest_viable_spread(
        spot=spot,
        base_short_K=base_short_call,
        wing_width=wing_width,
        T_years=T_years,
        sigma=sigma_weekly,
        spread_type="bear_call",
        lot_size=cfg.nifty_lot_size,
        capital_config=cfg,
    )
    print(json.dumps(result2, indent=2, default=str))

    print("\n=== Scenario 3: Bull Put, Very Tight Capital ===")
    tight_cfg = cfg.copy()
    tight_cfg["capital_inr"] = 50_000
    tight_cfg["target_return_weekly"] = 0.10
    result3 = find_safest_viable_spread(
        spot=spot,
        base_short_K=base_short_put,
        wing_width=wing_width,
        T_years=T_years,
        sigma=sigma_weekly,
        spread_type="bull_put",
        lot_size=cfg.nifty_lot_size,
        capital_config=tight_cfg,
    )
    print(json.dumps(result3, indent=2, default=str))

    print("\nUnit tests complete")
