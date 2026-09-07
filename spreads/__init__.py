"""Credit spread subsystem. Use direct submodule imports to avoid circular imports."""

__all__ = [
    "breach_probability", "pop_from_chain_iv", "cvar",
    "generate_all_spreads", "generate_credit_spread",
    "estimate_spread_premium", "NIFTY_LOT_SIZE",
    "calculate_nse_charges", "apply_slippage", "estimate_ic_premium",
    "find_safest_viable_spread", "compute_margin_per_lot",
    "compute_lots_tradeable", "compute_required_premium_pts",
]

_LAZY = {
    "breach_probability": "spreads.module4b_risk",
    "pop_from_chain_iv": "spreads.module4b_risk",
    "cvar": "spreads.module4b_risk",
    "generate_all_spreads": "spreads.module9_spreads",
    "generate_credit_spread": "spreads.module9_spreads",
    "estimate_spread_premium": "spreads.module9_spreads",
    "NIFTY_LOT_SIZE": "spreads.module9_spreads",
    "calculate_nse_charges": "spreads.module10_nse_costs",
    "apply_slippage": "spreads.module10_nse_costs",
    "estimate_ic_premium": "spreads.module10_nse_costs",
    "find_safest_viable_spread": "spreads.module12_capital",
    "compute_margin_per_lot": "spreads.module12_capital",
    "compute_lots_tradeable": "spreads.module12_capital",
    "compute_required_premium_pts": "spreads.module12_capital",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(name)
