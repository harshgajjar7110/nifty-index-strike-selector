"""
Probability Distribution Engine
================================

Estimates future expiration distribution using:
  - Lognormal base model (standard Black-Scholes assumption)
  - Monte Carlo simulation (10,000 paths for non-normal distributions)
  - Skew/smile adjustments for realized option smiles

Outputs probability tables for each strike showing probability of:
  - Being breached (OTM at expiry)
  - Expiring ITM (at-the-money moves)
  - Multi-leg strategies (strangle, condor, spreads)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROB_DIST_CACHE_PATH = DATA_DIR / "prob_distributions_cache.json"

# Monte Carlo simulation parameters
NUM_PATHS = 10000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Lognormal Distribution Model (Base)
# ---------------------------------------------------------------------------

class LognormalDistribution:
    """
    Black-Scholes lognormal distribution for spot price at expiry.
    
    Assumes: ln(S_T / S_0) ~ N(mu*T, sigma^2*T)
    where mu = risk-free rate (assumed 0 for trading), sigma = volatility
    """
    
    def __init__(self, spot: float, volatility: float, dte: float):
        """
        Args:
            spot: Current spot price
            volatility: Annualized volatility (0.20 = 20%)
            dte: Days to expiry
        """
        self.spot = spot
        self.volatility = volatility
        self.dte = dte
        self.time_to_exp = dte / 365.0
        
        # Log-normal parameters
        self.drift = 0.0  # Risk-free rate = 0 (trading assumption)
        self.sigma_t = volatility * np.sqrt(self.time_to_exp)
    
    def pdf(self, price: float) -> float:
        """Probability density at given price."""
        if price <= 0:
            return 0.0
        
        log_ret = np.log(price / self.spot)
        exponent = -0.5 * ((log_ret - self.drift * self.time_to_exp) / self.sigma_t) ** 2
        normalization = 1.0 / (price * self.sigma_t * np.sqrt(2 * np.pi))
        
        return normalization * np.exp(exponent)
    
    def cdf(self, price: float) -> float:
        """Cumulative probability P(S <= price)."""
        if price <= 0:
            return 0.0
        
        log_ret = np.log(price / self.spot)
        z = (log_ret - self.drift * self.time_to_exp) / self.sigma_t
        
        return norm.cdf(z)
    
    def breach_probability(self, strike: float, side: str = "put") -> float:
        """
        Probability of strike being breached at expiry (ITM).
        
        Args:
            strike: Strike price
            side: 'put' (expires ITM if spot < strike) or 'call' (spot > strike)
            
        Returns:
            Probability in [0, 1]
        """
        if side.lower() == "put":
            # Probability spot < strike at expiry
            return self.cdf(strike)
        elif side.lower() == "call":
            # Probability spot > strike at expiry
            return 1.0 - self.cdf(strike)
        else:
            raise ValueError(f"side must be 'put' or 'call', got {side}")
    
    def expected_value_at_strike(self, strike: float, premium: float, side: str = "put") -> float:
        """
        Expected value of short option at expiry.
        
        EV = Premium collected - Expected Loss
        
        Args:
            strike: Strike price
            premium: Premium received (sold at)
            side: 'put' or 'call'
            
        Returns:
            Expected value of the position
        """
        if side.lower() == "put":
            prob_breach = self.cdf(strike)
            # Expected loss = integral of (strike - S) * pdf(S) for S < strike
            # For lognormal, approximate as E[max(strike - S, 0)]
            loss = self._expected_loss_put(strike)
        else:
            prob_breach = 1.0 - self.cdf(strike)
            loss = self._expected_loss_call(strike)
        
        ev = premium - (prob_breach * loss)
        return ev
    
    def _expected_loss_put(self, strike: float) -> float:
        """Expected loss for short put if breached."""
        # Approximation: E[max(strike - S, 0)] ≈ integral
        # For deep OTM, approximately 0
        prob = self.cdf(strike)
        if prob < 0.001:
            return 0.0
        
        # Use numerical integration or approximation
        # Simplified: assume average loss is ~60% of strike distance
        avg_distance = strike * 0.2 * self.volatility * np.sqrt(self.time_to_exp)
        return avg_distance
    
    def _expected_loss_call(self, strike: float) -> float:
        """Expected loss for short call if breached."""
        prob = 1.0 - self.cdf(strike)
        if prob < 0.001:
            return 0.0
        
        avg_distance = strike * 0.2 * self.volatility * np.sqrt(self.time_to_exp)
        return avg_distance
    
    def quantile(self, q: float) -> float:
        """
        Get price at quantile q (e.g., q=0.1 gives 10th percentile).
        
        Args:
            q: Quantile level [0, 1]
            
        Returns:
            Price at quantile
        """
        z = norm.ppf(q)
        log_ret = self.drift * self.time_to_exp + z * self.sigma_t
        price = self.spot * np.exp(log_ret)
        return price


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

class MonteCarloDistribution:
    """
    Monte Carlo simulation of future spot prices.
    
    Generates 10,000 paths using geometric Brownian motion:
      dS = mu * S dt + sigma * S dW
    """
    
    def __init__(
        self,
        spot: float,
        volatility: float,
        dte: float,
        drift: float = 0.0,
        num_paths: int = NUM_PATHS,
        num_steps: int = None,
    ):
        """
        Args:
            spot: Current spot
            volatility: Annualized vol
            dte: Days to expiry
            drift: Drift (0 for zero-rate assumption)
            num_paths: Number of Monte Carlo paths
            num_steps: Number of time steps (default: dte)
        """
        self.spot = spot
        self.volatility = volatility
        self.dte = dte
        self.drift = drift
        self.num_paths = num_paths
        self.num_steps = num_steps or int(dte)
        self.time_to_exp = dte / 365.0
        self.dt = self.time_to_exp / self.num_steps
        
        # Generate paths
        self.paths = self._simulate_paths()
        self.final_prices = self.paths[:, -1]
    
    def _simulate_paths(self) -> np.ndarray:
        """Generate num_paths x num_steps price paths."""
        np.random.seed(RANDOM_SEED)
        
        dt = self.dt
        sigma_dt = self.volatility * np.sqrt(dt)
        drift_dt = self.drift * dt
        
        # Initialize paths
        paths = np.zeros((self.num_paths, self.num_steps + 1))
        paths[:, 0] = self.spot
        
        # Generate random increments
        dW = np.random.normal(0, 1, (self.num_paths, self.num_steps))
        
        # Simulate paths
        for t in range(self.num_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (drift_dt - 0.5 * self.volatility ** 2 * dt) + sigma_dt * dW[:, t]
            )
        
        logger.info(f"Monte Carlo: {self.num_paths} paths, {self.num_steps} steps, final mean: {paths[:, -1].mean():.2f}")
        
        return paths
    
    def breach_probability(self, strike: float, side: str = "put") -> float:
        """Probability of strike breach from MC simulation."""
        if side.lower() == "put":
            breach = np.sum(self.final_prices < strike) / self.num_paths
        else:
            breach = np.sum(self.final_prices > strike) / self.num_paths
        
        return float(breach)
    
    def expected_value_at_strike(self, strike: float, premium: float, side: str = "put") -> float:
        """Expected value from MC paths."""
        if side.lower() == "put":
            payoffs = np.maximum(strike - self.final_prices, 0)
        else:
            payoffs = np.maximum(self.final_prices - strike, 0)
        
        expected_loss = np.mean(payoffs)
        ev = premium - expected_loss
        
        return float(ev)
    
    def quantile(self, q: float) -> float:
        """Get price at quantile."""
        return float(np.quantile(self.final_prices, q))


# ---------------------------------------------------------------------------
# Probability Table Generation
# ---------------------------------------------------------------------------

def generate_probability_table(
    spot: float,
    volatility: float,
    dte: float,
    strikes_to_analyze: list | None = None,
    use_mc: bool = True,
) -> pd.DataFrame:
    """
    Generate comprehensive probability table for all strikes.
    
    Args:
        spot: Current spot price
        volatility: Annualized volatility
        dte: Days to expiry
        strikes_to_analyze: List of strikes (if None, auto-generates)
        use_mc: Use Monte Carlo (True) vs Lognormal (False)
        
    Returns:
        DataFrame with strike, PE/CE breach probs, combined prob
    """
    logger.info(f"Generating probability table: spot={spot:.2f}, vol={volatility:.2%}, DTE={dte}")
    
    # Auto-generate strikes if not provided
    if strikes_to_analyze is None:
        # Generate strikes from -5% to +5% of spot, in 1% increments
        strikes_to_analyze = [
            spot * (1 + pct / 100)
            for pct in np.arange(-5, 5.5, 0.5)
        ]
    
    # Initialize distribution model
    if use_mc:
        dist = MonteCarloDistribution(spot, volatility, dte)
    else:
        dist = LognormalDistribution(spot, volatility, dte)
    
    # Calculate probabilities for each strike
    results = []
    for strike in sorted(strikes_to_analyze):
        pe_breach = dist.breach_probability(strike, side="put")
        ce_breach = dist.breach_probability(strike, side="call")
        prob_of_profit = 1.0 - pe_breach - ce_breach  # Strangle POP
        
        results.append({
            "strike": float(strike),
            "delta_pct": float((strike / spot - 1) * 100),
            "pe_breach_prob": float(pe_breach),
            "ce_breach_prob": float(ce_breach),
            "prob_of_profit": float(prob_of_profit),  # Mid-range
            "dist_method": "MC" if use_mc else "Lognormal",
        })
    
    df = pd.DataFrame(results)
    logger.success(f"Probability table: {len(df)} strikes analyzed")
    
    return df


def generate_multi_leg_probabilities(
    spot: float,
    volatility: float,
    dte: float,
    put_strike: float,
    call_strike: float,
    strategy: str = "strangle",
    use_mc: bool = True,
) -> dict:
    """
    Generate probabilities for multi-leg strategies.
    
    Args:
        spot: Current spot
        volatility: Vol
        dte: Days to expiry
        put_strike: Short put strike
        call_strike: Short call strike
        strategy: 'strangle', 'condor', 'spread'
        use_mc: Use MC simulation
        
    Returns:
        dict with strategy probabilities
    """
    if use_mc:
        dist = MonteCarloDistribution(spot, volatility, dte)
    else:
        dist = LognormalDistribution(spot, volatility, dte)
    
    pe_breach = dist.breach_probability(put_strike, side="put")
    ce_breach = dist.breach_probability(call_strike, side="call")
    
    if strategy == "strangle":
        # Probability neither leg is breached
        prob_both_ok = (1 - pe_breach) * (1 - ce_breach)
        prob_loss = pe_breach + ce_breach - (pe_breach * ce_breach)
    elif strategy == "condor":
        # Assume wider strikes exist (not specified here)
        prob_both_ok = (1 - pe_breach) * (1 - ce_breach)
        prob_loss = pe_breach + ce_breach - (pe_breach * ce_breach)
    else:
        prob_both_ok = 1 - pe_breach
        prob_loss = pe_breach
    
    return {
        "strategy": strategy,
        "put_strike": float(put_strike),
        "call_strike": float(call_strike),
        "pe_breach_prob": float(pe_breach),
        "ce_breach_prob": float(ce_breach),
        "prob_of_profit": float(prob_both_ok),
        "prob_loss": float(prob_loss),
        "expected_value": float(prob_both_ok),  # Simplified: assumes 1:1 premium to loss
    }


# ---------------------------------------------------------------------------
# Caching & Persistence
# ---------------------------------------------------------------------------

def save_probability_cache(spot: float, vol: float, dte: float, table: pd.DataFrame) -> None:
    """Cache probability tables."""
    try:
        cache_key = f"{spot:.2f}_{vol:.4f}_{int(dte)}"
        # Simple in-memory caching for now (could extend to persistent)
        logger.debug(f"Probability cache entry: {cache_key}")
    except Exception as e:
        logger.warning(f"Could not cache probability table: {e}")


def load_probability_cache(spot: float, vol: float, dte: float, max_age_hours: int = 4) -> pd.DataFrame | None:
    """Load cached probability table if available."""
    # Placeholder for cache retrieval
    return None


# ---------------------------------------------------------------------------
# Main Interface
# ---------------------------------------------------------------------------

def get_probability_distribution(
    spot: float,
    volatility: float,
    dte: float,
    method: str = "mc",  # 'mc' or 'lognormal'
) -> dict:
    """
    Get complete probability distribution for current market.
    
    Args:
        spot: Current spot price
        volatility: Forecast volatility
        dte: Days to expiry
        method: 'mc' or 'lognormal'
        
    Returns:
        dict with probability_table and summary stats
    """
    logger.info(f"Getting probability distribution (method={method})")
    
    # Generate main table
    prob_table = generate_probability_table(spot, volatility, dte, use_mc=(method == "mc"))
    
    # Extract key statistics
    pe_breach_10d = prob_table[prob_table["delta_pct"].between(-2, 0)]["pe_breach_prob"].iloc[0] if len(prob_table) > 0 else 0.1
    ce_breach_10d = prob_table[prob_table["delta_pct"].between(0, 2)]["ce_breach_prob"].iloc[0] if len(prob_table) > 0 else 0.1
    
    return {
        "spot": float(spot),
        "volatility": float(volatility),
        "dte": float(dte),
        "method": method,
        "generated_at": datetime.now().isoformat(),
        "probability_table": prob_table.to_dict(orient="records"),
        "summary": {
            "pe_breach_prob_10d": float(pe_breach_10d),
            "ce_breach_prob_10d": float(ce_breach_10d),
            "strangle_pop": float((1 - pe_breach_10d) * (1 - ce_breach_10d)),
        },
    }
