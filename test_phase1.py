#!/usr/bin/env python3
"""
Phase 1 Implementation Validation Test
========================================

Tests:
1. HMM Regime Detection module imports and basic functionality
2. Volatility Forecast module imports and basic functionality
3. Data Pipeline extended functions
4. Feature Engineering with regime features
5. Pipeline orchestration with new modes
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

def test_imports():
    """Test all new modules import successfully."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    
    try:
        import engine_regime_detection
        print("✓ engine_regime_detection imported")
    except Exception as e:
        print(f"✗ engine_regime_detection: {e}")
        return False
    
    try:
        import engine_volatility_forecast
        print("✓ engine_volatility_forecast imported")
    except Exception as e:
        print(f"✗ engine_volatility_forecast: {e}")
        return False
    
    try:
        from module1_data_pipeline import estimate_market_breadth, aggregate_to_period
        print("✓ module1_data_pipeline extended functions imported")
    except Exception as e:
        print(f"✗ module1_data_pipeline extended: {e}")
        return False
    
    try:
        from run_pipeline import (
            mode_regime_train,
            mode_forecast_vol,
            mode_analyze_regime,
        )
        print("✓ run_pipeline new modes imported")
    except Exception as e:
        print(f"✗ run_pipeline new modes: {e}")
        return False
    
    print()
    return True


def test_regime_detection():
    """Test regime detection functions."""
    print("=" * 60)
    print("TEST 2: Regime Detection Functions")
    print("=" * 60)
    
    try:
        from engine_regime_detection import (
            prepare_regime_features,
            extract_hmm_features,
            classify_vol_regime,
        )
        import pandas as pd
        import numpy as np
        
        # Create synthetic data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "close": np.linspace(18000, 19000, 100),
            "high": np.linspace(18100, 19100, 100),
            "low": np.linspace(17900, 18900, 100),
            "atr": np.linspace(100, 120, 100),
            "vix": np.linspace(14, 16, 100),
        }, index=dates)
        
        # Test feature preparation
        reg_df = prepare_regime_features(df)
        print(f"✓ prepare_regime_features: {len(reg_df)} rows processed")
        
        # Test feature extraction
        features = extract_hmm_features(reg_df)
        print(f"✓ extract_hmm_features: {features.shape[0]} x {features.shape[1]}")
        
        # Test vol regime classification
        regime_low = classify_vol_regime(0.10)
        regime_high = classify_vol_regime(0.30)
        print(f"✓ classify_vol_regime: 10% → {regime_low}, 30% → {regime_high}")
        
    except Exception as e:
        print(f"✗ Regime detection functions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_volatility_forecast():
    """Test volatility forecasting functions."""
    print("=" * 60)
    print("TEST 3: Volatility Forecast Functions")
    print("=" * 60)
    
    try:
        from engine_volatility_forecast import (
            forecast_volatility_garch,
            forecast_volatility_ewma,
            forecast_volatility_ensemble,
            classify_vol_regime,
            estimate_vol_term_structure,
        )
        import pandas as pd
        import numpy as np
        
        # Create synthetic returns
        returns = pd.Series(np.random.normal(0, 0.01, 200))
        
        # Test individual forecasts
        garch_vol = forecast_volatility_garch(returns, periods_ahead=7)
        print(f"✓ GARCH forecast: {len(garch_vol)} periods, vol[0]={garch_vol[0]:.4f}")
        
        ewma_vol = forecast_volatility_ewma(returns, periods_ahead=7)
        print(f"✓ EWMA forecast: {len(ewma_vol)} periods, vol[0]={ewma_vol[0]:.4f}")
        
        # Test ensemble
        ensemble = forecast_volatility_ensemble(returns, periods_ahead=7)
        print(f"✓ Ensemble forecast: {len(ensemble)} models, ensemble vol[0]={ensemble['ensemble'][0]:.4f}")
        
        # Test term structure
        ts = estimate_vol_term_structure(ensemble)
        print(f"✓ Vol term structure: {len(ts)} DTEs")
        
    except Exception as e:
        print(f"✗ Volatility forecast functions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_data_pipeline_extensions():
    """Test data pipeline extensions."""
    print("=" * 60)
    print("TEST 4: Data Pipeline Extensions")
    print("=" * 60)
    
    try:
        from module1_data_pipeline import (
            estimate_market_breadth,
            aggregate_to_period,
        )
        import pandas as pd
        import numpy as np
        
        # Test breadth estimation (will use defaults if no data)
        breadth = estimate_market_breadth()
        print(f"✓ Market breadth: {breadth.get('breadth_sentiment', 'unknown')}")
        
        # Test aggregation
        df = pd.DataFrame({
            "open": [100, 102, 101, 103],
            "high": [102, 104, 103, 105],
            "low": [99, 101, 100, 102],
            "close": [101, 103, 102, 104],
            "volume": [1000, 1100, 1050, 1150],
        }, index=pd.date_range("2023-01-02", periods=4, freq="D"))
        
        agg = aggregate_to_period(df, period="W-FRI")
        print(f"✓ Aggregate to period: {len(agg)} weeks from 4 days")
        
    except Exception as e:
        print(f"✗ Data pipeline extensions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  PHASE 1 IMPLEMENTATION VALIDATION TEST  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    all_pass = True
    
    all_pass &= test_imports()
    all_pass &= test_regime_detection()
    all_pass &= test_volatility_forecast()
    all_pass &= test_data_pipeline_extensions()
    
    print("=" * 60)
    if all_pass:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    print()
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
