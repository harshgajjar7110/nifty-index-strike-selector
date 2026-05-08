"""
Module 11b: NSE India Option Chain Scraper
Robust wrapper around jugaad-data / nsepython / raw NSE API
with retry logic and fallback chains.

Usage:
    from module11b_nse_scraper import fetch_nse_option_chain
    data = fetch_nse_option_chain("NIFTY")
    # data = {
    #   "spot": 24190.1,
    #   "expiry_dates": ["12-May-2026", "19-May-2026", ...],
    #   "strikes": {
    #       24200: {"call_oi": 15000, "put_oi": 8200, "call_iv": 0.165, "put_iv": 0.158},
    #       ...
    #   },
    #   "timestamp": "2026-05-08T10:30:00"
    # }
"""

import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from loguru import logger

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "option_chain_nse_latest.json"
CACHE_TTL_MINUTES = 15


def _fetch_jugaad(symbol: str = "NIFTY") -> Optional[dict]:
    """Primary fetcher via jugaad-data (most reliable)."""
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        data = n.index_option_chain(symbol)
        recs = data["records"]
        return {
            "spot": float(recs["underlyingValue"]),
            "expiry_dates": recs.get("expiryDates", []),
            "raw_data": recs["data"],
            "source": "jugaad-data",
        }
    except Exception as e:
        logger.warning(f"jugaad-data fetch failed: {e}")
        return None


def _fetch_nsepython(symbol: str = "NIFTY") -> Optional[dict]:
    """Secondary fetcher via nsepython library."""
    try:
        from nsepython import nse_optionchain_scrapper
        data = nse_optionchain_scrapper(symbol)
        if not data or "records" not in data:
            return None
        recs = data["records"]
        return {
            "spot": float(recs.get("underlyingValue", 0)),
            "expiry_dates": recs.get("expiryDates", []),
            "raw_data": recs.get("data", []),
            "source": "nsepython",
        }
    except Exception as e:
        logger.warning(f"nsepython fetch failed: {e}")
        return None


def _fetch_raw_nse(symbol: str = "NIFTY") -> Optional[dict]:
    """Tertiary fetcher via raw NSE API (often blocked by Akamai)."""
    import requests

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }

    s = requests.Session()
    s.headers.update(HEADERS)

    try:
        # Step 1: Get homepage for cookies
        r1 = s.get("https://www.nseindia.com/option-chain", timeout=15)
        if r1.status_code != 200:
            return None
        time.sleep(2)

        # Step 2: Fetch API
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        r2 = s.get(url, timeout=20)
        if r2.status_code != 200:
            return None

        data = r2.json()
        if not data or "records" not in data:
            return None

        recs = data["records"]
        return {
            "spot": float(recs.get("underlyingValue", 0)),
            "expiry_dates": recs.get("expiryDates", []),
            "raw_data": recs.get("data", []),
            "source": "nse-api",
        }
    except Exception as e:
        logger.warning(f"Raw NSE API fetch failed: {e}")
        return None


def _parse_strikes(raw_data: list, expiry_date_str: str) -> dict:
    """Parse raw NSE option chain data into strike-level OI/IV map."""
    from datetime import datetime

    # Normalize expiry date string to handle both formats
    # Input could be "12-May-2026" (from expiryDates list) or "12-05-2026" (from record)
    def _normalize_expiry(s: str) -> str:
        """Convert both '12-May-2026' and '12-05-2026' to standardized form."""
        for fmt in ("%d-%b-%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt).strftime("%d-%m-%Y")
            except ValueError:
                continue
        return s

    target = _normalize_expiry(expiry_date_str)

    strikes = {}
    for rec in raw_data:
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})

        rec_expiry = ce.get("expiryDate") or pe.get("expiryDate") or ""
        if _normalize_expiry(rec_expiry) != target:
            continue

        k = int(rec["strikePrice"])
        strikes[k] = {
            "call_oi": int(ce.get("openInterest", 0)),
            "put_oi": int(pe.get("openInterest", 0)),
            "call_iv": float(ce.get("impliedVolatility", 0)) / 100.0,
            "put_iv": float(pe.get("impliedVolatility", 0)) / 100.0,
            "call_ltp": float(ce.get("lastPrice", 0)),
            "put_ltp": float(pe.get("lastPrice", 0)),
            "call_volume": int(ce.get("totalTradedVolume", 0)),
            "put_volume": int(pe.get("totalTradedVolume", 0)),
        }
    return strikes


def _calculate_metrics(strikes: dict, spot: float) -> dict:
    """Calculate ATM IV, PCR, and Max Pain from strike data."""
    if not strikes:
        return {}

    # ATM IV (average of call+put at nearest strike)
    atm_strike = int(round(spot / 50) * 50)
    atm_row = (
        strikes.get(atm_strike)
        or strikes.get(atm_strike - 50)
        or strikes.get(atm_strike + 50)
    )
    atm_iv = None
    if atm_row:
        iv = (atm_row["call_iv"] + atm_row["put_iv"]) / 2
        atm_iv = iv if iv > 0.01 else None

    # PCR (put OI / call OI for all strikes)
    total_call_oi = sum(s["call_oi"] for s in strikes.values())
    total_put_oi = sum(s["put_oi"] for s in strikes.values())
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else None

    # Max Pain
    ks = sorted(strikes.keys())
    min_payout = float("inf")
    max_pain = ks[0]
    for candidate in ks:
        payout = 0
        for s, row in strikes.items():
            payout += max(s - candidate, 0) * row["put_oi"]
            payout += max(candidate - s, 0) * row["call_oi"]
        if payout < min_payout:
            min_payout = payout
            max_pain = candidate

    return {
        "atm_iv": atm_iv,
        "pcr": round(pcr, 2) if pcr else None,
        "max_pain": int(round(max_pain / 50) * 50),
    }


def fetch_nse_option_chain(
    symbol: str = "NIFTY",
    use_cache: bool = True,
    cache_ttl_minutes: int = 15,
) -> Optional[dict]:
    """
    Fetch NSE option chain with automatic retry and fallback.

    Returns standardized dict or None on failure.
    """
    # 1. Check cache
    if use_cache and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            age = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds() / 60
            if age < cache_ttl_minutes:
                logger.info(f"NSE scraper: using cache ({age:.1f} min old)")
                return cached
        except Exception as e:
            logger.debug(f"Cache read error: {e}")

    # 2. Try fetchers in order
    raw_result = None
    for source_name, fetch_fn in [
        ("jugaad-data", _fetch_jugaad),
        ("nsepython", _fetch_nsepython),
        ("nse-api", _fetch_raw_nse),
    ]:
        try:
            raw_result = fetch_fn(symbol)
            if raw_result and raw_result.get("raw_data"):
                logger.info(f"NSE scraper: fetched via {source_name} ({len(raw_result['raw_data'])} records)")
                break
        except Exception as e:
            logger.warning(f"NSE scraper: {source_name} failed: {e}")

    if not raw_result:
        logger.error("NSE scraper: all sources failed")
        return None

    # 3. Parse nearest expiry
    expiry_dates = raw_result.get("expiry_dates", [])
    if not expiry_dates:
        logger.error("NSE scraper: no expiry dates found")
        return None

    nearest_expiry = expiry_dates[0]
    strikes = _parse_strikes(raw_result["raw_data"], nearest_expiry)

    if not strikes:
        logger.error(f"NSE scraper: no strikes found for expiry {nearest_expiry}")
        return None

    # 4. Calculate metrics
    spot = raw_result["spot"]
    metrics = _calculate_metrics(strikes, spot)

    result = {
        "spot": round(spot, 2),
        "expiry_date": nearest_expiry,
        "expiry_dates": expiry_dates,
        "strikes": strikes,
        "atm_iv": metrics.get("atm_iv"),
        "pcr": metrics.get("pcr"),
        "max_pain": metrics.get("max_pain"),
        "source": raw_result.get("source"),
        "timestamp": datetime.now().isoformat(),
    }

    # 5. Save cache
    CACHE_FILE.write_text(json.dumps(result, indent=2, default=str))
    logger.info(f"NSE scraper: saved cache ({len(strikes)} strikes)")

    return result


if __name__ == "__main__":
    result = fetch_nse_option_chain()
    if result:
        print(f"Spot: {result['spot']}")
        print(f"ATM IV: {result['atm_iv']}")
        print(f"PCR: {result['pcr']}")
        print(f"Max Pain: {result['max_pain']}")
        print(f"Strikes: {len(result['strikes'])}")
        print(f"Source: {result['source']}")
    else:
        print("Failed to fetch option chain")
