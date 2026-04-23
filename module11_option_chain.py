"""
Module 11: NSE Option Chain Integration
Fetches live OI, IV, and price data via jugaad-data (primary) or raw NSE API (fallback).
Calculates Max Pain, ATM Implied Volatility (IV), and Put-Call Ratio (PCR).
"""

import json
import os
from datetime import date, datetime
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "option_chain_latest.json"
CACHE_TTL  = 15  # minutes

# ---------------------------------------------------------------------------
# Primary fetch via jugaad-data
# ---------------------------------------------------------------------------

def _fetch_raw_jugaad() -> tuple[list[dict], float, list[str]]:
    """
    Returns (records, underlying_spot, expiry_dates_list).
    records: list of {strikePrice, CE:{...}, PE:{...}, expiryDates}
    """
    from jugaad_data.nse import NSELive
    n = NSELive()
    data = n.index_option_chain("NIFTY")
    recs  = data["records"]
    return recs["data"], float(recs["underlyingValue"]), recs["expiryDates"]


# ---------------------------------------------------------------------------
# Fallback fetch via raw NSE API
# ---------------------------------------------------------------------------

def _fetch_raw_nse() -> tuple[list[dict], float, list[str]]:
    """Raw NSE API fallback (Akamai may block outside market hours)."""
    import requests, time

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
    }
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get("https://www.nseindia.com/option-chain", timeout=10)
    time.sleep(1)
    resp = s.get(
        "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data or "records" not in data:
        raise ValueError("NSE API returned empty/invalid response")
    recs = data["records"]
    expiry_dates = recs.get("expiryDates", [])
    spot = float(recs.get("underlyingValue", 0))
    return recs["data"], spot, expiry_dates


# ---------------------------------------------------------------------------
# Parse: build strikes dict for one expiry
# ---------------------------------------------------------------------------

def _parse_expiry(raw: list[dict], expiry_date: date) -> dict:
    """
    Filter raw records for a specific expiry date.
    jugaad-data CE/PE records use expiryDate='28-04-2026' (dd-mm-yyyy).
    expiry_dates list uses '28-Apr-2026' (dd-Mon-yyyy).
    We match against both formats.
    """
    # Build both formats for comparison
    fmt_numeric = expiry_date.strftime("%d-%m-%Y")  # 28-04-2026
    fmt_alpha   = f"{expiry_date.day}-{expiry_date.strftime('%b-%Y')}"  # 28-Apr-2026

    strikes = {}
    for rec in raw:
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})
        # Use whichever leg has expiryDate
        rec_expiry = ce.get("expiryDate") or pe.get("expiryDate") or rec.get("expiryDate", "")

        if rec_expiry not in (fmt_numeric, fmt_alpha):
            continue

        k = int(rec["strikePrice"])
        strikes[k] = {
            "call_oi": int(ce.get("openInterest", 0)),
            "put_oi":  int(pe.get("openInterest", 0)),
            "call_iv": float(ce.get("impliedVolatility", 0)) / 100.0,
            "put_iv":  float(pe.get("impliedVolatility", 0)) / 100.0,
        }
    return strikes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_max_pain(strikes_data: dict) -> int:
    """Max pain = expiry price where total option holder payout is minimized."""
    ks = sorted(strikes_data.keys())
    min_payout = float("inf")
    max_pain_strike = ks[0]
    for candidate in ks:
        payout = 0
        for s, row in strikes_data.items():
            payout += max(s - candidate, 0) * row["put_oi"]
            payout += max(candidate - s, 0) * row["call_oi"]
        if payout < min_payout:
            min_payout = payout
            max_pain_strike = candidate
    return int(round(max_pain_strike / 50) * 50)


def get_atm_iv(strikes_data: dict, spot: float) -> float | None:
    """Average IV of call+put at ATM strike (±50 fallback)."""
    atm = int(round(spot / 50) * 50)
    row = (
        strikes_data.get(atm)
        or strikes_data.get(atm - 50)
        or strikes_data.get(atm + 50)
    )
    if not row:
        return None
    iv = (row["call_iv"] + row["put_iv"]) / 2
    return iv if iv > 0.01 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_option_chain(expiry_date: date | None = None) -> dict | None:
    """
    Fetch NSE NIFTY option chain.
    Returns standardised dict or None on failure (non-blocking).
    Uses jugaad-data as primary source; falls back to raw NSE API.
    """
    # 1. Check cache
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            age = (datetime.now() - datetime.fromisoformat(cached["fetched_at"])).total_seconds() / 60
            if age < CACHE_TTL:
                logger.info(f"OI: using cache ({age:.1f} min old)")
                return cached
        except Exception as e:
            logger.debug(f"Cache read error: {e}")

    # 2. Fetch raw data (jugaad-data → NSE API fallback)
    raw_records = spot = expiry_dates = None
    for source, fetch_fn in [("jugaad-data", _fetch_raw_jugaad), ("nse-api", _fetch_raw_nse)]:
        try:
            raw_records, spot, expiry_dates = fetch_fn()
            if raw_records:
                logger.info(f"OI: fetched via {source} ({len(raw_records)} records)")
                break
        except Exception as e:
            logger.warning(f"OI: {source} failed: {e}")

    if not raw_records:
        logger.error("OI: all sources failed — skipping OI enrichment")
        return None

    # 3. Resolve expiry date
    if expiry_date is None:
        if expiry_dates:
            # expiry_dates list is '28-Apr-2026' format
            expiry_date = datetime.strptime(expiry_dates[0], "%d-%b-%Y").date()
        else:
            logger.error("OI: no expiry dates in response")
            return None

    # 4. Parse strikes for this expiry
    strikes_data = _parse_expiry(raw_records, expiry_date)
    if not strikes_data:
        logger.error(f"OI: no strikes found for expiry {expiry_date}")
        return None

    # 5. Compute metrics
    atm_iv   = get_atm_iv(strikes_data, spot)
    max_pain = calculate_max_pain(strikes_data)

    total_call_oi = sum(v["call_oi"] for v in strikes_data.values())
    total_put_oi  = sum(v["put_oi"]  for v in strikes_data.values())
    pcr = round(total_put_oi / total_call_oi, 4) if total_call_oi > 0 else 1.0

    result = {
        "expiry":       expiry_date.isoformat(),
        "fetched_at":   datetime.now().isoformat(timespec="seconds"),
        "atm_strike":   int(round(spot / 50) * 50),
        "atm_iv":       round(atm_iv, 4) if atm_iv else None,
        "pcr":          pcr,
        "max_pain":     max_pain,
        "spot":         round(spot, 2),
        "expiry_dates": expiry_dates,  # full NSE expiry schedule for downstream cross-check
        "strikes":      {str(k): v for k, v in strikes_data.items()},
    }

    # 6. Cache
    CACHE_FILE.write_text(json.dumps(result, indent=2))
    logger.info(
        f"OI: ATM IV={f'{atm_iv:.1%}' if atm_iv else 'N/A'}  "
        f"PCR={pcr:.2f}  MaxPain={max_pain}  Spot={spot:,.0f}"
    )
    return result


if __name__ == "__main__":
    res = fetch_option_chain()
    if res:
        print(f"Expiry  : {res['expiry']}")
        print(f"Spot    : {res['spot']:,.2f}")
        atm_iv_str = f"{res['atm_iv']:.1%}" if res['atm_iv'] else 'N/A'
        print(f"ATM IV  : {atm_iv_str}")
        print(f"PCR     : {res['pcr']:.2f}")
        print(f"Max Pain: {res['max_pain']:,}")
        print(f"Strikes : {len(res['strikes'])} entries")
    else:
        print("Failed to fetch option chain.")
