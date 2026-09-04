import sys
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from module11_option_chain import _fetch_raw_jugaad, _parse_expiry

def generate_volatility_surface():
    print("Fetching live option chain from NSE...")
    try:
        raw_recs, spot, expiries = _fetch_raw_jugaad()
    except Exception as e:
        print(f"Failed to fetch live data: {e}")
        return

    # Data arrays for Plotly
    dates = []
    strikes = []
    call_ivs = []
    put_ivs = []
    
    print(f"Live Spot: {spot}. Extracting IV for first 4 expiries...")
    
    # Take the first 4 expiries to build the "wave" down the time axis
    for exp_str in expiries[:4]:
        try:
            exp_date = datetime.strptime(exp_str, "%d-%b-%Y").date()
        except Exception:
            continue
            
        dte = max((exp_date - datetime.today().date()).days, 1)
        chain = _parse_expiry(raw_recs, exp_date)
        
        for k, v in chain.items():
            # Filter strikes to a reasonable window around spot (e.g., +/- 8%)
            if 0.92 * spot <= k <= 1.08 * spot:
                # Use the NSE provided IV if it is valid
                if v["call_iv"] > 0.01 and v["put_iv"] > 0.01:
                    dates.append(dte)
                    strikes.append(k)
                    call_ivs.append(v["call_iv"] * 100) # Convert to percentage
                    put_ivs.append(v["put_iv"] * 100)
                    
    if not dates:
        print("No valid IV data found near spot.")
        return
        
    print(f"Plotting {len(dates)} data points...")
    fig = go.Figure()
    
    # 1. Add Call Volatility Surface (Green)
    fig.add_trace(go.Mesh3d(
        x=dates,
        y=strikes,
        z=call_ivs,
        opacity=0.6,
        color='lightgreen',
        name='Calls (Smile)'
    ))
    
    # 2. Add Put Volatility Surface (Red)
    fig.add_trace(go.Mesh3d(
        x=dates,
        y=strikes,
        z=put_ivs,
        opacity=0.6,
        color='crimson',
        name='Puts (Skew)'
    ))
    
    # Format the graph
    fig.update_layout(
        title=f"Nifty 50 Live Volatility Surface (Spot: {spot})",
        scene=dict(
            xaxis_title='Days to Expiration (Time)',
            yaxis_title='Strike Price (Moneyness)',
            zaxis_title='Implied Volatility (%)',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.5) # Default nice viewing angle
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    html_path = output_dir / "nifty_volatility_surface.html"
    
    # Save as interactive HTML
    fig.write_html(str(html_path))
    print(f"\nSUCCESS! Interactive 3D Surface saved to: {html_path}")
    print("Open this file in any web browser to view the wave!")

if __name__ == "__main__":
    generate_volatility_surface()
