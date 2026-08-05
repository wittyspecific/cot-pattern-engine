import numpy as np
import pandas as pd
from pattern_engine.divergence import analyze_noncommercial_divergence


def _base(n=180):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-07", periods=n, freq="W-TUE"),
        "close": 100 + np.cumsum(rng.normal(0, 0.2, n)),
        "noncomm_long": 200_000 + np.cumsum(rng.normal(0, 500, n)),
        "noncomm_short": 100_000 + np.cumsum(rng.normal(0, 500, n)),
    })


def test_active_bearish_divergence_from_new_shorts():
    df = _base()
    idx = df.index[-5:]
    df.loc[idx, "close"] = [115, 117, 119, 121, 124]
    df.loc[idx, "noncomm_short"] = [100000, 105000, 112000, 122000, 138000]
    df.loc[idx, "noncomm_long"] = [200000, 200200, 200300, 200400, 200500]
    result = analyze_noncommercial_divergence(df, horizons=(4,), min_price_move=0.02, min_flow_z=1.0)
    assert result["available"]
    assert result["signal"] == "bearish"
    assert result["horizons"][0]["mode"] == "active_dominant"
    assert result["horizons"][0]["dominance"] >= 0.70


def test_long_liquidation_alone_is_not_divergence():
    df = _base()
    idx = df.index[-5:]
    df.loc[idx, "close"] = [115, 117, 119, 121, 124]
    df.loc[idx, "noncomm_long"] = [200000, 195000, 187000, 176000, 160000]
    df.loc[idx, "noncomm_short"] = 100000
    result = analyze_noncommercial_divergence(df, horizons=(4,), min_price_move=0.02, min_flow_z=1.0)
    assert result["signal"] == "none"
