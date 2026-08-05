import numpy as np
import pandas as pd

from pattern_engine.divergence import analyze_noncommercial_divergence


def _base_frame(n=140):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-03", periods=n, freq="W-FRI"),
        "close": 100 + np.cumsum(rng.normal(0, 0.3, n)),
        "noncomm_long": 100000 + np.cumsum(rng.normal(0, 700, n)),
        "noncomm_short": 80000 + np.cumsum(rng.normal(0, 700, n)),
    })


def test_bearish_requires_dominant_short_building():
    df = _base_frame()
    df.loc[df.index[-9]:, "close"] = np.linspace(100, 110, 9)
    df.loc[df.index[-9]:, "noncomm_short"] = np.linspace(80000, 120000, 9)
    df.loc[df.index[-9]:, "noncomm_long"] = np.linspace(100000, 103000, 9)
    result = analyze_noncommercial_divergence(df, horizons=(1, 2, 4, 8), min_price_move=0.01, min_flow_z=0.5)
    assert result["signal"] == "bearish"
    assert any(i["signal"] == "bearish" and i["dominance"] >= 0.70 for i in result["horizons"])


def test_long_liquidation_alone_is_not_divergence():
    df = _base_frame()
    df.loc[df.index[-9]:, "close"] = np.linspace(100, 110, 9)
    df.loc[df.index[-9]:, "noncomm_long"] = np.linspace(120000, 80000, 9)
    df.loc[df.index[-9]:, "noncomm_short"] = 80000
    result = analyze_noncommercial_divergence(df, horizons=(1, 2, 4, 8), min_price_move=0.01, min_flow_z=0.5)
    assert result["signal"] == "none"


def test_balanced_buildup_fails_dominance_filter():
    df = _base_frame()
    df.loc[df.index[-9]:, "close"] = np.linspace(100, 110, 9)
    df.loc[df.index[-9]:, "noncomm_short"] = np.linspace(80000, 110000, 9)
    df.loc[df.index[-9]:, "noncomm_long"] = np.linspace(100000, 128000, 9)
    result = analyze_noncommercial_divergence(df, horizons=(1, 2, 4, 8), min_price_move=0.01, min_flow_z=0.5)
    assert result["signal"] == "none"
