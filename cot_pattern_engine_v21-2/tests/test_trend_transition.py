import numpy as np
import pandas as pd
from pattern_engine.trend_transition import analyze_trend_transition


def test_bullish_transition_detected():
    n = 180
    dates = pd.date_range("2020-01-07", periods=n, freq="W-TUE")
    x = np.arange(n, dtype=float)
    df = pd.DataFrame({
        "date": dates, "close": 100 + x * 0.05,
        "oi": np.full(n, 1000.0),
        "commercial_net": np.linspace(-100, 100, n),
        "noncommercial_net": np.linspace(100, -100, n),
        "nonreportable_net": np.linspace(40, -40, n),
        "comm_long": 300 + x, "comm_short": 300 - x * 0.2,
        "noncomm_long": 300 - x * 0.5, "noncomm_short": 200 + x * 0.5,
        "nonrep_long": 150 - x * 0.1, "nonrep_short": 100 + x * 0.1,
    })
    df.loc[n-4:, "noncomm_long"] += np.array([0, 15, 35, 60])
    df.loc[n-4:, "noncomm_short"] -= np.array([0, 10, 25, 45])
    df.loc[n-4:, "close"] += np.array([0, 0.5, 1.5, 3.0])
    result = analyze_trend_transition(df, flow_weeks=4, extreme_cutoff=85)
    assert result["available"]
    assert result["direction"] == "bullish"
    assert result["phase"] >= 2
