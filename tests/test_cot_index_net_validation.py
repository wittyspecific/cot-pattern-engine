import numpy as np
import pandas as pd

from pattern_engine.cot_index import add_cot_indices, analyze_cot_index_pattern


def make_frame(n=260):
    dates = pd.date_range("2010-01-05", periods=n, freq="W-TUE")
    # Repeating long cycles create many historical high/high and low/low extremes.
    x = np.sin(np.arange(n) * 2 * np.pi / 52)
    commercial = 100_000 * x
    retail = 40_000 * x
    returns = np.where(x > 0.85, 0.05, np.where(x < -0.85, -0.05, 0.005 * x))
    return pd.DataFrame({
        "date": dates,
        "commercial_net": commercial,
        "nonreportable_net": retail,
        "return_8w": returns,
    })


def test_adds_net_percentiles():
    out = add_cot_indices(make_frame(), lookback_weeks=104, min_periods=52)
    assert "commercial_net_percentile" in out
    assert "retail_net_percentile" in out
    assert out["commercial_net_percentile"].dropna().between(0, 100).all()


def test_requires_double_extreme_confirmation():
    frame = make_frame()
    # Force current report to have a high rolling index but a non-extreme absolute percentile
    # by choosing a value only slightly above the recent minimum in a compressed final window.
    frame.loc[frame.index[-1], "commercial_net"] = 0.0
    result = analyze_cot_index_pattern(frame, min_history=80, min_sample=1)
    assert not result.available
    assert "doppelt bestätigte" in result.reason


def test_joint_pattern_returns_historical_matches():
    frame = make_frame()
    # Place current observation at a clear joint upper extreme for both groups.
    frame.loc[frame.index[-1], "commercial_net"] = 120_000
    frame.loc[frame.index[-1], "nonreportable_net"] = 50_000
    frame.loc[frame.index[-1], "return_8w"] = 0.05
    result = analyze_cot_index_pattern(
        frame, min_history=80, min_sample=1, n_neighbors=20, min_gap_weeks=4
    )
    assert result.commercial_validated
    assert result.retail_validated
    assert result.sample_size >= 1
    assert {"commercial_net_percentile", "retail_net_percentile"}.issubset(result.matches.columns)
