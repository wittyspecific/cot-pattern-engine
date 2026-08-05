import numpy as np
import pandas as pd

from pattern_engine.clusters import analyze_current_cluster


def test_cluster_research_returns_distribution():
    rng = np.random.default_rng(7)
    n = 240
    state = np.repeat([0, 1, 2, 3], n // 4)
    dates = pd.date_range("2000-01-04", periods=n, freq="W-TUE")
    base = state * 3.0 + rng.normal(0, 0.25, n)
    frame = pd.DataFrame({
        "date": dates,
        "commercial_net": base * 10000,
        "noncommercial_net": -base * 9000,
        "nonreportable_net": -base * 1000,
        "commercial_net_oi": base / 20,
        "noncommercial_net_oi": -base / 22,
        "nonreportable_net_oi": -base / 200,
        "return_8w": np.where(state == 3, 0.05 + rng.normal(0, 0.01, n), rng.normal(0, 0.02, n)),
    })
    result = analyze_current_cluster(frame, horizon_weeks=8, min_gap_weeks=1, min_sample=8)
    assert result.available
    assert result.bias == "bullish"
    assert result.sample_size >= 8
    assert result.confidence_low is not None
    assert result.median_return > 0

from pattern_engine.clusters import analyze_cluster_timing


def test_cluster_timing_detects_median_onset():
    dates = pd.date_range("2010-01-05", periods=12, freq="12W-TUE")
    matches = pd.DataFrame({"date": dates})
    price_dates = pd.date_range("2010-01-05", "2013-06-30", freq="B")
    prices = pd.DataFrame({"date": price_dates, "close": 100.0})
    for event in dates:
        mask = prices["date"] >= event
        idx = prices.index[mask][:41]
        if len(idx) == 41:
            path = np.r_[np.zeros(5), np.linspace(0.0, 0.06, 36)]
            prices.loc[idx, "close"] = 100.0 * (1.0 + path)
    matches.attrs["current_report_date"] = dates[-1]
    result = analyze_cluster_timing(matches, prices, "bullish", observation_weeks=8)
    assert result.available
    assert result.onset_day is not None
    assert result.peak_day >= result.onset_day
