from __future__ import annotations

import numpy as np
import pandas as pd

from fundamental_engine.analysis import (
    add_score_history,
    analyze_fundamental_regime,
    build_monthly_snapshots,
    transformed_series,
)
from fundamental_engine.config import SERIES


def _monthly_level(start="2010-01-01", periods=180, base=100.0, monthly_growth=0.002):
    dates = pd.date_range(start, periods=periods, freq="MS")
    values = base * np.power(1.0 + monthly_growth, np.arange(periods))
    return pd.DataFrame({"date": dates, "value": values})


def test_monthly_series_is_only_available_after_release_lag():
    spec = SERIES["cpi"]
    transformed = transformed_series(_monthly_level(periods=30), spec)
    row = transformed.iloc[0]
    observation = pd.Timestamp(row["observation_date"])
    available = pd.Timestamp(row["available_date"])
    assert available > observation + pd.offsets.MonthEnd(0)
    assert (available - (observation + pd.offsets.MonthEnd(0))).days == spec.release_lag_days


def test_weight_sign_changes_asset_score_direction():
    frames = {
        "fed_funds": transformed_series(
            pd.DataFrame({
                "date": pd.date_range("2000-01-01", periods=260, freq="MS"),
                "value": np.r_[np.linspace(1.0, 2.0, 250), np.linspace(2.2, 7.0, 10)],
            }),
            SERIES["fed_funds"],
        )
    }
    snapshots = build_monthly_snapshots(frames, SERIES, start="2000-01-31")
    positive = add_score_history(snapshots, {"fed_funds": 1.0}).dropna(subset=["fundamental_score"])
    negative = add_score_history(snapshots, {"fed_funds": -1.0}).dropna(subset=["fundamental_score"])
    assert positive.iloc[-1]["fundamental_score"] > 0
    assert negative.iloc[-1]["fundamental_score"] < 0


def test_fundamental_analysis_returns_independent_analog_cases():
    rng = np.random.default_rng(7)
    keys = ["core_cpi", "gdp", "unemployment", "nfp", "fed_funds", "real_yield_10y"]
    frames = {}
    for key in keys:
        spec = SERIES[key]
        if spec.frequency == "quarterly":
            dates = pd.date_range("1990-01-01", "2025-10-01", freq="QS")
        elif spec.frequency == "monthly":
            dates = pd.date_range("1990-01-01", "2025-12-01", freq="MS")
        else:
            dates = pd.date_range("1990-01-01", "2025-12-31", freq="B")
        if spec.transform == "yoy_pct":
            values = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.003, len(dates))))
        elif spec.transform == "diff":
            values = 100_000 + np.cumsum(rng.normal(150, 50, len(dates)))
        else:
            values = 3 + np.cumsum(rng.normal(0, 0.02, len(dates)))
        frames[key] = transformed_series(pd.DataFrame({"date": dates, "value": values}), spec)

    prices = pd.DataFrame({"date": pd.date_range("1990-01-01", "2026-06-30", freq="B")})
    prices["close"] = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, len(prices))))
    weights = {
        "core_cpi": -0.5,
        "gdp": 0.8,
        "unemployment": -0.6,
        "nfp": 0.4,
        "fed_funds": -0.7,
        "real_yield_10y": -0.8,
    }
    result = analyze_fundamental_regime(frames, SERIES, weights, prices, horizon_weeks=8, n_neighbors=12)
    assert result.available
    assert 1 <= result.sample_size <= 12
    assert {"return_4w", "return_8w", "return_12w"}.issubset(result.analogs.columns)
    dates = pd.to_datetime(result.analogs["date"]).sort_values()
    assert all((dates.iloc[i] - dates.iloc[i - 1]).days >= 84 for i in range(1, len(dates)))


def test_analysis_refuses_too_few_series():
    prices = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100, freq="B"), "close": 100.0})
    result = analyze_fundamental_regime({}, SERIES, {}, prices)
    assert not result.available
    assert "Weniger als drei" in result.reason


def test_fundamental_service_has_no_cot_pipeline_dependency():
    from pathlib import Path
    import fundamental_engine.service as service

    source = Path(service.__file__).read_text(encoding="utf-8")
    assert "data_engine.cftc" not in source
    assert "load_market_data" not in source
    assert "pattern_engine" not in source
