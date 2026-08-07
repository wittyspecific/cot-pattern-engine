import numpy as np
import pandas as pd

from pattern_engine.cot_index import add_cot_indices, analyze_cot_index_pattern


def _sample(n=360):
    dates = pd.date_range("2010-01-05", periods=n, freq="W-TUE")
    cycle = np.sin(np.arange(n) / 9.0)
    commercial = 100_000 * cycle
    retail = 60_000 * cycle + 5_000 * np.cos(np.arange(n) / 4.0)
    returns = np.where(cycle > 0, 0.04, -0.04)
    return pd.DataFrame({
        "date": dates,
        "commercial_net": commercial,
        "noncommercial_net": -commercial - retail,
        "nonreportable_net": retail,
        "oi": 600_000.0,
        "commercial_net_oi": commercial / 600_000.0,
        "noncommercial_net_oi": (-commercial-retail) / 600_000.0,
        "nonreportable_net_oi": retail / 600_000.0,
        "return_8w": returns,
    })


def test_indices_are_bounded():
    result = add_cot_indices(_sample())
    values = result[["commercial_cot_index", "retail_cot_index"]].dropna()
    assert ((values >= 0) & (values <= 100)).all().all()


def test_index_pattern_requires_joint_validation_or_returns_matches():
    frame = _sample()
    # The module must always expose the current values. It only produces outcome
    # statistics when index and absolute-net percentiles confirm the same extremes.
    result = analyze_cot_index_pattern(frame, n_neighbors=20, min_sample=1)
    assert result.commercial_index is not None
    assert result.retail_index is not None
    assert result.commercial_net_percentile is not None
    assert result.retail_net_percentile is not None
    if result.available:
        assert result.commercial_validated
        assert result.retail_validated
        assert result.sample_size >= 1
        assert not result.matches.empty
    else:
        assert "bestätigt" in result.reason or "Extremkonstellation" in result.reason
