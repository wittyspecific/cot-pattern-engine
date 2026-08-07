import numpy as np
import pandas as pd

from pattern_engine.net_levels import analyze_group_net, analyze_overlap, build_reference_zones


def sample_frame(n=240):
    dates = pd.date_range("2010-01-05", periods=n, freq="W-TUE")
    x = np.linspace(-2, 2, n)
    return pd.DataFrame({
        "date": dates,
        "commercial_net": 100000 * np.sin(x),
        "nonreportable_net": -30000 * np.sin(x) + 5000 * np.cos(2*x),
        "commercial_net_oi": .2 * np.sin(x),
        "nonreportable_net_oi": -.08 * np.sin(x),
        "return_8w": .05 * np.sin(x) + .01,
    })


def test_group_research_returns_cases():
    r = analyze_group_net(sample_frame(), "Commercials", horizon_weeks=8, n_neighbors=20)
    assert r.available
    assert 1 <= r.sample_size <= 20


def test_overlap_returns_cases():
    r = analyze_overlap(sample_frame(), horizon_weeks=8, n_neighbors=20)
    assert r.available
    assert r.group.startswith("Commercial")


def test_reference_zones_include_current_zone():
    zones = build_reference_zones(sample_frame(), "Commercials", horizon_weeks=8)
    assert zones
    assert sum(z.is_current for z in zones) == 1


def test_pure_net_pattern_uses_all_three_absolute_net_columns():
    from pattern_engine.net_levels import analyze_pure_net_pattern

    dates = pd.date_range("2010-01-05", periods=80, freq="W-TUE")
    frame = pd.DataFrame({
        "date": dates,
        "commercial_net": np.linspace(-100000, 100000, 80),
        "noncommercial_net": np.linspace(90000, -90000, 80),
        "nonreportable_net": np.linspace(10000, -10000, 80),
        "return_8w": np.where(np.arange(80) % 3, 0.04, -0.02),
    })
    result = analyze_pure_net_pattern(frame, horizon_weeks=8, n_neighbors=12, min_gap_weeks=2)
    assert result.available
    assert result.group == "Reines Nettopositionsmuster"
    assert result.sample_size > 0
    assert {"commercial_net", "noncommercial_net", "nonreportable_net"}.issubset(result.matches.columns)
