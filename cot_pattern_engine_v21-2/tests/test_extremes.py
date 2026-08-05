import pandas as pd

from pattern_engine.extremes import analyze_position_extremes


def test_bullish_joint_extreme_is_detected():
    rows = []
    for i in range(120):
        rows.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(weeks=i),
            "commercial_net": i,
            "noncommercial_net": -i,
            "nonreportable_net": i / 10,
            "commercial_net_oi": i / 1000,
            "noncommercial_net_oi": -i / 1000,
            "nonreportable_net_oi": i / 10000,
        })
    rows.append({
        "date": pd.Timestamp("2023-01-01"),
        "commercial_net": 1000,
        "noncommercial_net": -1000,
        "nonreportable_net": -100,
        "commercial_net_oi": 1.0,
        "noncommercial_net_oi": -1.0,
        "nonreportable_net_oi": -0.1,
    })
    result = analyze_position_extremes(pd.DataFrame(rows), extreme_cutoff=90.0, min_history=100)
    assert result["available"] is True
    assert result["signal"] == "bullish_reversal_zone"
    assert result["confirmed_by_oi"] is True
