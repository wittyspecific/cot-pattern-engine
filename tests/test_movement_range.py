import pandas as pd

from scanner.market_scanner import _dominant_range


def test_dominant_range_uses_only_positive_winners():
    low, high = _dominant_range(pd.Series([0.01, 0.03, 0.05, 0.09, -0.20]))
    assert low > 0
    assert high > low


def test_dominant_range_uses_only_negative_winners():
    low, high = _dominant_range(pd.Series([-0.01, -0.03, -0.05, -0.09, 0.20]))
    assert low < 0
    assert high < 0
