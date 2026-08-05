import numpy as np
import pandas as pd

from pattern_engine.divergence import analyze_noncommercial_divergence


def test_net_only_data_is_not_used_for_active_divergence():
    n = 220
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-07", periods=n, freq="W-TUE"),
        "close": np.linspace(100.0, 110.0, n),
        "noncommercial_net": np.linspace(0.0, -20000.0, n),
    })
    result = analyze_noncommercial_divergence(frame)
    assert not result["available"]
    assert result["signal"] == "none"
