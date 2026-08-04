import pandas as pd
from data_engine.quality import assess_freshness

def test_stale_data_is_rejected():
    frame = pd.DataFrame({"date": ["2022-01-01"]})
    result = assess_freshness(frame, "date", 14, today=pd.Timestamp("2026-08-04"))
    assert not result.is_current
    assert result.status == "stale"

def test_current_data_is_accepted():
    frame = pd.DataFrame({"date": ["2026-07-28"]})
    result = assess_freshness(frame, "date", 14, today=pd.Timestamp("2026-08-04"))
    assert result.is_current
