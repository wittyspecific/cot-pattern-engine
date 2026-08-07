import pandas as pd
from seasonality_engine import analyze_price_seasonality, seasonality_matrix

def synthetic_prices():
    dates = pd.bdate_range('2010-01-01','2025-12-31')
    # deterministic upward series makes every forward return positive
    return pd.DataFrame({'date': dates, 'close': range(100,100+len(dates))})

def test_price_seasonality_isolated_direction():
    r = analyze_price_seasonality(synthetic_prices(), 4, window_years=10, as_of=pd.Timestamp('2025-08-06'))
    assert r.available_years >= 9
    assert r.direction == 'Bullish'
    assert r.hit_rate == 1.0

def test_matrix_has_requested_windows_and_horizons():
    m = seasonality_matrix(synthetic_prices(), as_of=pd.Timestamp('2025-08-06'))
    assert len(m) == 12
    assert set(m['Horizont']) == {'4W','8W','12W'}
    assert set(m['Historie']) == {'5 Jahre','10 Jahre','15 Jahre','Gesamt'}
