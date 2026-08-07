import pandas as pd

from pattern_engine.divergence import analyze_noncommercial_divergence


def test_falling_price_with_both_long_and_short_building_is_mixed_accumulation():
    n = 80
    dates = pd.date_range('2024-01-02', periods=n, freq='W-TUE')
    close = pd.Series([100.0] * (n - 4) + [95.0, 90.0, 85.0, 80.0])
    longs = pd.Series(range(1000, 1000 + n), dtype=float)
    shorts = pd.Series(range(2000, 2000 + n), dtype=float)
    longs.iloc[-4:] += [100, 200, 400, 800]
    shorts.iloc[-4:] += [200, 400, 800, 1600]
    data = pd.DataFrame({
        'date': dates,
        'close': close,
        'noncomm_long': longs,
        'noncomm_short': shorts,
    })
    result = analyze_noncommercial_divergence(data, horizons=(4,), min_price_move=0.02, min_flow_z=-10, min_dominance=0.70)
    item = result['horizons'][0]
    assert item['state'] == 'bullish_accumulation_mixed'
    assert item['signal'] == 'none'
    assert 'gemischte Akkumulationsphase' in item['explanation']
