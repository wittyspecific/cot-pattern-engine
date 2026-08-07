import numpy as np
import pandas as pd

from seasonality_engine import annual_seasonality_curve, calibrate_seasonality


def synthetic_prices(years=12):
    dates = pd.bdate_range("2014-01-01", periods=years * 260)
    week = dates.isocalendar().week.to_numpy(dtype=float)
    year = dates.year.to_numpy(dtype=float)
    seasonal = 0.08 * np.sin(2 * np.pi * (week - 1) / 52.0)
    trend = (year - year.min()) * 0.01
    close = 100 * np.exp(trend + seasonal)
    return pd.DataFrame({"date": dates, "close": close})


def test_annual_curve_spans_calendar_year():
    curve = annual_seasonality_curve(synthetic_prices(), window_years=10)
    assert list(curve["Kalenderwoche"]) == list(range(1, 53))
    assert curve["Seasonality"].notna().all()


def test_calibration_returns_current_and_seasonal_chart():
    result = calibrate_seasonality(synthetic_prices(), window_years=10)
    assert result.available
    assert -1.0 <= result.similarity <= 1.0
    assert {"Seasonality", "Kalibrierte Seasonality", "Aktueller Verlauf"}.issubset(result.chart.columns)
