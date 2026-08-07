import numpy as np, pandas as pd
from seasonality_engine import calibrate_seasonality
def synthetic_prices():
    dates=pd.bdate_range("2005-01-03","2026-08-06");d=np.arange(len(dates))
    return pd.DataFrame({"date":dates,"close":100*np.exp(.00005*d+.04*np.sin(2*np.pi*d/252))})
def test_five_windows():
    r=calibrate_seasonality(synthetic_prices(),15,as_of=pd.Timestamp("2026-08-06"))
    assert r.available
    assert all(x is not None for x in [r.similarity_3d,r.similarity_10d,r.similarity_2w,r.similarity_4w,r.similarity_8w])
    shifts=[r.phase_3d,r.phase_10d,r.phase_2w,r.phase_4w,r.phase_8w]
    assert r.lead_lag_weeks==int(round(float(np.median(shifts))))
