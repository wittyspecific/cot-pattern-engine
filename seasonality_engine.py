from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd

TRADING_DAYS = {4: 20, 8: 40, 12: 60}

@dataclass(frozen=True)
class SeasonalityResult:
    horizon_weeks: int
    window_years: int | None
    available_years: int
    up: int
    down: int
    hit_rate: float | None
    direction: str
    median_return: float | None
    q25: float | None
    q75: float | None
    observations: pd.DataFrame

@dataclass(frozen=True)
class SeasonalityCalibration:
    available: bool
    years: int = 0
    similarity: float | None = None
    similarity_3d: float | None = None
    similarity_10d: float | None = None
    similarity_2w: float | None = None
    similarity_4w: float | None = None
    similarity_8w: float | None = None
    phase_3d: int | None = None
    phase_10d: int | None = None
    phase_2w: int | None = None
    phase_4w: int | None = None
    phase_8w: int | None = None
    lead_lag_weeks: int | None = None
    calibrated_week: int | None = None
    status: str = "Nicht verfügbar"
    seasonal_progress: float | None = None
    progress_status: str = "Nicht verfügbar"
    stability: float | None = None
    current_two_week_return: float | None = None
    seasonal_two_week_return: float | None = None
    performance_gap: float | None = None
    chart: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    reason: str = ""


def _clean(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices[["date", "close"]].copy().dropna()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def _weekly_anchors(prices: pd.DataFrame) -> pd.DataFrame:
    df = _clean(prices)
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)
    return df.groupby(["iso_year", "iso_week"], as_index=False).first()


def analyze_price_seasonality(prices: pd.DataFrame, horizon_weeks: int, window_years: int | None = None, as_of: pd.Timestamp | None = None) -> SeasonalityResult:
    if horizon_weeks not in TRADING_DAYS:
        raise ValueError("horizon_weeks must be one of 4, 8, 12")
    df = _clean(prices)
    if df.empty:
        return SeasonalityResult(horizon_weeks, window_years, 0, 0, 0, None, "Neutral", None, None, None, pd.DataFrame())
    cutoff = min(pd.Timestamp(as_of) if as_of is not None else df["date"].max(), df["date"].max())
    target_iso = cutoff.isocalendar()
    target_week, target_year = int(target_iso.week), int(target_iso.year)
    anchors = _weekly_anchors(df)
    candidates = anchors[(anchors["iso_week"] == target_week) & (anchors["iso_year"] < target_year)].copy()
    if window_years is not None:
        candidates = candidates[candidates["iso_year"] >= target_year - int(window_years)]
    forward = TRADING_DAYS[horizon_weeks]
    rows = []
    dates = df["date"].to_numpy()
    closes = df["close"].astype(float).to_numpy()
    for _, row in candidates.iterrows():
        idx = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(row["date"]))))
        end_idx = idx + forward
        if idx >= len(df) or end_idx >= len(df):
            continue
        rows.append({"year": int(row["iso_year"]), "return": float(closes[end_idx] / closes[idx] - 1.0)})
    obs = pd.DataFrame(rows)
    if obs.empty:
        return SeasonalityResult(horizon_weeks, window_years, 0, 0, 0, None, "Neutral", None, None, None, obs)
    vals = obs["return"]
    up, down = int((vals > 0).sum()), int((vals < 0).sum())
    total = up + down
    if up > down:
        direction, hit, directional = "Bullish", up / total if total else None, vals[vals > 0]
    elif down > up:
        direction, hit, directional = "Bearish", down / total if total else None, vals[vals < 0]
    else:
        direction, hit, directional = "Neutral", 0.5 if total else None, vals
    return SeasonalityResult(
        horizon_weeks, window_years, len(obs), up, down, hit, direction,
        float(vals.median()),
        float(directional.quantile(.25)) if not directional.empty else None,
        float(directional.quantile(.75)) if not directional.empty else None,
        obs,
    )


def seasonality_matrix(prices: pd.DataFrame, windows=(5, 10, 15, None), horizons=(4, 8, 12), as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    for window in windows:
        for horizon in horizons:
            r = analyze_price_seasonality(prices, horizon, window, as_of)
            rows.append({"Historie": "Gesamt" if window is None else f"{window} Jahre", "Horizont": f"{horizon}W", "Jahre": r.available_years, "Richtung": r.direction, "Trefferquote": r.hit_rate, "Median": r.median_return, "Range unten": r.q25, "Range oben": r.q75})
    return pd.DataFrame(rows)


def _historical_paths(prices: pd.DataFrame, window_years: int | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    df = _clean(prices)
    if df.empty:
        return pd.DataFrame(index=range(1, 53))
    cutoff = min(pd.Timestamp(as_of) if as_of is not None else df["date"].max(), df["date"].max())
    current_year = int(cutoff.isocalendar().year)
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int).clip(upper=52)
    history = df[df["iso_year"] < current_year].copy()
    if window_years is not None:
        history = history[history["iso_year"] >= current_year - int(window_years)]
    weekly = history.groupby(["iso_year", "iso_week"], as_index=False)["close"].last()
    paths = []
    for year, group in weekly.groupby("iso_year"):
        group = group.sort_values("iso_week")
        if len(group) < 35:
            continue
        full = group.set_index("iso_week")["close"].reindex(range(1, 53)).interpolate(limit_direction="both")
        base = float(full.iloc[0])
        if base <= 0 or not np.isfinite(base):
            continue
        paths.append((full / base - 1.0).rename(int(year)))
    return pd.concat(paths, axis=1) if paths else pd.DataFrame(index=range(1, 53))


def annual_seasonality_curve(prices: pd.DataFrame, window_years: int | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Robust full-year seasonality path by ISO week (1..52)."""
    frame = _historical_paths(prices, window_years, as_of)
    if frame.empty:
        return pd.DataFrame(columns=["Kalenderwoche", "Seasonality", "Median", "Q25", "Q75", "Jahre"])
    return pd.DataFrame({
        "Kalenderwoche": range(1, 53),
        "Seasonality": frame.mean(axis=1).to_numpy(),
        "Median": frame.median(axis=1).to_numpy(),
        "Q25": frame.quantile(.25, axis=1).to_numpy(),
        "Q75": frame.quantile(.75, axis=1).to_numpy(),
        "Jahre": frame.shape[1],
    })


def _current_year_curve(prices: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    df = _clean(prices)
    if df.empty:
        return pd.DataFrame(columns=["Kalenderwoche", "Aktueller Verlauf"])
    cutoff = min(pd.Timestamp(as_of) if as_of is not None else df["date"].max(), df["date"].max())
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int).clip(upper=52)
    current_year = int(cutoff.isocalendar().year)
    current = df[(df["iso_year"] == current_year) & (df["date"] <= cutoff)].groupby("iso_week", as_index=False)["close"].last()
    if current.empty:
        return pd.DataFrame(columns=["Kalenderwoche", "Aktueller Verlauf"])
    base = float(current.iloc[0]["close"])
    current["Aktueller Verlauf"] = current["close"] / base - 1.0
    return current[["iso_week", "Aktueller Verlauf"]].rename(columns={"iso_week": "Kalenderwoche"})


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std < 1e-12:
        return arr - np.nanmean(arr)
    return (arr - np.nanmean(arr)) / std


def _dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """DTW similarity in [0,1] for two short return sequences."""
    x, y = _zscore(a), _zscore(b)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return 0.0
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    distance = float(dp[n, m] / (n + m))
    return float(math.exp(-distance))


def _window_similarity(current_levels: pd.Series, seasonal_levels: pd.Series, weeks: int) -> float | None:
    if len(current_levels) < weeks + 1 or len(seasonal_levels) < weeks + 1:
        return None
    a = current_levels.iloc[-(weeks + 1):].diff().dropna().to_numpy()
    b = seasonal_levels.iloc[-(weeks + 1):].diff().dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return None
    return _dtw_similarity(a, b)


def _seasonality_stability(prices: pd.DataFrame, as_of: pd.Timestamp | None = None) -> float | None:
    curves = []
    for window in (5, 10, 15, None):
        curve = annual_seasonality_curve(prices, window, as_of)
        if not curve.empty and curve["Jahre"].iloc[0] >= 4:
            curves.append(curve.set_index("Kalenderwoche")["Median"].diff().dropna())
    if len(curves) < 2:
        return None
    correlations = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            corr = curves[i].corr(curves[j])
            if np.isfinite(corr):
                correlations.append(max(0.0, float(corr)))
    return float(np.mean(correlations)) if correlations else None


def _forecast_from_phase(paths: pd.DataFrame, phase_week: int, horizons=(4, 8, 12)) -> pd.DataFrame:
    rows = []
    if paths.empty:
        return pd.DataFrame()
    for horizon in horizons:
        end_week = phase_week + horizon
        if end_week > 52:
            continue
        returns = paths.loc[end_week] - paths.loc[phase_week]
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            continue
        up, down = int((returns > 0).sum()), int((returns < 0).sum())
        direction = "Bullish" if up > down else "Bearish" if down > up else "Neutral"
        hit = max(up, down) / len(returns) if len(returns) else None
        rows.append({
            "Horizont": f"{horizon}W",
            "Fälle": int(len(returns)),
            "Gestiegen": up,
            "Gefallen": down,
            "Richtung": direction,
            "Trefferquote": hit,
            "Median": float(returns.median()),
            "Q25": float(returns.quantile(.25)),
            "Q75": float(returns.quantile(.75)),
        })
    return pd.DataFrame(rows)


def _daily_returns(prices, days, as_of=None):
    df=_clean(prices)
    cutoff=min(pd.Timestamp(as_of) if as_of is not None else df["date"].max(),df["date"].max())
    df=df[df["date"]<=cutoff]
    if len(df)<days+1: return None
    return df["close"].astype(float).iloc[-days-1:].pct_change().dropna().to_numpy()

def _historical_daily_pattern(prices,target_date,days,shift_weeks=0,window_years=None):
    df=_clean(prices); target=pd.Timestamp(target_date)+pd.Timedelta(weeks=shift_weeks)
    cy=pd.Timestamp(target_date).year
    miny=cy-int(window_years) if window_years is not None else int(df["date"].dt.year.min())
    pats=[]
    for year in range(miny,cy):
        day=min(target.day,pd.Period(f"{year}-{target.month:02d}").days_in_month)
        anchor=pd.Timestamp(year=year,month=target.month,day=day)
        idx=int(np.searchsorted(df["date"].to_numpy(),np.datetime64(anchor),side="right")-1)
        if idx<days or idx>=len(df) or df.iloc[idx]["date"].year!=year: continue
        vals=df["close"].astype(float).iloc[idx-days:idx+1]
        if len(vals)==days+1: pats.append(vals.pct_change().dropna().to_numpy())
    return np.nanmedian(np.vstack(pats),axis=0) if pats else None

def _best_daily(prices,days,max_shift,window_years,as_of):
    df=_clean(prices)
    cutoff=min(pd.Timestamp(as_of) if as_of is not None else df["date"].max(),df["date"].max())
    cur=_daily_returns(df,days,cutoff); best=(None,None)
    if cur is None:return best
    for shift in range(-max_shift,max_shift+1):
        sea=_historical_daily_pattern(df,cutoff,days,shift,window_years)
        if sea is None:continue
        sim=_dtw_similarity(cur,sea)
        if best[1] is None or sim>best[1]:best=(shift,sim)
    return best

def _best_weekly(current_series,median_map,weeks,max_shift):
    best=(None,None)
    for shift in range(-max_shift,max_shift+1):
        cv=[];sv=[]
        for week,value in current_series.items():
            phase=int(week+shift)
            if 1<=phase<=52 and phase in median_map.index:
                cv.append(float(value));sv.append(float(median_map.loc[phase]))
        if len(cv)<weeks+1:continue
        sim=_window_similarity(pd.Series(cv),pd.Series(sv),weeks)
        if sim is not None and (best[1] is None or sim>best[1]):best=(shift,sim)
    return best

def calibrate_seasonality(prices: pd.DataFrame, window_years: int | None = 15, max_shift_weeks: int = 8, as_of: pd.Timestamp | None = None) -> SeasonalityCalibration:
    seasonal = annual_seasonality_curve(prices, window_years, as_of)
    current = _current_year_curve(prices, as_of)
    paths = _historical_paths(prices, window_years, as_of)
    if seasonal.empty or len(current) < 9:
        return SeasonalityCalibration(False, reason="Für die Kalibrierung fehlen ausreichend historische oder aktuelle Wochen.")

    median_map = seasonal.set_index("Kalenderwoche")["Median"]
    latest_week = int(current["Kalenderwoche"].max())
    current_series = current.set_index("Kalenderwoche")["Aktueller Verlauf"]
    # Jedes Fenster wird unabhängig optimiert; keine subjektive Gewichtung.
    phase_3d,best_3d=_best_daily(prices,3,max_shift_weeks,window_years,as_of)
    phase_10d,best_10d=_best_daily(prices,10,max_shift_weeks,window_years,as_of)
    phase_2w,best_2w=_best_weekly(current_series,median_map,2,max_shift_weeks)
    phase_4w,best_4w=_best_weekly(current_series,median_map,4,max_shift_weeks)
    phase_8w,best_8w=_best_weekly(current_series,median_map,8,max_shift_weeks)
    shifts=[x for x in (phase_3d,phase_10d,phase_2w,phase_4w,phase_8w) if x is not None]
    if not shifts:
        return SeasonalityCalibration(False,reason="Keine unabhängige Mehrfenster-Kalibrierung berechenbar.")
    shift=int(round(float(np.median(shifts))))
    sims={2:best_2w,4:best_4w,8:best_8w}
    vals=[x for x in (best_3d,best_10d,best_2w,best_4w,best_8w) if x is not None]
    similarity=float(np.median(vals)) if vals else None
    phase_week = int(np.clip(latest_week + shift, 1, 52))
    cur_2w = float(current_series.iloc[-1] - current_series.iloc[-3])
    phase_now = float(median_map.loc[phase_week])
    phase_prev = float(median_map.loc[max(1, phase_week - 2)])
    seas_2w = phase_now - phase_prev
    gap = cur_2w - seas_2w

    # Progress compares the latest 8-week movement with the movement of the calibrated seasonal phase.
    if len(current_series) >= 9 and phase_week >= 9:
        current_8w = float(current_series.iloc[-1] - current_series.iloc[-9])
        seasonal_8w = float(median_map.loc[phase_week] - median_map.loc[phase_week - 8])
        if abs(seasonal_8w) > 1e-8 and np.sign(current_8w) == np.sign(seasonal_8w):
            progress = abs(current_8w / seasonal_8w)
        elif abs(seasonal_8w) > 1e-8:
            progress = -abs(current_8w / seasonal_8w)
        else:
            progress = None
    else:
        progress = None
    if progress is None:
        progress_status = "Nicht belastbar"
    elif progress < 0:
        progress_status = "Gegenläufig zur Seasonality"
    elif progress < .80:
        progress_status = "Bewegung hinkt hinterher"
    elif progress <= 1.20:
        progress_status = "Bewegung verläuft normal"
    else:
        progress_status = "Bewegung ist vorausgelaufen"

    if shift >= 2:
        status = f"Saisonale Phase etwa {shift} Wochen voraus"
    elif shift <= -2:
        status = f"Saisonale Phase etwa {abs(shift)} Wochen zurück"
    else:
        status = "Nahe an der saisonalen Taktung"

    stability = _seasonality_stability(prices, as_of)
    chart = seasonal[["Kalenderwoche", "Median", "Q25", "Q75"]].copy()
    chart = chart.rename(columns={"Median": "Median-Seasonality", "Q25": "25. Perzentil", "Q75": "75. Perzentil"})
    # Backwards-compatible alias retained for older integrations and tests.
    chart["Seasonality"] = chart["Median-Seasonality"]
    chart["Kalibrierte Seasonality"] = chart["Kalenderwoche"].map(
        lambda w: median_map.get(int(np.clip(w + shift, 1, 52)), np.nan)
    )
    chart = chart.merge(current, on="Kalenderwoche", how="left")
    forecast = _forecast_from_phase(paths, phase_week)
    return SeasonalityCalibration(
        available=True,
        years=int(seasonal["Jahre"].iloc[0]),
        similarity=similarity,
        similarity_3d=best_3d, similarity_10d=best_10d,
        similarity_2w=best_2w, similarity_4w=best_4w, similarity_8w=best_8w,
        phase_3d=phase_3d, phase_10d=phase_10d, phase_2w=phase_2w,
        phase_4w=phase_4w, phase_8w=phase_8w,
        lead_lag_weeks=shift,
        calibrated_week=phase_week,
        status=status,
        seasonal_progress=progress,
        progress_status=progress_status,
        stability=stability,
        current_two_week_return=cur_2w,
        seasonal_two_week_return=seas_2w,
        performance_gap=gap,
        chart=chart,
        forecast=forecast,
    )


def seasonality_curves(prices: pd.DataFrame, windows=(5, 10, 15, None), max_weeks: int = 12, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    out = None
    for window in windows:
        curve = annual_seasonality_curve(prices, window, as_of)[["Kalenderwoche", "Median"]]
        label = "Gesamt" if window is None else f"{window} Jahre"
        curve = curve.rename(columns={"Median": label})
        out = curve if out is None else out.merge(curve, on="Kalenderwoche", how="outer")
    return out.sort_values("Kalenderwoche").reset_index(drop=True) if out is not None else pd.DataFrame()
