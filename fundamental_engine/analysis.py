from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .config import FundamentalSeriesSpec


@dataclass
class FundamentalResearchResult:
    available: bool
    reason: str = ""
    score: float | None = None
    label: str = "Nicht verfügbar"
    confidence: str = "niedrig"
    snapshot_date: pd.Timestamp | None = None
    driver_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    score_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    analogs: pd.DataFrame = field(default_factory=pd.DataFrame)
    sample_size: int = 0
    up_count: int = 0
    down_count: int = 0
    hit_rate: float | None = None
    median_return: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    sources: dict[str, str] = field(default_factory=dict)
    missing_series: list[str] = field(default_factory=list)


def _periods_per_year(frequency: str) -> int:
    return {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}[frequency]


def transformed_series(frame: pd.DataFrame, spec: FundamentalSeriesSpec) -> pd.DataFrame:
    data = frame[["date", "value"]].dropna().sort_values("date").copy()
    values = pd.to_numeric(data["value"], errors="coerce")
    if spec.transform == "level":
        transformed = values
    elif spec.transform == "yoy_pct":
        transformed = values.pct_change(_periods_per_year(spec.frequency), fill_method=None) * 100.0
    elif spec.transform == "diff":
        transformed = values.diff()
    elif spec.transform == "rolling4":
        transformed = values.rolling(4, min_periods=3).mean()
    else:
        raise ValueError(f"Unbekannte Transformation: {spec.transform}")

    if spec.frequency == "monthly":
        available_date = data["date"] + pd.offsets.MonthEnd(0) + pd.to_timedelta(spec.release_lag_days, unit="D")
    elif spec.frequency == "quarterly":
        available_date = data["date"] + pd.offsets.QuarterEnd(0) + pd.to_timedelta(spec.release_lag_days, unit="D")
    else:
        available_date = data["date"] + pd.to_timedelta(spec.release_lag_days, unit="D")

    out = pd.DataFrame({
        "observation_date": data["date"].to_numpy(),
        "available_date": pd.to_datetime(available_date).to_numpy(),
        "raw_value": values.to_numpy(),
        "feature_value": transformed.to_numpy(),
    })
    return out.dropna(subset=["available_date", "feature_value"]).sort_values("available_date")


def _rolling_z(series: pd.Series, window: int = 120, min_periods: int = 36) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def build_monthly_snapshots(
    series_frames: Mapping[str, pd.DataFrame],
    specs: Mapping[str, FundamentalSeriesSpec],
    start: str = "1990-01-31",
) -> pd.DataFrame:
    if not series_frames:
        return pd.DataFrame()
    max_date = max(pd.to_datetime(frame["available_date"]).max() for frame in series_frames.values() if not frame.empty)
    dates = pd.DataFrame({"date": pd.date_range(start=start, end=max_date, freq="ME")})
    snapshots = dates.copy()

    for key, frame in series_frames.items():
        if frame.empty:
            continue
        right = frame[["available_date", "feature_value", "raw_value", "observation_date"]].copy()
        right = right.sort_values("available_date")
        left = dates.rename(columns={"date": "snapshot_date"}).sort_values("snapshot_date")
        merged = pd.merge_asof(
            left,
            right,
            left_on="snapshot_date",
            right_on="available_date",
            direction="backward",
        )
        snapshots[key] = merged["feature_value"].to_numpy()
        snapshots[f"{key}__raw"] = merged["raw_value"].to_numpy()
        snapshots[f"{key}__observation_date"] = pd.to_datetime(merged["observation_date"]).to_numpy()
        snapshots[f"{key}__available_date"] = pd.to_datetime(merged["available_date"]).to_numpy()
        snapshots[f"{key}__z"] = _rolling_z(pd.Series(merged["feature_value"].to_numpy(), index=snapshots.index))

    return snapshots


def _score_label(score: float) -> str:
    if score >= 50:
        return "deutlich bullish"
    if score >= 20:
        return "leicht bullish"
    if score <= -50:
        return "deutlich bearish"
    if score <= -20:
        return "leicht bearish"
    return "neutral / gemischt"


def _regime_label(value: float | None, positive: str, neutral: str, negative: str) -> str:
    if value is None or pd.isna(value):
        return "Nicht verfügbar"
    if value >= 0.50:
        return positive
    if value <= -0.50:
        return negative
    return neutral


def add_score_history(snapshots: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    out = snapshots.copy()
    available_keys = [key for key in weights if f"{key}__z" in out]
    if not available_keys:
        out["fundamental_score"] = np.nan
        return out

    numerator = pd.Series(0.0, index=out.index)
    denominator = pd.Series(0.0, index=out.index)
    for key in available_keys:
        z = pd.to_numeric(out[f"{key}__z"], errors="coerce")
        weight = float(weights[key])
        numerator = numerator.add(z.fillna(0.0) * weight, fill_value=0.0)
        denominator = denominator.add(z.notna().astype(float) * abs(weight), fill_value=0.0)
    raw = numerator / denominator.replace(0, np.nan)
    out["fundamental_score"] = np.tanh(raw / 1.25) * 100.0
    out["available_weight_share"] = denominator / max(sum(abs(float(v)) for v in weights.values()), 1e-9)
    return out


def _forward_returns(prices: pd.DataFrame, event_dates: pd.Series, horizons=(4, 8, 12)) -> pd.DataFrame:
    px = prices[["date", "close"]].dropna().copy().sort_values("date")
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    events = pd.DataFrame({"date": pd.to_datetime(event_dates).dt.tz_localize(None)}).sort_values("date")
    start = pd.merge_asof(events, px.rename(columns={"close": "start_close"}), on="date", direction="forward", tolerance=pd.Timedelta(days=10))
    out = start.copy()
    for weeks in horizons:
        targets = pd.DataFrame({
            "date": start["date"],
            "target_date": start["date"] + pd.Timedelta(weeks=weeks),
        }).sort_values("target_date")
        end = pd.merge_asof(
            targets,
            px.rename(columns={"date": "price_date", "close": "end_close"}),
            left_on="target_date",
            right_on="price_date",
            direction="forward",
            tolerance=pd.Timedelta(days=10),
        ).sort_values("date")
        out[f"return_{weeks}w"] = end["end_close"].to_numpy() / out["start_close"].to_numpy() - 1.0
    return out


def _select_analogs(
    scored: pd.DataFrame,
    weights: Mapping[str, float],
    prices: pd.DataFrame,
    horizon_weeks: int,
    n_neighbors: int,
    min_gap_months: int = 3,
) -> pd.DataFrame:
    keys = [key for key in weights if f"{key}__z" in scored]
    if len(keys) < 3:
        return pd.DataFrame()
    zcols = [f"{key}__z" for key in keys]
    clean = scored.dropna(subset=zcols + ["fundamental_score"]).copy().sort_values("date")
    if len(clean) < 40:
        return pd.DataFrame()
    current = clean.iloc[-1]
    history = clean.iloc[:-1].copy()
    abs_weights = np.array([abs(float(weights[key])) for key in keys], dtype=float)
    abs_weights = abs_weights / abs_weights.sum()
    diffs = history[zcols].to_numpy(dtype=float) - current[zcols].to_numpy(dtype=float)
    history["distance"] = np.sqrt(np.sum(np.square(diffs) * abs_weights, axis=1))
    history = history.sort_values("distance")

    chosen: list[int] = []
    chosen_dates: list[pd.Timestamp] = []
    gap = pd.DateOffset(months=min_gap_months)
    for idx, row in history.iterrows():
        event_date = pd.Timestamp(row["date"])
        if any(abs((event_date - used).days) < gap.kwds.get("months", min_gap_months) * 28 for used in chosen_dates):
            continue
        chosen.append(idx)
        chosen_dates.append(event_date)
        if len(chosen) >= n_neighbors * 2:
            break
    candidates = history.loc[chosen].copy()
    returns = _forward_returns(prices, candidates["date"], horizons=(4, 8, 12))
    candidates = candidates.merge(returns, on="date", how="left")
    candidates = candidates.dropna(subset=[f"return_{horizon_weeks}w"]).head(n_neighbors)
    if candidates.empty:
        return candidates
    scale = max(float(candidates["distance"].median()), 1e-9)
    candidates["similarity"] = 100.0 * np.exp(-candidates["distance"] / scale)
    return candidates.sort_values("distance")


def _dominant_range(values: pd.Series) -> tuple[float | None, float | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    positive = values[values > 0]
    negative = values[values < 0]
    selected = positive if len(positive) > len(negative) else negative if len(negative) > len(positive) else pd.Series(dtype=float)
    if selected.empty:
        return None, None
    return float(selected.quantile(0.25)), float(selected.quantile(0.75))


def _current_driver_table(
    current: pd.Series,
    weights: Mapping[str, float],
    specs: Mapping[str, FundamentalSeriesSpec],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, weight in weights.items():
        zcol = f"{key}__z"
        if zcol not in current.index or pd.isna(current.get(zcol)):
            continue
        z = float(current[zcol])
        contribution = z * float(weight)
        effect = "Unterstützend" if contribution > 0.10 else "Belastend" if contribution < -0.10 else "Neutral"
        spec = specs[key]
        rows.append({
            "Bereich": spec.group,
            "Datenreihe": spec.label,
            "Aktueller Messwert": float(current[key]),
            "Einheit": spec.display_unit,
            "Z-Score": z,
            "Gewicht": float(weight),
            "Beitrag": contribution,
            "Asset-Wirkung": effect,
            "Beobachtungsperiode": pd.to_datetime(current.get(f"{key}__observation_date")),
            "Als verfügbar behandelt ab": pd.to_datetime(current.get(f"{key}__available_date")),
            "FRED-ID": spec.fred_id,
        })
    return pd.DataFrame(rows).sort_values("Beitrag", key=lambda s: s.abs(), ascending=False) if rows else pd.DataFrame()


def _regime_table(current: pd.Series) -> pd.DataFrame:
    def avg(items: list[tuple[str, float]]) -> float | None:
        vals = []
        for key, sign in items:
            value = current.get(f"{key}__z")
            if value is not None and not pd.isna(value):
                vals.append(float(value) * sign)
        return float(np.mean(vals)) if vals else None

    inflation = avg([("cpi", 1), ("core_cpi", 1), ("core_pce", 1), ("breakeven_10y", 1)])
    growth = avg([("gdp", 1), ("industrial_production", 1), ("retail_sales", 1)])
    labor = avg([("nfp", 1), ("unemployment", -1), ("initial_claims", -1)])
    tightness = avg([("fed_funds", 1), ("policy_rate", 1), ("yield_2y", 1), ("yield_10y", 0.5), ("real_yield_10y", 1)])
    liquidity = avg([("m2", 1), ("financial_stress", -1)])
    rows = [
        {"Dimension": "Inflation", "Z-Score": inflation, "Regime": _regime_label(inflation, "überdurchschnittlicher Inflationsdruck", "normaler Inflationsbereich", "unterdurchschnittlicher Inflationsdruck")},
        {"Dimension": "Wachstum", "Z-Score": growth, "Regime": _regime_label(growth, "starkes Wachstum", "gemischtes Wachstum", "schwaches Wachstum")},
        {"Dimension": "Arbeitsmarkt", "Z-Score": labor, "Regime": _regime_label(labor, "starker Arbeitsmarkt", "ausgeglichener Arbeitsmarkt", "schwacher Arbeitsmarkt")},
        {"Dimension": "Geldpolitik", "Z-Score": tightness, "Regime": _regime_label(tightness, "restriktiv", "neutral", "locker")},
        {"Dimension": "Liquidität / Stress", "Z-Score": liquidity, "Regime": _regime_label(liquidity, "liquiditätsfreundlich", "gemischt", "liquiditätsarm / angespannt")},
    ]
    return pd.DataFrame(rows)


def analyze_fundamental_regime(
    transformed_frames: Mapping[str, pd.DataFrame],
    specs: Mapping[str, FundamentalSeriesSpec],
    weights: Mapping[str, float],
    prices: pd.DataFrame,
    horizon_weeks: int = 8,
    n_neighbors: int = 20,
    sources: Mapping[str, str] | None = None,
    missing_series: list[str] | None = None,
) -> FundamentalResearchResult:
    if len(transformed_frames) < 3:
        return FundamentalResearchResult(False, "Weniger als drei Fundamentaldatenreihen sind verfügbar.", sources=dict(sources or {}), missing_series=list(missing_series or []))

    snapshots = build_monthly_snapshots(transformed_frames, specs)
    scored = add_score_history(snapshots, weights)
    usable = scored.dropna(subset=["fundamental_score"]).copy()
    usable = usable.loc[usable["available_weight_share"] >= 0.55]
    if usable.empty:
        return FundamentalResearchResult(False, "Noch keine ausreichend vollständige Fundamentalhistorie verfügbar.", sources=dict(sources or {}), missing_series=list(missing_series or []))

    current = usable.iloc[-1]
    driver_table = _current_driver_table(current, weights, specs)
    if driver_table.empty:
        return FundamentalResearchResult(False, "Aktuell konnten keine standardisierten Treiber berechnet werden.", sources=dict(sources or {}), missing_series=list(missing_series or []))

    score = float(current["fundamental_score"])
    analogs = _select_analogs(usable, weights, prices, horizon_weeks, n_neighbors)
    return_col = f"return_{horizon_weeks}w"
    returns = pd.to_numeric(analogs.get(return_col), errors="coerce").dropna() if not analogs.empty else pd.Series(dtype=float)
    up = int((returns > 0).sum())
    down = int((returns < 0).sum())
    sample = len(returns)
    hit_rate = max(up, down) / sample if sample else None
    low, high = _dominant_range(returns)

    coverage = float(current.get("available_weight_share", 0.0))
    confidence = "hoch" if coverage >= 0.85 and sample >= 15 else "mittel" if coverage >= 0.65 and sample >= 8 else "niedrig"
    history_cols = ["date", "fundamental_score"] + [f"{key}__z" for key in weights if f"{key}__z" in usable]
    return FundamentalResearchResult(
        available=True,
        score=score,
        label=_score_label(score),
        confidence=confidence,
        snapshot_date=pd.Timestamp(current["date"]),
        driver_table=driver_table,
        regime_table=_regime_table(current),
        score_history=usable[history_cols].copy(),
        analogs=analogs,
        sample_size=sample,
        up_count=up,
        down_count=down,
        hit_rate=hit_rate,
        median_return=float(returns.median()) if sample else None,
        range_low=low,
        range_high=high,
        sources=dict(sources or {}),
        missing_series=list(missing_series or []),
    )
