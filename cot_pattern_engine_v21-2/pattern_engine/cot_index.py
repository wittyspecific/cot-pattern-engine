from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


INDEX_COLS = ["commercial_cot_index", "retail_cot_index"]


@dataclass(frozen=True)
class CotIndexResearch:
    available: bool
    commercial_index: float | None
    retail_index: float | None
    commercial_zone: str
    retail_zone: str
    sample_size: int
    bias: str
    hit_rate: float | None
    confidence_low: float | None
    confidence_high: float | None
    median_return: float | None
    q25_return: float | None
    q75_return: float | None
    best_return: float | None
    worst_return: float | None
    quality_score: float | None
    matches: pd.DataFrame
    reason: str


def _rolling_cot_index(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    rolling_min = values.rolling(window, min_periods=min_periods).min()
    rolling_max = values.rolling(window, min_periods=min_periods).max()
    spread = (rolling_max - rolling_min).replace(0.0, np.nan)
    return 100.0 * (values - rolling_min) / spread


def add_cot_indices(
    frame: pd.DataFrame,
    lookback_weeks: int = 156,
    min_periods: int = 52,
) -> pd.DataFrame:
    """Add conventional 0–100 COT indices for Commercials and Retail.

    Index = 100 * (current net - rolling minimum) / (rolling maximum - rolling minimum).
    The rolling window uses only information available at the corresponding date.
    """
    result = frame.copy().sort_values("date")
    result["commercial_cot_index"] = _rolling_cot_index(
        result["commercial_net"], lookback_weeks, min_periods
    )
    result["retail_cot_index"] = _rolling_cot_index(
        result["nonreportable_net"], lookback_weeks, min_periods
    )
    return result


def _zone(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "Nicht verfügbar"
    if value >= 80.0:
        return "Extrem hoch (≥ 80)"
    if value <= 20.0:
        return "Extrem niedrig (≤ 20)"
    return "Neutral (20–80)"


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _independent_matches(frame: pd.DataFrame, min_gap_weeks: int, limit: int) -> pd.DataFrame:
    selected: list[int] = []
    dates: list[pd.Timestamp] = []
    gap = pd.Timedelta(weeks=min_gap_weeks)
    for idx, row in frame.sort_values(["index_distance", "date"]).iterrows():
        dt = pd.Timestamp(row["date"])
        if any(abs(dt - used) < gap for used in dates):
            continue
        selected.append(idx)
        dates.append(dt)
        if len(selected) >= limit:
            break
    return frame.loc[selected].sort_values("index_distance")


def analyze_cot_index_pattern(
    data: pd.DataFrame,
    horizon_weeks: int = 8,
    n_neighbors: int = 30,
    lookback_weeks: int = 156,
    min_history: int = 104,
    min_sample: int = 12,
    min_gap_weeks: int = 8,
) -> CotIndexResearch:
    """Research similar Commercial/Retail COT-index configurations.

    Similarity is determined in the two-dimensional index space after standardizing
    historical observations. The current observation is excluded from outcome
    statistics. Nearby reports from the same episode are de-duplicated.
    """
    return_col = f"return_{int(horizon_weeks)}w"
    indexed = add_cot_indices(data, lookback_weeks=lookback_weeks)
    required = ["date", *INDEX_COLS, return_col]
    clean = indexed.dropna(subset=required).sort_values("date").copy()
    if len(clean) < min_history + 1:
        return CotIndexResearch(
            False, None, None, "Nicht verfügbar", "Nicht verfügbar", 0, "neutral",
            None, None, None, None, None, None, None, None, None,
            clean.iloc[0:0], "Zu wenige vollständige COT-Index-Beobachtungen."
        )

    current = clean.iloc[[-1]].copy()
    history = clean.iloc[:-1].copy()
    current_commercial = float(current.iloc[0]["commercial_cot_index"])
    current_retail = float(current.iloc[0]["retail_cot_index"])

    scaler = StandardScaler().fit(history[INDEX_COLS])
    x_hist = scaler.transform(history[INDEX_COLS])
    x_now = scaler.transform(current[INDEX_COLS])
    n_raw = min(max(n_neighbors * 4, n_neighbors), len(history))
    model = NearestNeighbors(n_neighbors=n_raw).fit(x_hist)
    distances, indices = model.kneighbors(x_now)
    ranked = history.iloc[indices[0]].copy()
    ranked["index_distance"] = distances[0]
    matches = _independent_matches(ranked, min_gap_weeks=min_gap_weeks, limit=n_neighbors)
    returns = matches[return_col].dropna().astype(float)
    n = int(len(returns))

    if n < min_sample:
        return CotIndexResearch(
            False, current_commercial, current_retail, _zone(current_commercial),
            _zone(current_retail), n, "neutral", None, None, None, None, None,
            None, None, None, None, matches,
            f"Nur {n} unabhängige Indexmuster gefunden; mindestens {min_sample} werden verlangt."
        )

    positive = int((returns > 0).sum())
    negative = int((returns < 0).sum())
    positive_rate = positive / n
    negative_rate = negative / n
    if positive_rate >= 0.60:
        bias, successes, hit_rate = "bullish", positive, positive_rate
    elif negative_rate >= 0.60:
        bias, successes, hit_rate = "bearish", negative, negative_rate
    else:
        bias = "neutral"
        successes = max(positive, negative)
        hit_rate = max(positive_rate, negative_rate)

    low, high = _wilson_interval(successes, n)
    sample_factor = min(1.0, np.log1p(n) / np.log1p(50))
    distance_median = float(matches["index_distance"].median())
    similarity_factor = float(np.clip(1.0 - distance_median / 2.5, 0.0, 1.0))
    directional_factor = 0.0 if bias == "neutral" else 1.0
    quality = 100.0 * directional_factor * (
        0.65 * low + 0.20 * sample_factor + 0.15 * similarity_factor
    )

    return CotIndexResearch(
        True,
        current_commercial,
        current_retail,
        _zone(current_commercial),
        _zone(current_retail),
        n,
        bias,
        hit_rate,
        low,
        high,
        float(returns.median()),
        float(returns.quantile(0.25)),
        float(returns.quantile(0.75)),
        float(returns.max()),
        float(returns.min()),
        float(quality),
        matches,
        "",
    )
