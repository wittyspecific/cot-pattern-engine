from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


INDEX_COLS = ["commercial_cot_index", "retail_cot_index"]
NET_PERCENTILE_COLS = ["commercial_net_percentile", "retail_net_percentile"]
PATTERN_COLS = [*INDEX_COLS, *NET_PERCENTILE_COLS]


@dataclass(frozen=True)
class CotIndexResearch:
    available: bool
    commercial_index: float | None
    retail_index: float | None
    commercial_net_percentile: float | None
    retail_net_percentile: float | None
    commercial_zone: str
    retail_zone: str
    commercial_validated: bool
    retail_validated: bool
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



def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling percentile rank using only observations available at each report date."""
    values = pd.to_numeric(series, errors="coerce")

    def percentile_rank(window_values: pd.Series) -> float:
        clean = window_values.dropna()
        if clean.empty:
            return np.nan
        current = clean.iloc[-1]
        # Average rank handles ties without making a repeated maximum automatically 100.
        return float(clean.rank(method="average", pct=True).iloc[-1] * 100.0)

    return values.rolling(window, min_periods=min_periods).apply(percentile_rank, raw=False)



def add_cot_indices(
    frame: pd.DataFrame,
    lookback_weeks: int = 156,
    min_periods: int = 52,
) -> pd.DataFrame:
    """Add COT indices and rolling net-position percentiles for Commercials/Retail.

    Both measures are point-in-time: the rolling calculation uses no observations
    after the corresponding COT report date.
    """
    result = frame.copy().sort_values("date")
    result["commercial_cot_index"] = _rolling_cot_index(
        result["commercial_net"], lookback_weeks, min_periods
    )
    result["retail_cot_index"] = _rolling_cot_index(
        result["nonreportable_net"], lookback_weeks, min_periods
    )
    result["commercial_net_percentile"] = _rolling_percentile(
        result["commercial_net"], lookback_weeks, min_periods
    )
    result["retail_net_percentile"] = _rolling_percentile(
        result["nonreportable_net"], lookback_weeks, min_periods
    )
    return result



def _zone(value: float | None, cutoff: float = 80.0) -> str:
    if value is None or not np.isfinite(value):
        return "Nicht verfügbar"
    lower = 100.0 - cutoff
    if value >= cutoff:
        return f"Extrem hoch (≥ {cutoff:.0f})"
    if value <= lower:
        return f"Extrem niedrig (≤ {lower:.0f})"
    return f"Neutral ({lower:.0f}–{cutoff:.0f})"



def _extreme_side(value: float | None, cutoff: float = 80.0) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    if value >= cutoff:
        return "high"
    if value <= 100.0 - cutoff:
        return "low"
    return None



def _same_extreme(index_value: float | None, net_percentile: float | None, cutoff: float) -> bool:
    side = _extreme_side(index_value, cutoff)
    return side is not None and side == _extreme_side(net_percentile, cutoff)



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
    for idx, row in frame.sort_values(["pattern_distance", "date"]).iterrows():
        dt = pd.Timestamp(row["date"])
        if any(abs(dt - used) < gap for used in dates):
            continue
        selected.append(idx)
        dates.append(dt)
        if len(selected) >= limit:
            break
    return frame.loc[selected].sort_values("pattern_distance")



def _empty_result(
    *,
    current_commercial: float | None = None,
    current_retail: float | None = None,
    commercial_net_percentile: float | None = None,
    retail_net_percentile: float | None = None,
    cutoff: float = 80.0,
    commercial_validated: bool = False,
    retail_validated: bool = False,
    sample_size: int = 0,
    matches: pd.DataFrame | None = None,
    reason: str,
) -> CotIndexResearch:
    return CotIndexResearch(
        available=False,
        commercial_index=current_commercial,
        retail_index=current_retail,
        commercial_net_percentile=commercial_net_percentile,
        retail_net_percentile=retail_net_percentile,
        commercial_zone=_zone(current_commercial, cutoff),
        retail_zone=_zone(current_retail, cutoff),
        commercial_validated=commercial_validated,
        retail_validated=retail_validated,
        sample_size=sample_size,
        bias="neutral",
        hit_rate=None,
        confidence_low=None,
        confidence_high=None,
        median_return=None,
        q25_return=None,
        q75_return=None,
        best_return=None,
        worst_return=None,
        quality_score=None,
        matches=pd.DataFrame() if matches is None else matches,
        reason=reason,
    )



def analyze_cot_index_pattern(
    data: pd.DataFrame,
    horizon_weeks: int = 8,
    n_neighbors: int = 30,
    lookback_weeks: int = 156,
    min_history: int = 104,
    min_sample: int = 1,
    min_gap_weeks: int = 8,
    extreme_cutoff: float = 80.0,
) -> CotIndexResearch:
    """Find historical COT-index patterns validated by absolute net extremes.

    A pattern is eligible only when, for both Commercials and Retail:
    1) the current COT index is in an extreme zone (>=80 or <=20 by default), and
    2) the rolling absolute-net percentile is in the same extreme zone.

    Historical candidates must reproduce the same four-part configuration. Within
    that regime, nearest neighbours are selected in standardized four-dimensional
    space: Commercial index, Retail index, Commercial net percentile, Retail net
    percentile. This directly tests how often the jointly validated pattern led to
    an up or down move.
    """
    return_col = f"return_{int(horizon_weeks)}w"
    indexed = add_cot_indices(data, lookback_weeks=lookback_weeks)
    required = ["date", *PATTERN_COLS, return_col]
    clean = indexed.dropna(subset=required).sort_values("date").copy()
    if len(clean) < min_history + 1:
        return _empty_result(
            matches=clean.iloc[0:0],
            cutoff=extreme_cutoff,
            reason="Zu wenige vollständige COT-Index- und Netto-Perzentil-Beobachtungen.",
        )

    current = clean.iloc[[-1]].copy()
    history = clean.iloc[:-1].copy()
    row = current.iloc[0]
    current_commercial = float(row["commercial_cot_index"])
    current_retail = float(row["retail_cot_index"])
    commercial_net_pct = float(row["commercial_net_percentile"])
    retail_net_pct = float(row["retail_net_percentile"])
    commercial_validated = _same_extreme(current_commercial, commercial_net_pct, extreme_cutoff)
    retail_validated = _same_extreme(current_retail, retail_net_pct, extreme_cutoff)

    if not commercial_validated or not retail_validated:
        missing: list[str] = []
        if not commercial_validated:
            missing.append("Commercials")
        if not retail_validated:
            missing.append("Retail")
        return _empty_result(
            current_commercial=current_commercial,
            current_retail=current_retail,
            commercial_net_percentile=commercial_net_pct,
            retail_net_percentile=retail_net_pct,
            cutoff=extreme_cutoff,
            commercial_validated=commercial_validated,
            retail_validated=retail_validated,
            matches=history.iloc[0:0],
            reason=(
                "Keine doppelt bestätigte Extremkonstellation: Bei "
                + " und ".join(missing)
                + " liegen COT-Index und absolute Netto-Position nicht im gleichen Extrembereich."
            ),
        )

    commercial_side = _extreme_side(current_commercial, extreme_cutoff)
    retail_side = _extreme_side(current_retail, extreme_cutoff)
    lower = 100.0 - extreme_cutoff

    def side_mask(series: pd.Series, side: str) -> pd.Series:
        return series >= extreme_cutoff if side == "high" else series <= lower

    candidates = history.loc[
        side_mask(history["commercial_cot_index"], commercial_side)
        & side_mask(history["commercial_net_percentile"], commercial_side)
        & side_mask(history["retail_cot_index"], retail_side)
        & side_mask(history["retail_net_percentile"], retail_side)
    ].copy()

    if candidates.empty:
        return _empty_result(
            current_commercial=current_commercial,
            current_retail=current_retail,
            commercial_net_percentile=commercial_net_pct,
            retail_net_percentile=retail_net_pct,
            cutoff=extreme_cutoff,
            commercial_validated=True,
            retail_validated=True,
            matches=candidates,
            reason="Die aktuelle doppelt bestätigte Extremkonstellation kam historisch noch nicht erneut vor.",
        )

    scaler = StandardScaler().fit(history[PATTERN_COLS])
    x_candidates = scaler.transform(candidates[PATTERN_COLS])
    x_now = scaler.transform(current[PATTERN_COLS])
    n_raw = min(max(n_neighbors * 4, n_neighbors), len(candidates))
    model = NearestNeighbors(n_neighbors=n_raw).fit(x_candidates)
    distances, indices = model.kneighbors(x_now)
    ranked = candidates.iloc[indices[0]].copy()
    ranked["pattern_distance"] = distances[0]
    matches = _independent_matches(ranked, min_gap_weeks=min_gap_weeks, limit=n_neighbors)
    returns = matches[return_col].dropna().astype(float)
    n = int(len(returns))

    if n < min_sample:
        return _empty_result(
            current_commercial=current_commercial,
            current_retail=current_retail,
            commercial_net_percentile=commercial_net_pct,
            retail_net_percentile=retail_net_pct,
            cutoff=extreme_cutoff,
            commercial_validated=True,
            retail_validated=True,
            sample_size=n,
            matches=matches,
            reason=f"Nur {n} unabhängige, doppelt bestätigte Muster gefunden.",
        )

    positive = int((returns > 0).sum())
    negative = int((returns < 0).sum())
    positive_rate = positive / n
    negative_rate = negative / n
    if positive_rate > negative_rate:
        bias, successes, hit_rate = "bullish", positive, positive_rate
    elif negative_rate > positive_rate:
        bias, successes, hit_rate = "bearish", negative, negative_rate
    else:
        bias, successes, hit_rate = "neutral", positive, positive_rate

    low, high = _wilson_interval(successes, n)
    sample_factor = min(1.0, np.log1p(n) / np.log1p(50))
    distance_median = float(matches["pattern_distance"].median())
    similarity_factor = float(np.clip(1.0 - distance_median / 3.5, 0.0, 1.0))
    directional_factor = abs(positive_rate - negative_rate)
    quality = 100.0 * (
        0.55 * low + 0.25 * sample_factor + 0.20 * similarity_factor
    ) * directional_factor

    return CotIndexResearch(
        available=True,
        commercial_index=current_commercial,
        retail_index=current_retail,
        commercial_net_percentile=commercial_net_pct,
        retail_net_percentile=retail_net_pct,
        commercial_zone=_zone(current_commercial, extreme_cutoff),
        retail_zone=_zone(current_retail, extreme_cutoff),
        commercial_validated=True,
        retail_validated=True,
        sample_size=n,
        bias=bias,
        hit_rate=hit_rate,
        confidence_low=low,
        confidence_high=high,
        median_return=float(returns.median()),
        q25_return=float(returns.quantile(0.25)),
        q75_return=float(returns.quantile(0.75)),
        best_return=float(returns.max()),
        worst_return=float(returns.min()),
        quality_score=float(quality),
        matches=matches,
        reason="",
    )
