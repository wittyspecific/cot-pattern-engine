from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pattern_engine.analysis import HYBRID_FEATURES


@dataclass(frozen=True)
class ClusterResearch:
    available: bool
    cluster_id: int | None
    cluster_count: int
    sample_size: int
    silhouette: float | None
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


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _independent_events(frame: pd.DataFrame, min_gap_weeks: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected: list[int] = []
    used_dates: list[pd.Timestamp] = []
    gap = pd.Timedelta(weeks=min_gap_weeks)
    # Most recent observation from each episode is retained.
    for idx, row in frame.sort_values("date", ascending=False).iterrows():
        event_date = pd.Timestamp(row["date"])
        if any(abs(event_date - used) < gap for used in used_dates):
            continue
        selected.append(idx)
        used_dates.append(event_date)
    return frame.loc[selected].sort_values("date")


def analyze_current_cluster(
    df: pd.DataFrame,
    horizon_weeks: int = 8,
    min_clusters: int = 3,
    max_clusters: int = 8,
    min_sample: int = 12,
    min_gap_weeks: int = 8,
    random_state: int = 42,
) -> ClusterResearch:
    """Find the current market-state cluster and summarize forward outcomes.

    Clustering uses the three absolute net positions and the corresponding shares
    of open interest. The number of clusters is selected using the best silhouette
    score on historical observations only. The current observation is assigned
    after fitting and is never used to estimate forward performance.
    """
    return_col = f"return_{int(horizon_weeks)}w"
    required = HYBRID_FEATURES + ["date", return_col]
    clean = df.dropna(subset=required).sort_values("date").copy()
    if len(clean) < 80:
        return ClusterResearch(
            available=False,
            cluster_id=None,
            cluster_count=0,
            sample_size=0,
            silhouette=None,
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
            matches=clean.iloc[0:0],
            reason="Zu wenige vollständige Beobachtungen für eine robuste Clusteranalyse.",
        )

    current = clean.iloc[[-1]].copy()
    history = clean.iloc[:-1].copy()
    scaler = StandardScaler().fit(history[HYBRID_FEATURES])
    x_hist = scaler.transform(history[HYBRID_FEATURES])
    x_current = scaler.transform(current[HYBRID_FEATURES])

    upper_k = min(max_clusters, max(min_clusters, len(history) // 20))
    candidates: list[tuple[float, int, KMeans]] = []
    for k in range(min_clusters, upper_k + 1):
        if len(history) <= k:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(x_hist)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(x_hist, labels))
        candidates.append((score, k, model))

    if not candidates:
        return ClusterResearch(
            available=False,
            cluster_id=None,
            cluster_count=0,
            sample_size=0,
            silhouette=None,
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
            matches=history.iloc[0:0],
            reason="Es konnte keine stabile Clusterstruktur bestimmt werden.",
        )

    silhouette, k, model = max(candidates, key=lambda item: item[0])
    history["cluster_id"] = model.labels_
    current_cluster = int(model.predict(x_current)[0])
    matches = history.loc[history["cluster_id"] == current_cluster].copy()
    matches = _independent_events(matches, min_gap_weeks=min_gap_weeks)
    returns = matches[return_col].dropna().astype(float)
    n = int(len(returns))
    if n < min_sample:
        return ClusterResearch(
            available=False,
            cluster_id=current_cluster,
            cluster_count=k,
            sample_size=n,
            silhouette=silhouette,
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
            matches=matches,
            reason=f"Der aktuelle Cluster enthält nur {n} unabhängige Fälle; mindestens {min_sample} werden verlangt.",
        )

    positive = int((returns > 0).sum())
    negative = int((returns < 0).sum())
    positive_rate = positive / n
    negative_rate = negative / n
    if positive_rate >= 0.60:
        bias = "bullish"
        successes = positive
        hit_rate = positive_rate
    elif negative_rate >= 0.60:
        bias = "bearish"
        successes = negative
        hit_rate = negative_rate
    else:
        bias = "neutral"
        successes = max(positive, negative)
        hit_rate = max(positive_rate, negative_rate)

    low, high = _wilson_interval(successes, n)
    # Conservative ranking: lower CI bound, sample depth and cluster separation.
    sample_factor = min(1.0, np.log1p(n) / np.log1p(50))
    silhouette_factor = float(np.clip((silhouette + 0.1) / 0.6, 0.0, 1.0))
    directional_factor = 0.0 if bias == "neutral" else 1.0
    quality_score = 100.0 * directional_factor * (0.65 * low + 0.20 * sample_factor + 0.15 * silhouette_factor)

    return ClusterResearch(
        True,
        current_cluster,
        k,
        n,
        silhouette,
        bias,
        hit_rate,
        low,
        high,
        float(returns.median()),
        float(returns.quantile(0.25)),
        float(returns.quantile(0.75)),
        float(returns.max()),
        float(returns.min()),
        float(quality_score),
        matches,
        "",
    )


@dataclass(frozen=True)
class ClusterTiming:
    available: bool
    observation_days: int
    onset_day: int | None
    onset_window_start: int | None
    onset_window_end: int | None
    peak_day: int | None
    peak_median_return: float | None
    terminal_median_return: float | None
    current_trading_day: int
    status: str
    status_text: str
    median_path: pd.DataFrame
    reason: str


def analyze_cluster_timing(
    matches: pd.DataFrame,
    prices: pd.DataFrame,
    bias: str,
    observation_weeks: int = 8,
    minimum_move: float = 0.005,
) -> ClusterTiming:
    """Estimate when the median cluster move typically starts and peaks.

    Timing is measured from the COT report date (Tuesday). The median path is
    calculated from daily closes of independent historical cluster episodes.
    The onset is the first day on which the median path reaches a robust
    directional threshold and remains in that direction on at least three of
    the following five observations. Individual winning paths are also used to
    form an interquartile onset window.
    """
    if matches.empty or bias not in {"bullish", "bearish"}:
        return ClusterTiming(False, observation_weeks * 5, None, None, None, None, None, None, 0, "neutral", "Kein eindeutiger Richtungscluster.", pd.DataFrame(), "Kein eindeutiger Richtungscluster.")

    px = prices[["date", "close"]].dropna().sort_values("date").copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.drop_duplicates("date", keep="last").reset_index(drop=True)
    if px.empty:
        return ClusterTiming(False, observation_weeks * 5, None, None, None, None, None, None, 0, "unavailable", "Keine Preisdaten verfügbar.", pd.DataFrame(), "Keine Preisdaten verfügbar.")

    max_days = int(observation_weeks * 5)
    direction = 1.0 if bias == "bullish" else -1.0
    paths: list[pd.Series] = []
    individual_onsets: list[int] = []

    for _, event in matches.iterrows():
        event_date = pd.Timestamp(event["date"]).normalize()
        segment = px.loc[px["date"] >= event_date].head(max_days + 1)
        if len(segment) < max_days + 1:
            continue
        path = segment["close"].astype(float).reset_index(drop=True)
        path = path / float(path.iloc[0]) - 1.0
        path.index = range(len(path))
        paths.append(path)

        directional = path * direction
        final_directional = float(directional.iloc[-1])
        if final_directional <= 0:
            continue
        threshold = max(float(minimum_move), 0.25 * final_directional)
        reached = np.flatnonzero(directional.to_numpy() >= threshold)
        for day in reached:
            end = min(len(directional), int(day) + 5)
            if int((directional.iloc[int(day):end] > 0).sum()) >= min(3, end - int(day)):
                individual_onsets.append(int(day))
                break

    if len(paths) < 5:
        return ClusterTiming(False, max_days, None, None, None, None, None, None, 0, "unavailable", "Zu wenige vollständige historische Preisverläufe.", pd.DataFrame(), "Zu wenige vollständige historische Preisverläufe.")

    matrix = pd.concat(paths, axis=1)
    median = matrix.median(axis=1)
    q25 = matrix.quantile(0.25, axis=1)
    q75 = matrix.quantile(0.75, axis=1)
    directional_median = median * direction
    terminal_directional = float(directional_median.iloc[-1])
    robust_threshold = max(float(minimum_move), 0.25 * max(terminal_directional, 0.0))

    onset_day = None
    reached = np.flatnonzero(directional_median.to_numpy() >= robust_threshold)
    for day in reached:
        end = min(len(directional_median), int(day) + 5)
        if int((directional_median.iloc[int(day):end] > 0).sum()) >= min(3, end - int(day)):
            onset_day = int(day)
            break

    peak_day = int(np.argmax(directional_median.to_numpy()))
    peak_return = float(median.iloc[peak_day])
    terminal_return = float(median.iloc[-1])

    if individual_onsets:
        window_start = max(1, int(np.floor(np.quantile(individual_onsets, 0.25))))
        window_end = max(window_start, int(np.ceil(np.quantile(individual_onsets, 0.75))))
    else:
        window_start = onset_day
        window_end = onset_day

    current_report = pd.Timestamp(matches["date"].max()).normalize()
    # The true current report is generally later than every historical match;
    # use the latest COT date represented in the caller when available via attrs.
    if "current_report_date" in matches.attrs:
        current_report = pd.Timestamp(matches.attrs["current_report_date"]).normalize()
    current_trading_day = int((px["date"] >= current_report).sum() - 1)
    current_trading_day = max(0, current_trading_day)

    if onset_day is None:
        status = "unclear"
        status_text = "Im Medianpfad lässt sich kein stabiler Bewegungsbeginn bestimmen."
    elif current_trading_day < (window_start or onset_day):
        remaining = (window_start or onset_day) - current_trading_day
        status = "before"
        status_text = f"Noch vor dem historischen Suchfenster; Beginn typischerweise in etwa {remaining} Handelstag(en)."
    elif current_trading_day <= (window_end or onset_day):
        status = "onset"
        status_text = "Der Markt befindet sich im typischen historischen Beginnfenster."
    elif current_trading_day <= peak_day:
        status = "active"
        status_text = "Die historische Medianbewegung wäre typischerweise bereits aktiv, aber noch vor ihrem Medianmaximum."
    else:
        status = "late"
        status_text = "Das historische Medianmaximum liegt zeitlich bereits zurück; neue Einstiege sind weniger klar aus dem Cluster-Timing ableitbar."

    path_frame = pd.DataFrame({
        "Handelstag": median.index.astype(int),
        "Median": median.values,
        "25. Perzentil": q25.values,
        "75. Perzentil": q75.values,
    })

    return ClusterTiming(
        available=True,
        observation_days=max_days,
        onset_day=onset_day,
        onset_window_start=window_start,
        onset_window_end=window_end,
        peak_day=peak_day,
        peak_median_return=peak_return,
        terminal_median_return=terminal_return,
        current_trading_day=current_trading_day,
        status=status,
        status_text=status_text,
        median_path=path_frame,
        reason="",
    )
