from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

NET_COLS = ["commercial_net", "noncommercial_net", "nonreportable_net"]


def prepare_dataset(cot: pd.DataFrame, prices: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    cot = cot.copy().sort_values("date")
    prices = prices.copy().sort_values("date")
    px = prices[["date", "close"]].dropna()
    merged = pd.merge_asof(cot, px, on="date", direction="backward")
    for col in NET_COLS:
        merged[f"{col}_oi"] = merged[col] / merged["oi"].replace(0, np.nan)
        rolling = merged[col].rolling(156, min_periods=52)
        merged[f"{col}_z"] = (merged[col] - rolling.mean()) / rolling.std(ddof=0)
        merged[f"{col}_pct"] = merged[col].rolling(156, min_periods=52).rank(pct=True)
    for weeks in horizons:
        target_dates = merged["date"] + pd.to_timedelta(weeks * 7, unit="D")
        future = pd.merge_asof(
            pd.DataFrame({"target": target_dates}).sort_values("target"),
            px.rename(columns={"date": "target", "close": "future_close"}),
            on="target",
            direction="forward",
        )
        merged[f"return_{weeks}w"] = future["future_close"].to_numpy() / merged["close"].to_numpy() - 1
    return merged


def feature_columns(mode: str) -> list[str]:
    suffix = {
        "Absolute Nettoposition": "",
        "% Open Interest": "_oi",
        "Z-Score": "_z",
        "Perzentil": "_pct",
    }[mode]
    return [f"{c}{suffix}" for c in NET_COLS]


def scenario_filter(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, (low, high) in bounds.items():
        mask &= df[col].between(low, high, inclusive="both")
    return df.loc[mask].copy()


def similar_weeks(df: pd.DataFrame, columns: list[str], n_neighbors: int = 30) -> pd.DataFrame:
    clean = df.dropna(subset=columns).copy()
    if len(clean) < 3:
        return clean
    current = clean.iloc[[-1]]
    history = clean.iloc[:-1]
    n = min(n_neighbors, len(history))
    scaler = StandardScaler().fit(history[columns])
    x_hist = scaler.transform(history[columns])
    x_now = scaler.transform(current[columns])
    model = NearestNeighbors(n_neighbors=n).fit(x_hist)
    distances, indices = model.kneighbors(x_now)
    result = history.iloc[indices[0]].copy()
    result["distance"] = distances[0]
    return result.sort_values("distance")


def summary_stats(matches: pd.DataFrame, horizon: int) -> dict[str, float]:
    s = matches[f"return_{horizon}w"].dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "hit_rate": np.nan,
            "best": np.nan,
            "worst": np.nan,
            "std": np.nan,
        }
    return {
        "count": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "hit_rate": (s > 0).mean(),
        "best": s.max(),
        "worst": s.min(),
        "std": s.std(ddof=1),
    }


def event_paths(matches: pd.DataFrame, prices: pd.DataFrame, weeks: int) -> pd.DataFrame:
    prices = prices.sort_values("date").set_index("date")["close"]
    paths = []
    max_days = weeks * 5
    for _, row in matches.iterrows():
        segment = prices.loc[prices.index >= row["date"]].head(max_days + 1)
        if len(segment) < 2:
            continue
        normalized = segment / segment.iloc[0] - 1
        normalized.index = range(len(normalized))
        paths.append(normalized.rename(str(row["date"].date())))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat(paths, axis=1)
    frame["Mittelwert"] = frame.mean(axis=1)
    frame["Median"] = frame.drop(columns=["Mittelwert"]).median(axis=1)
    frame.index.name = "Handelstage"
    return frame


def timing_analysis(
    matches: pd.DataFrame,
    prices: pd.DataFrame,
    observation_weeks: int = 8,
    move_threshold: float = 0.01,
) -> dict[str, object]:
    """Estimate directional bias and typical first movement window.

    Direction is classified by the closing return at the end of a fixed eight-week
    observation window. For winning cases, onset is the first trading day on which
    the price reaches the configured threshold in the dominant direction.
    """
    px = prices[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    max_trading_days = observation_weeks * 5
    rows: list[dict[str, object]] = []

    for _, event in matches.iterrows():
        event_date = pd.Timestamp(event["date"])
        segment = px.loc[px["date"] >= event_date].head(max_trading_days + 1).copy()
        if len(segment) < max_trading_days + 1:
            continue
        start_price = float(segment.iloc[0]["close"])
        path = segment["close"].astype(float) / start_price - 1.0
        final_return = float(path.iloc[-1])
        rows.append(
            {
                "date": event_date,
                "start_price": start_price,
                "final_return": final_return,
                "path": path.reset_index(drop=True),
            }
        )

    if not rows:
        return {
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": np.nan,
            "negative_rate": np.nan,
            "bias": "neutral",
            "onset_days": [],
            "window_start": None,
            "window_end": None,
            "median_onset": None,
            "details": pd.DataFrame(),
        }

    positive_count = sum(float(row["final_return"]) > 0 for row in rows)
    negative_count = sum(float(row["final_return"]) < 0 for row in rows)
    count = len(rows)
    positive_rate = positive_count / count
    negative_rate = negative_count / count

    if positive_rate >= 0.60:
        bias = "bullish"
        direction = 1
    elif negative_rate >= 0.60:
        bias = "bearish"
        direction = -1
    else:
        bias = "neutral"
        direction = 0

    onset_days: list[int] = []
    details: list[dict[str, object]] = []
    for row in rows:
        onset = None
        final_return = float(row["final_return"])
        is_directional_winner = direction != 0 and final_return * direction > 0
        if is_directional_winner:
            directional_path = row["path"] * direction
            reached = np.flatnonzero(directional_path.to_numpy() >= move_threshold)
            if len(reached):
                onset = int(reached[0])
                onset_days.append(onset)
        details.append(
            {
                "date": row["date"],
                "start_price": row["start_price"],
                "final_return": final_return,
                "direction": "Bullish" if final_return > 0 else ("Bearish" if final_return < 0 else "Unverändert"),
                "onset_trading_day": onset,
            }
        )

    if onset_days:
        window_start = max(1, int(np.floor(np.quantile(onset_days, 0.25))))
        window_end = max(window_start, int(np.ceil(np.quantile(onset_days, 0.75))))
        median_onset = int(round(float(np.median(onset_days))))
    else:
        window_start = None
        window_end = None
        median_onset = None

    return {
        "count": count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "bias": bias,
        "onset_days": onset_days,
        "window_start": window_start,
        "window_end": window_end,
        "median_onset": median_onset,
        "details": pd.DataFrame(details),
    }


OI_RATIO_COLS = [
    "commercial_net_oi",
    "noncommercial_net_oi",
    "nonreportable_net_oi",
]


def net_similar_weeks(
    df: pd.DataFrame,
    n_neighbors: int = 30,
    min_gap_weeks: int = 8,
) -> pd.DataFrame:
    """Select independent historical weeks using absolute net positions only."""
    columns = NET_COLS
    clean = df.dropna(subset=columns + ["date"]).sort_values("date").copy()
    if len(clean) < 3:
        return clean.iloc[0:0].copy()

    current = clean.iloc[[-1]]
    history = clean.iloc[:-1].copy()
    scaler = StandardScaler().fit(history[columns])
    x_hist = scaler.transform(history[columns])
    x_now = scaler.transform(current[columns])[0]
    distances = np.sqrt(np.mean(np.square(x_hist - x_now), axis=1))

    ranked = history.copy()
    ranked["net_distance"] = distances
    ranked = ranked.sort_values(["net_distance", "date"])

    selected_indices: list[int] = []
    selected_dates: list[pd.Timestamp] = []
    gap = pd.Timedelta(weeks=min_gap_weeks)
    for idx, row in ranked.iterrows():
        event_date = pd.Timestamp(row["date"])
        if any(abs(event_date - used_date) < gap for used_date in selected_dates):
            continue
        selected_indices.append(idx)
        selected_dates.append(event_date)
        if len(selected_indices) >= n_neighbors:
            break

    result = ranked.loc[selected_indices].copy()
    if result.empty:
        return result
    scale = max(float(result["net_distance"].median()), 1e-9)
    result["net_similarity_score"] = 100.0 * np.exp(-result["net_distance"] / scale)
    return result.sort_values("net_distance")


def validate_open_interest_context(
    primary_matches: pd.DataFrame,
    full_df: pd.DataFrame,
    max_rms_z: float = 1.0,
) -> pd.DataFrame:
    """Filter primary net-position matches by comparable OI-relative positioning.

    The three net-position shares of open interest are standardized on all
    historical observations. A match is confirmed when its root-mean-square
    standardized distance from the current report is at most ``max_rms_z``.
    With the default of 1.0, the three relative position measures differ by
    approximately no more than one historical standard deviation on average.
    """
    clean = full_df.dropna(subset=OI_RATIO_COLS + ["date"]).sort_values("date").copy()
    if clean.empty or primary_matches.empty:
        return primary_matches.iloc[0:0].copy()

    current = clean.iloc[[-1]]
    history = clean.iloc[:-1]
    scaler = StandardScaler().fit(history[OI_RATIO_COLS])
    x_now = scaler.transform(current[OI_RATIO_COLS])[0]

    candidates = primary_matches.dropna(subset=OI_RATIO_COLS).copy()
    if candidates.empty:
        return candidates
    x_candidates = scaler.transform(candidates[OI_RATIO_COLS])
    distances = np.sqrt(np.mean(np.square(x_candidates - x_now), axis=1))
    candidates["oi_context_distance"] = distances
    candidates["oi_context_similarity"] = 100.0 * np.exp(-distances)
    candidates["oi_context_confirmed"] = distances <= float(max_rms_z)
    return candidates.loc[candidates["oi_context_confirmed"]].sort_values("oi_context_distance")


HYBRID_FEATURES = [
    "commercial_net",
    "noncommercial_net",
    "nonreportable_net",
    "commercial_net_oi",
    "noncommercial_net_oi",
    "nonreportable_net_oi",
]


def hybrid_similar_weeks(
    df: pd.DataFrame,
    n_neighbors: int = 30,
    min_gap_weeks: int = 8,
    net_weight: float = 0.55,
) -> pd.DataFrame:
    """Select independent historical weeks similar to the current COT structure.

    Both absolute net positions and net positions as a share of open interest are
    compared. Each feature is standardized on historical observations. The total
    weight is split between the three absolute and three relative features.
    Consecutive observations from the same market episode are de-duplicated using
    a minimum date gap.
    """
    columns = HYBRID_FEATURES
    clean = df.dropna(subset=columns + ["date"]).sort_values("date").copy()
    if len(clean) < 3:
        return clean.iloc[0:0].copy()

    current = clean.iloc[[-1]]
    history = clean.iloc[:-1].copy()
    if history.empty:
        return history

    scaler = StandardScaler().fit(history[columns])
    x_hist = scaler.transform(history[columns])
    x_now = scaler.transform(current[columns])[0]

    net_weight = float(np.clip(net_weight, 0.0, 1.0))
    group_weights = np.array(
        [net_weight / 3] * 3 + [(1.0 - net_weight) / 3] * 3,
        dtype=float,
    )
    weighted_diff = (x_hist - x_now) * np.sqrt(group_weights)
    distances = np.sqrt(np.square(weighted_diff).sum(axis=1))

    ranked = history.copy()
    ranked["distance"] = distances
    ranked = ranked.sort_values(["distance", "date"])

    selected_indices: list[int] = []
    selected_dates: list[pd.Timestamp] = []
    gap = pd.Timedelta(weeks=min_gap_weeks)
    for idx, row in ranked.iterrows():
        event_date = pd.Timestamp(row["date"])
        if any(abs(event_date - used_date) < gap for used_date in selected_dates):
            continue
        selected_indices.append(idx)
        selected_dates.append(event_date)
        if len(selected_indices) >= n_neighbors:
            break

    result = ranked.loc[selected_indices].copy()
    if result.empty:
        return result

    # Relative 0-100 score for communication; ranking itself uses distance.
    scale = max(float(result["distance"].median()), 1e-9)
    result["similarity_score"] = 100.0 * np.exp(-result["distance"] / scale)
    return result.sort_values("distance")
