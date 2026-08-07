from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GroupNetResearch:
    available: bool
    group: str
    current_net: float | None
    current_net_oi: float | None
    sample_size: int
    up_count: int
    down_count: int
    dominant_direction: str
    hit_rate: float | None
    median_return: float | None
    range_low: float | None
    range_high: float | None
    matches: pd.DataFrame
    reason: str = ""


@dataclass(frozen=True)
class ReferenceZone:
    zone_id: int
    lower: float
    upper: float
    center: float
    visits: int
    up_count: int
    down_count: int
    dominant_direction: str
    hit_rate: float | None
    median_return: float | None
    range_low: float | None
    range_high: float | None
    is_current: bool


GROUP_COLUMNS = {
    "Commercials": ("commercial_net", "commercial_net_oi"),
    "Retail": ("nonreportable_net", "nonreportable_net_oi"),
}


def _independent_nearest(
    frame: pd.DataFrame,
    feature_cols: list[str],
    n_neighbors: int,
    min_gap_weeks: int,
) -> pd.DataFrame:
    clean = frame.dropna(subset=feature_cols + ["date"]).sort_values("date").copy()
    if len(clean) < 3:
        return clean.iloc[0:0].copy()
    current = clean.iloc[[-1]]
    history = clean.iloc[:-1].copy()
    scaler = StandardScaler().fit(history[feature_cols])
    x_hist = scaler.transform(history[feature_cols])
    x_now = scaler.transform(current[feature_cols])[0]
    history["distance"] = np.sqrt(np.mean((x_hist - x_now) ** 2, axis=1))
    ranked = history.sort_values(["distance", "date"])
    selected: list[int] = []
    selected_dates: list[pd.Timestamp] = []
    gap = pd.Timedelta(weeks=min_gap_weeks)
    for idx, row in ranked.iterrows():
        dt = pd.Timestamp(row["date"])
        if any(abs(dt - used) < gap for used in selected_dates):
            continue
        selected.append(idx)
        selected_dates.append(dt)
        if len(selected) >= n_neighbors:
            break
    return ranked.loc[selected].sort_values("distance").copy()


def _summarize(
    matches: pd.DataFrame,
    return_col: str,
) -> tuple[int, int, int, str, float | None, float | None, float | None, float | None]:
    values = pd.to_numeric(matches.get(return_col), errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return 0, 0, 0, "neutral", None, None, None, None
    up = int((values > 0).sum())
    down = int((values < 0).sum())
    if up > down:
        direction = "bullish"
        directional = values[values > 0]
        hit = up / n
    elif down > up:
        direction = "bearish"
        directional = values[values < 0]
        hit = down / n
    else:
        direction = "neutral"
        directional = pd.Series(dtype=float)
        hit = 0.5
    median = float(values.median())
    low = float(directional.quantile(0.25)) if not directional.empty else None
    high = float(directional.quantile(0.75)) if not directional.empty else None
    return n, up, down, direction, hit, median, low, high


def analyze_group_net(
    df: pd.DataFrame,
    group: str,
    horizon_weeks: int = 8,
    n_neighbors: int = 30,
    min_gap_weeks: int = 8,
    include_oi_context: bool = False,
) -> GroupNetResearch:
    if group not in GROUP_COLUMNS:
        raise ValueError(f"Unbekannte Gruppe: {group}")
    net_col, oi_col = GROUP_COLUMNS[group]
    return_col = f"return_{int(horizon_weeks)}w"
    required = [net_col, oi_col, "date", return_col]
    clean = df.dropna(subset=required).sort_values("date").copy()
    if len(clean) < 3:
        return GroupNetResearch(False, group, None, None, 0, 0, 0, "neutral", None, None, None, None, clean.iloc[0:0], "Zu wenige Daten.")
    feature_cols = [net_col, oi_col] if include_oi_context else [net_col]
    matches = _independent_nearest(clean, feature_cols, n_neighbors, min_gap_weeks)
    n, up, down, direction, hit, median, low, high = _summarize(matches, return_col)
    current = clean.iloc[-1]
    return GroupNetResearch(
        available=n > 0,
        group=group,
        current_net=float(current[net_col]),
        current_net_oi=float(current[oi_col]),
        sample_size=n,
        up_count=up,
        down_count=down,
        dominant_direction=direction,
        hit_rate=hit,
        median_return=median,
        range_low=low,
        range_high=high,
        matches=matches,
        reason="" if n else "Keine vollständigen historischen Vergleichsfälle.",
    )


def analyze_overlap(
    df: pd.DataFrame,
    horizon_weeks: int = 8,
    n_neighbors: int = 30,
    min_gap_weeks: int = 8,
    include_oi_context: bool = False,
) -> GroupNetResearch:
    return_col = f"return_{int(horizon_weeks)}w"
    feature_cols = ["commercial_net", "nonreportable_net"]
    if include_oi_context:
        feature_cols += ["commercial_net_oi", "nonreportable_net_oi"]
    required = feature_cols + ["date", return_col]
    clean = df.dropna(subset=required).sort_values("date").copy()
    if len(clean) < 3:
        return GroupNetResearch(False, "Überschneidung", None, None, 0, 0, 0, "neutral", None, None, None, None, clean.iloc[0:0], "Zu wenige Daten.")
    matches = _independent_nearest(clean, feature_cols, n_neighbors, min_gap_weeks)
    n, up, down, direction, hit, median, low, high = _summarize(matches, return_col)
    current = clean.iloc[-1]
    return GroupNetResearch(
        available=n > 0,
        group="Commercial/Retail-Überschneidung",
        current_net=None,
        current_net_oi=None,
        sample_size=n,
        up_count=up,
        down_count=down,
        dominant_direction=direction,
        hit_rate=hit,
        median_return=median,
        range_low=low,
        range_high=high,
        matches=matches,
        reason="" if n else "Keine vollständigen historischen Vergleichsfälle.",
    )



def analyze_pure_net_pattern(
    df: pd.DataFrame,
    horizon_weeks: int = 8,
    n_neighbors: int = 30,
    min_gap_weeks: int = 8,
) -> GroupNetResearch:
    """Analyse ausschließlich der absoluten Nettopositionen aller drei Gruppen.

    COT-Index, Open Interest und Preismerkmale fließen nicht in die
    Ähnlichkeitssuche ein. Der Preis wird erst nach der Auswahl der
    historischen Nettopositionsmuster zur Ergebnisbewertung verwendet.
    """
    return_col = f"return_{int(horizon_weeks)}w"
    feature_cols = ["commercial_net", "noncommercial_net", "nonreportable_net"]
    required = feature_cols + ["date", return_col]
    clean = df.dropna(subset=required).sort_values("date").copy()
    if len(clean) < 3:
        return GroupNetResearch(
            False, "Reines Nettopositionsmuster", None, None, 0, 0, 0,
            "neutral", None, None, None, None, clean.iloc[0:0],
            "Zu wenige Daten.",
        )
    matches = _independent_nearest(clean, feature_cols, n_neighbors, min_gap_weeks)
    n, up, down, direction, hit, median, low, high = _summarize(matches, return_col)
    return GroupNetResearch(
        available=n > 0,
        group="Reines Nettopositionsmuster",
        current_net=None,
        current_net_oi=None,
        sample_size=n,
        up_count=up,
        down_count=down,
        dominant_direction=direction,
        hit_rate=hit,
        median_return=median,
        range_low=low,
        range_high=high,
        matches=matches,
        reason="" if n else "Keine vollständigen historischen Vergleichsfälle.",
    )


def build_reference_zones(
    df: pd.DataFrame,
    group: str,
    horizon_weeks: int = 8,
    n_zones: int = 8,
    min_gap_weeks: int = 8,
) -> list[ReferenceZone]:
    if group not in GROUP_COLUMNS:
        raise ValueError(f"Unbekannte Gruppe: {group}")
    net_col, _ = GROUP_COLUMNS[group]
    return_col = f"return_{int(horizon_weeks)}w"
    clean = df.dropna(subset=[net_col, return_col, "date"]).sort_values("date").copy()
    if len(clean) < max(30, n_zones * 4):
        return []
    current_value = float(clean.iloc[-1][net_col])
    history = clean.iloc[:-1].copy()
    k = min(n_zones, max(3, len(history) // 20))
    model = KMeans(n_clusters=k, n_init=20, random_state=42)
    history["zone"] = model.fit_predict(history[[net_col]])
    current_zone = int(model.predict(pd.DataFrame({net_col: [current_value]}))[0])
    zones: list[ReferenceZone] = []
    gap = pd.Timedelta(weeks=min_gap_weeks)
    for zone_id, grp in history.groupby("zone"):
        selected_rows = []
        used_dates: list[pd.Timestamp] = []
        for _, row in grp.sort_values("date", ascending=False).iterrows():
            dt = pd.Timestamp(row["date"])
            if any(abs(dt - used) < gap for used in used_dates):
                continue
            selected_rows.append(row)
            used_dates.append(dt)
        if not selected_rows:
            continue
        episode = pd.DataFrame(selected_rows)
        n, up, down, direction, hit, median, low, high = _summarize(episode, return_col)
        zones.append(ReferenceZone(
            zone_id=int(zone_id),
            lower=float(grp[net_col].min()),
            upper=float(grp[net_col].max()),
            center=float(grp[net_col].median()),
            visits=n,
            up_count=up,
            down_count=down,
            dominant_direction=direction,
            hit_rate=hit,
            median_return=median,
            range_low=low,
            range_high=high,
            is_current=int(zone_id) == current_zone,
        ))
    return sorted(zones, key=lambda z: z.center)
