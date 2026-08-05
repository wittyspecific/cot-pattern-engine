from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import NET_COLS, OI_RATIO_COLS


@dataclass(frozen=True)
class GroupExtreme:
    label: str
    net_value: float
    oi_ratio: float
    net_percentile: float
    oi_percentile: float
    combined_percentile: float
    state: str
    icon: str


def _empirical_percentile(history: pd.Series, current: float) -> float:
    values = pd.to_numeric(history, errors="coerce").dropna()
    if values.empty or not np.isfinite(current):
        return float("nan")
    # Mid-rank empirical percentile. Equal values receive half weight.
    below = float((values < current).sum())
    equal = float((values == current).sum())
    return 100.0 * (below + 0.5 * equal) / len(values)


def _state(percentile: float, extreme_cutoff: float) -> tuple[str, str]:
    if not np.isfinite(percentile):
        return "Keine Aussage", "⚪"
    lower = 100.0 - extreme_cutoff
    if percentile >= extreme_cutoff:
        return "Historisch extrem hoch", "🔴"
    if percentile <= lower:
        return "Historisch extrem niedrig", "🔴"
    if percentile >= 80.0:
        return "Hoch", "🟠"
    if percentile <= 20.0:
        return "Niedrig", "🟠"
    return "Normal", "⚪"


def analyze_position_extremes(
    data: pd.DataFrame,
    extreme_cutoff: float = 90.0,
    min_history: int = 104,
) -> dict[str, object]:
    """Classify current COT positioning against its own historical distribution.

    Absolute net positions and net positions as a share of open interest are
    evaluated independently. Their mean percentile is shown as a compact summary,
    while the reversal-zone signal is based primarily on the opposing Commercial
    and Non-Commercial extremes. It is a context warning, not an entry signal.
    """
    required = NET_COLS + OI_RATIO_COLS + ["date"]
    clean = data.dropna(subset=required).sort_values("date").copy()
    if len(clean) < min_history + 1:
        return {
            "available": False,
            "history_count": max(0, len(clean) - 1),
            "groups": [],
            "signal": "insufficient",
            "title": "Nicht genügend Historie",
            "icon": "⚪",
            "description": "Für eine belastbare Extremanalyse werden mindestens zwei Jahre Historie benötigt.",
            "confirmed_by_oi": False,
        }

    current = clean.iloc[-1]
    history = clean.iloc[:-1]
    labels = ["Commercials", "Non-Commercials", "Retail / Non-Reportables"]
    groups: list[GroupExtreme] = []

    for label, net_col, oi_col in zip(labels, NET_COLS, OI_RATIO_COLS):
        net_pct = _empirical_percentile(history[net_col], float(current[net_col]))
        oi_pct = _empirical_percentile(history[oi_col], float(current[oi_col]))
        combined = float(np.nanmean([net_pct, oi_pct]))
        state, icon = _state(combined, extreme_cutoff)
        groups.append(
            GroupExtreme(
                label=label,
                net_value=float(current[net_col]),
                oi_ratio=float(current[oi_col]),
                net_percentile=net_pct,
                oi_percentile=oi_pct,
                combined_percentile=combined,
                state=state,
                icon=icon,
            )
        )

    commercial, noncommercial, retail = groups
    lower = 100.0 - extreme_cutoff

    abs_bullish = commercial.net_percentile >= extreme_cutoff and noncommercial.net_percentile <= lower
    abs_bearish = commercial.net_percentile <= lower and noncommercial.net_percentile >= extreme_cutoff
    oi_bullish = commercial.oi_percentile >= extreme_cutoff and noncommercial.oi_percentile <= lower
    oi_bearish = commercial.oi_percentile <= lower and noncommercial.oi_percentile >= extreme_cutoff

    if abs_bullish or oi_bullish:
        signal = "bullish_reversal_zone"
        icon = "🟢"
        title = "Bullischer historischer Extrembereich"
        description = (
            "Commercials sind historisch sehr stark long positioniert, während Non-Commercials "
            "historisch sehr stark short positioniert sind. Das kann auf erhöhtes Potenzial für "
            "eine bullische Trendwende hinweisen, bestätigt aber weder Zeitpunkt noch Einstieg."
        )
        confirmed = abs_bullish and oi_bullish
    elif abs_bearish or oi_bearish:
        signal = "bearish_reversal_zone"
        icon = "🔴"
        title = "Bearischer historischer Extrembereich"
        description = (
            "Commercials sind historisch sehr stark short positioniert, während Non-Commercials "
            "historisch sehr stark long positioniert sind. Das kann auf erhöhtes Potenzial für "
            "eine bearische Trendwende hinweisen, bestätigt aber weder Zeitpunkt noch Einstieg."
        )
        confirmed = abs_bearish and oi_bearish
    else:
        signal = "no_joint_extreme"
        icon = "⚪"
        title = "Kein gemeinsamer Wendebereich"
        description = (
            "Mindestens eine Händlergruppe ist auffällig positioniert, die gegenläufigen Extreme "
            "von Commercials und Non-Commercials liegen jedoch nicht gleichzeitig vor."
            if any(g.state != "Normal" for g in groups)
            else "Die aktuelle Positionierung liegt überwiegend innerhalb ihrer normalen historischen Bandbreite."
        )
        confirmed = False

    return {
        "available": True,
        "history_count": len(history),
        "groups": groups,
        "signal": signal,
        "title": title,
        "icon": icon,
        "description": description,
        "confirmed_by_oi": confirmed,
        "absolute_signal": "bullish" if abs_bullish else ("bearish" if abs_bearish else "neutral"),
        "oi_signal": "bullish" if oi_bullish else ("bearish" if oi_bearish else "neutral"),
        "retail_support": (
            "bullish" if retail.combined_percentile <= lower else
            "bearish" if retail.combined_percentile >= extreme_cutoff else
            "neutral"
        ),
        "extreme_cutoff": extreme_cutoff,
    }
