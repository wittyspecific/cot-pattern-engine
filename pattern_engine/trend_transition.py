from __future__ import annotations

import numpy as np
import pandas as pd

from .extremes import analyze_position_extremes
from .divergence import analyze_noncommercial_divergence

FLOW_COLS = ["noncomm_long", "noncomm_short", "comm_long", "comm_short", "nonrep_long", "nonrep_short"]


def _zscore_latest(series: pd.Series, min_periods: int = 52) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < min_periods + 1:
        return float("nan")
    current = float(clean.iloc[-1])
    hist = clean.iloc[:-1]
    std = float(hist.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return 0.0
    return (current - float(hist.mean())) / std


def _flow_label(long_change: float, short_change: float, long_z: float, short_z: float) -> tuple[str, str, str]:
    strong = max(abs(long_z) if np.isfinite(long_z) else 0, abs(short_z) if np.isfinite(short_z) else 0) >= 1.5
    prefix = "Starkes " if strong else ""
    if long_change > 0 and short_change <= 0:
        return "bullish", f"{prefix}Long Building / Short Covering", "Neue Longs werden aufgebaut und/oder Shorts geschlossen."
    if long_change <= 0 and short_change > 0:
        return "bearish", f"{prefix}Short Building / Long Liquidation", "Neue Shorts werden aufgebaut und/oder Longs geschlossen."
    if long_change > 0 and short_change > 0:
        if abs(long_z) >= abs(short_z):
            return "bullish", f"{prefix}Long Building bei beidseitigem Aufbau", "Beide Seiten wachsen, der Long-Aufbau ist relativ ungewöhnlicher."
        return "bearish", f"{prefix}Short Building bei beidseitigem Aufbau", "Beide Seiten wachsen, der Short-Aufbau ist relativ ungewöhnlicher."
    if long_change < 0 and short_change < 0:
        if abs(short_z) >= abs(long_z):
            return "bullish", f"{prefix}Short Covering bei Positionsabbau", "Beide Seiten reduzieren, der Short-Abbau ist relativ ungewöhnlicher."
        return "bearish", f"{prefix}Long Liquidation bei Positionsabbau", "Beide Seiten reduzieren, der Long-Abbau ist relativ ungewöhnlicher."
    return "neutral", "Unklarer Positionsfluss", "Long- und Short-Veränderungen liefern keine klare Richtung."


def analyze_trend_transition(data: pd.DataFrame, flow_weeks: int = 4, extreme_cutoff: float = 90.0) -> dict[str, object]:
    """Independent early trend-transition module.

    The module is intentionally separate from the historical net-position pattern engine.
    It requires an extreme Commercial/Non-Commercial setup, evaluates Retail as support,
    then checks whether Non-Commercial long/short flows begin to converge toward the
    Commercial direction. Price momentum is used only as a final confirmation layer.
    """
    required = {"date", "close", "commercial_net", "noncommercial_net", "nonreportable_net", "oi", *FLOW_COLS}
    if not required.issubset(data.columns) or len(data) < max(105, flow_weeks + 53):
        return {"available": False, "phase": 0, "score": 0, "direction": "neutral", "title": "Nicht verfügbar", "description": "Für die Trendwechsel-Erkennung fehlen Long-/Short-Daten oder ausreichende Historie."}

    frame = data.dropna(subset=list(required)).sort_values("date").copy()
    for base in ("commercial_net", "noncommercial_net", "nonreportable_net"):
        oi_col = f"{base}_oi"
        if oi_col not in frame.columns:
            frame[oi_col] = frame[base].astype(float) / frame["oi"].replace(0, np.nan)
    extremes = analyze_position_extremes(frame, extreme_cutoff=extreme_cutoff)
    if not extremes.get("available"):
        return {"available": False, "phase": 0, "score": 0, "direction": "neutral", "title": "Nicht verfügbar", "description": str(extremes.get("description", ""))}

    signal = extremes.get("signal")
    direction = "bullish" if signal == "bullish_reversal_zone" else "bearish" if signal == "bearish_reversal_zone" else "neutral"
    current = frame.iloc[-1]

    changes = {}
    for col in FLOW_COLS:
        diff_series = frame[col].astype(float).diff(flow_weeks)
        changes[col] = {"change": float(diff_series.iloc[-1]), "z": _zscore_latest(diff_series)}

    nc_dir, nc_label, nc_desc = _flow_label(
        changes["noncomm_long"]["change"], changes["noncomm_short"]["change"],
        changes["noncomm_long"]["z"], changes["noncomm_short"]["z"],
    )

    convergence = direction != "neutral" and nc_dir == direction
    flow_strength = max(abs(changes["noncomm_long"]["z"]), abs(changes["noncomm_short"]["z"]))
    sustained = False
    if len(frame) >= flow_weeks + 3 and direction != "neutral":
        net_diffs = frame["noncommercial_net"].astype(float).diff(flow_weeks).tail(3)
        sustained = bool((net_diffs > 0).sum() >= 2) if direction == "bullish" else bool((net_diffs < 0).sum() >= 2)

    price_change = float(frame["close"].astype(float).pct_change(flow_weeks).iloc[-1])
    price_confirmed = (direction == "bullish" and price_change > 0.01) or (direction == "bearish" and price_change < -0.01)
    divergence = analyze_noncommercial_divergence(frame, horizons=(1, 2, 4, 8))
    divergence_aligned = bool(divergence.get("available")) and divergence.get("signal") == direction

    retail_support = extremes.get("retail_support") == direction
    extreme_confirmed = bool(extremes.get("confirmed_by_oi"))

    extreme_score = 35 if direction != "neutral" else 0
    if extreme_confirmed:
        extreme_score += 5
    if retail_support:
        extreme_score += 10
    flow_score = 0
    if convergence:
        flow_score = 20 + min(10, int(max(0.0, flow_strength - 1.0) * 10))
    if sustained:
        flow_score += 10
    divergence_score = 0
    if divergence_aligned:
        divergence_score = 8 if divergence.get("strength") == "früh / schwach" else 12 if divergence.get("strength") == "mittel" else 15
    price_score = 10 if price_confirmed else 0
    score = min(100, extreme_score + flow_score + divergence_score + price_score)

    if direction == "neutral":
        phase = 0
        title = "Keine Trendtransition"
    elif not convergence:
        phase = 1
        title = "Historische Extrempositionierung"
    elif convergence and not sustained:
        phase = 2
        title = "Früher institutioneller Positionswechsel"
    elif sustained and not price_confirmed:
        phase = 3
        title = "Konvergenz der Non-Commercials bestätigt"
    else:
        phase = 4
        title = "Preis bestätigt die Trendtransition"

    icon = "🟢" if direction == "bullish" else "🔴" if direction == "bearish" else "⚪"
    direction_label = "Bullisch" if direction == "bullish" else "Bearisch" if direction == "bearish" else "Neutral"

    return {
        "available": True,
        "phase": phase,
        "score": score,
        "direction": direction,
        "direction_label": direction_label,
        "icon": icon,
        "title": title,
        "description": "Das Modul bewertet Extreme, Retail-Unterstützung, getrennte Non-Commercial-Long-/Short-Flows, Konvergenz und eine optionale Preisbestätigung. Es ist unabhängig von der statistischen Netto-Pattern-Analyse.",
        "flow_weeks": flow_weeks,
        "noncommercial_flow": nc_label,
        "noncommercial_flow_description": nc_desc,
        "convergence": convergence,
        "sustained": sustained,
        "price_confirmed": price_confirmed,
        "price_change": price_change,
        "divergence": divergence,
        "divergence_aligned": divergence_aligned,
        "retail_support": retail_support,
        "extreme_confirmed_by_oi": extreme_confirmed,
        "changes": changes,
        "extremes": extremes,
        "current_date": pd.Timestamp(current["date"]),
    }
