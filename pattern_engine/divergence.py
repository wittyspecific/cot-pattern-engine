from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_latest(series: pd.Series, min_periods: int = 52) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < min_periods + 1:
        return float("nan")
    current = float(clean.iloc[-1])
    history = clean.iloc[:-1]
    std = float(history.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return 0.0
    return (current - float(history.mean())) / std


def _dominance(active_change: float, other_change: float) -> float:
    """Share of the active position build in total absolute position activity."""
    denominator = abs(active_change) + abs(other_change)
    if denominator == 0:
        return 0.0
    return abs(active_change) / denominator


def _classify_horizon(
    price_change: float,
    long_change: float,
    short_change: float,
    long_z: float,
    short_z: float,
    min_price_move: float,
    min_flow_z: float,
    min_dominance: float,
) -> dict[str, object]:
    """Classify price, long flow and short flow separately.

    A clean active divergence still requires an unusual and sufficiently dominant
    new position build. Mixed cases are retained as informative states instead of
    being collapsed into "no divergence".
    """
    long_dominance = _dominance(long_change, short_change) if long_change > 0 else 0.0
    short_dominance = _dominance(short_change, long_change) if short_change > 0 else 0.0

    long_active = long_change > 0 and np.isfinite(long_z) and long_z >= min_flow_z
    short_active = short_change > 0 and np.isfinite(short_z) and short_z >= min_flow_z
    long_building = long_active and long_dominance >= min_dominance
    short_building = short_active and short_dominance >= min_dominance

    price_down = price_change <= -min_price_move
    price_up = price_change >= min_price_move
    signal = "none"
    state = "neutral"
    score = 0.0
    dominant_flow = "none"
    dominance = max(long_dominance, short_dominance)

    if long_change > 0 and short_change > 0:
        dominant_flow = "long_building" if long_change > short_change else "short_building"
    elif long_change > 0:
        dominant_flow = "long_building"
    elif short_change > 0:
        dominant_flow = "short_building"
    elif long_change < 0 and short_change < 0:
        dominant_flow = "position_reduction"
    elif long_change < 0:
        dominant_flow = "long_liquidation"
    elif short_change < 0:
        dominant_flow = "short_covering"

    if price_down and long_building and not short_active:
        signal, state = "bullish", "bullish_divergence"
        explanation = "Der Preis fällt, während dominant neue Long-Positionen aufgebaut werden."
        score = 2.0 + max(0.0, long_z - min_flow_z) + max(0.0, long_dominance - min_dominance) * 4
    elif price_up and short_building and not long_active:
        signal, state = "bearish", "bearish_divergence"
        explanation = "Der Preis steigt, während dominant neue Short-Positionen aufgebaut werden."
        score = 2.0 + max(0.0, short_z - min_flow_z) + max(0.0, short_dominance - min_dominance) * 4
    elif price_down and long_change > 0 and short_change > 0:
        state = "bullish_accumulation_mixed"
        explanation = (
            "Der Preis fällt und Long-Positionen werden aufgebaut. Gleichzeitig werden auch Shorts aufgebaut. "
            "Dies ist eine gemischte Akkumulationsphase: bullische Akkumulation ist vorhanden, "
            "der Abwärtstrend wird durch Short-Aufbau jedoch weiterhin bestätigt."
        )
    elif price_up and short_change > 0 and long_change > 0:
        state = "bearish_distribution_mixed"
        explanation = (
            "Der Preis steigt und Short-Positionen werden aufgebaut. Gleichzeitig werden auch Longs aufgebaut. "
            "Dies ist eine gemischte Distributionsphase: bearischer Gegenfluss ist vorhanden, "
            "der Aufwärtstrend wird durch Long-Aufbau jedoch weiterhin bestätigt."
        )
    elif price_down and long_change > 0:
        state = "bullish_accumulation"
        explanation = "Der Preis fällt, während Long-Positionen aufgebaut werden: bullische Akkumulation."
    elif price_up and short_change > 0:
        state = "bearish_distribution"
        explanation = "Der Preis steigt, während Short-Positionen aufgebaut werden: bearischer Gegenfluss."
    elif price_down and short_change > 0:
        state = "bearish_trend_confirmation"
        explanation = "Der Preis fällt und neue Shorts werden aufgebaut: bearische Trendbestätigung."
    elif price_up and long_change > 0:
        state = "bullish_trend_confirmation"
        explanation = "Der Preis steigt und neue Longs werden aufgebaut: bullische Trendbestätigung."
    elif long_change < 0 and short_change < 0:
        state = "position_reduction"
        explanation = "Long- und Short-Positionen werden gleichzeitig reduziert."
    else:
        explanation = "Kein eindeutiger neuer Positionsaufbau relativ zur Preisrichtung."

    if signal == "none":
        strength = "Hinweis" if state not in {"neutral", "position_reduction"} else "keine"
    elif score >= 4.0:
        strength = "stark"
    elif score >= 2.8:
        strength = "mittel"
    else:
        strength = "früh / schwach"

    return {
        "signal": signal,
        "state": state,
        "mode": "active_dominant" if signal != "none" else "matrix",
        "strength": strength,
        "score": score,
        "explanation": explanation,
        "dominant_flow": dominant_flow,
        "dominance": dominance,
        "long_dominance": long_dominance,
        "short_dominance": short_dominance,
        "components": {
            "long_building": long_building,
            "short_building": short_building,
            "long_active": long_active,
            "short_active": short_active,
            "long_liquidation": long_change < 0,
            "short_covering": short_change < 0,
        },
    }


def analyze_noncommercial_divergence(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 4, 8),
    min_price_move: float = 0.02,
    min_flow_z: float = 1.0,
    min_dominance: float = 0.70,
    min_position_z: float | None = None,
) -> dict[str, object]:
    """Compare price with active and dominant Non-Commercial position building.

    Signals:
    - bullish: price falls while dominant new long positions are built;
    - bearish: price rises while dominant new short positions are built.

    Position reduction alone (long liquidation or short covering) never creates a
    divergence signal. Horizons are weekly COT intervals, not trading days.
    """
    if min_position_z is not None:
        min_flow_z = float(min_position_z)

    required = {"date", "close", "noncomm_long", "noncomm_short"}
    if not required.issubset(data.columns):
        return {
            "available": False,
            "signal": "none",
            "title": "Nicht verfügbar",
            "icon": "⚪",
            "strength": "keine",
            "description": "Getrennte Non-Commercial-Long-/Short-Daten werden benötigt; eine Netto-Änderung allein erzeugt bewusst kein Divergenzsignal.",
            "confirmed_horizons": 0,
            "active_horizons": 0,
            "horizons": [],
        }

    frame = data[list(required)].dropna().sort_values("date").copy()
    if len(frame) < max(horizons) + 53:
        return {
            "available": False,
            "signal": "none",
            "title": "Nicht verfügbar",
            "icon": "⚪",
            "strength": "keine",
            "description": "Für eine belastbare Flow-Divergenzanalyse liegt noch zu wenig Historie vor.",
            "confirmed_horizons": 0,
            "active_horizons": 0,
            "horizons": [],
        }

    items: list[dict[str, object]] = []
    bullish_score = bearish_score = 0.0
    bullish_count = bearish_count = 0

    for weeks in horizons:
        price_series = frame["close"].astype(float).pct_change(weeks)
        long_series = frame["noncomm_long"].astype(float).diff(weeks)
        short_series = frame["noncomm_short"].astype(float).diff(weeks)
        price_change = float(price_series.iloc[-1])
        long_change = float(long_series.iloc[-1])
        short_change = float(short_series.iloc[-1])
        long_z = _zscore_latest(long_series)
        short_z = _zscore_latest(short_series)
        classification = _classify_horizon(
            price_change,
            long_change,
            short_change,
            long_z,
            short_z,
            min_price_move=min_price_move,
            min_flow_z=min_flow_z,
            min_dominance=min_dominance,
        )
        signal = str(classification["signal"])
        if signal == "bullish":
            bullish_count += 1
            bullish_score += float(classification["score"])
        elif signal == "bearish":
            bearish_count += 1
            bearish_score += float(classification["score"])
        items.append({
            "weeks": weeks,
            "price_change": price_change,
            "long_change": long_change,
            "short_change": short_change,
            "long_z": long_z,
            "short_z": short_z,
            "noncommercial_change": long_change - short_change,
            "noncommercial_change_z": float("nan"),
            **classification,
        })

    if bullish_score > bearish_score and bullish_count:
        signal, icon, title = "bullish", "🟢", "Bullische aktive Non-Commercial-Divergenz"
        description = "Der Preis fällt, während über mindestens ein Zeitfenster dominant neue Long-Positionen aufgebaut werden."
        confirmed, total_score = bullish_count, bullish_score
    elif bearish_score > bullish_score and bearish_count:
        signal, icon, title = "bearish", "🔴", "Bearische aktive Non-Commercial-Divergenz"
        description = "Der Preis steigt, während über mindestens ein Zeitfenster dominant neue Short-Positionen aufgebaut werden."
        confirmed, total_score = bearish_count, bearish_score
    else:
        signal, icon, title = "none", "⚪", "Keine aktive Non-Commercial-Divergenz"
        description = "Über 1, 2, 4 und 8 Wochen ist kein ausreichend dominanter neuer Positionsaufbau gegen die Preisrichtung erkennbar."
        confirmed, total_score = 0, 0.0

    if confirmed >= 3 and total_score >= 8.0:
        strength = "stark"
    elif confirmed >= 2 or total_score >= 3.5:
        strength = "mittel"
    elif confirmed == 1:
        strength = "früh / schwach"
    else:
        strength = "keine"

    return {
        "available": True,
        "signal": signal,
        "title": title,
        "icon": icon,
        "strength": strength,
        "description": description,
        "confirmed_horizons": confirmed,
        "active_horizons": confirmed,
        "min_dominance": min_dominance,
        "horizons": items,
    }
