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
    """Classify only active, dominant position building against price direction.

    A divergence is intentionally *not* generated from long liquidation or short
    covering alone. This prevents profit-taking and position reduction from being
    interpreted as fresh directional conviction.
    """
    long_dominance = _dominance(long_change, short_change) if long_change > 0 else 0.0
    short_dominance = _dominance(short_change, long_change) if short_change > 0 else 0.0

    long_building = (
        long_change > 0
        and np.isfinite(long_z)
        and long_z >= min_flow_z
        and long_dominance >= min_dominance
    )
    short_building = (
        short_change > 0
        and np.isfinite(short_z)
        and short_z >= min_flow_z
        and short_dominance >= min_dominance
    )

    signal = "none"
    explanation = "Keine aktive, dominante Positionseröffnung gegen die Preisrichtung."
    score = 0.0
    dominant_flow = "none"
    dominance = max(long_dominance, short_dominance)

    if price_change <= -min_price_move and long_building:
        signal = "bullish"
        dominant_flow = "long_building"
        dominance = long_dominance
        explanation = (
            "Der Preis fällt, während die Non-Commercials dominant neue Long-Positionen aufbauen. "
            "Reine Short-Eindeckungen werden nicht als Divergenz gewertet."
        )
        score = 2.0 + max(0.0, long_z - min_flow_z) + max(0.0, dominance - min_dominance) * 4
    elif price_change >= min_price_move and short_building:
        signal = "bearish"
        dominant_flow = "short_building"
        dominance = short_dominance
        explanation = (
            "Der Preis steigt, während die Non-Commercials dominant neue Short-Positionen aufbauen. "
            "Reine Long-Liquidationen werden nicht als Divergenz gewertet."
        )
        score = 2.0 + max(0.0, short_z - min_flow_z) + max(0.0, dominance - min_dominance) * 4
    elif price_change <= -min_price_move and long_change > 0 and long_z >= min_flow_z:
        explanation = f"Long-Aufbau vorhanden, aber mit {long_dominance:.0%} nicht dominant genug (Minimum {min_dominance:.0%})."
    elif price_change >= min_price_move and short_change > 0 and short_z >= min_flow_z:
        explanation = f"Short-Aufbau vorhanden, aber mit {short_dominance:.0%} nicht dominant genug (Minimum {min_dominance:.0%})."

    if signal == "none":
        strength = "keine"
    elif score >= 4.0:
        strength = "stark"
    elif score >= 2.8:
        strength = "mittel"
    else:
        strength = "früh / schwach"

    return {
        "signal": signal,
        "mode": "active_dominant" if signal != "none" else "none",
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
