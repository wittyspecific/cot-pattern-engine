from __future__ import annotations

from math import sqrt
from typing import Mapping, Any

import numpy as np
import pandas as pd

DIRECTIONS = {"Bullish", "Bearish"}
STATUS_ORDER = {
    "Vollständig bestätigt": 0,
    "Teilweise bestätigt": 1,
    "Widersprüchlich": 2,
    "Daten unzureichend": 3,
}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial directional hit rate."""
    if total <= 0 or successes < 0 or successes > total:
        return None, None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def module_evidence(
    direction: str | None,
    sample_size: int | None,
    up_count: int | None,
    down_count: int | None,
    median_return: float | None,
    range_low: float | None = None,
    range_high: float | None = None,
    available: bool = True,
    reason: str = "",
) -> dict[str, Any]:
    """Normalize one module without combining or weighting it with another."""
    normalized = str(direction or "Nicht verfügbar").title()
    n = max(int(sample_size or 0), 0)
    up = max(int(up_count or 0), 0)
    down = max(int(down_count or 0), 0)
    directional_successes = up if normalized == "Bullish" else down if normalized == "Bearish" else 0
    hit_rate = directional_successes / n if n else None
    low, high = wilson_interval(directional_successes, n) if normalized in DIRECTIONS else (None, None)
    return {
        "available": bool(available and n > 0),
        "direction": normalized,
        "sample_size": n,
        "up_count": up,
        "down_count": down,
        "hit_rate": hit_rate,
        "confidence_low": low,
        "confidence_high": high,
        "median_return": None if median_return is None or pd.isna(median_return) else float(median_return),
        "range_low": None if range_low is None or pd.isna(range_low) else float(range_low),
        "range_high": None if range_high is None or pd.isna(range_high) else float(range_high),
        "reason": reason,
    }


def classify_modules(modules: Mapping[str, Mapping[str, Any]]) -> tuple[str, str, int]:
    directional = [m for m in modules.values() if m.get("available") and m.get("direction") in DIRECTIONS]
    if not directional:
        return "Daten unzureichend", "–", 0
    counts = {
        direction: sum(1 for m in directional if m.get("direction") == direction)
        for direction in DIRECTIONS
    }
    present = [direction for direction, count in counts.items() if count > 0]
    if len(present) > 1:
        dominant = max(counts, key=counts.get)
        return "Widersprüchlich", dominant if counts[dominant] > 1 else "Konflikt", max(counts.values())
    direction = present[0]
    confirmations = counts[direction]
    if confirmations == 3:
        return "Vollständig bestätigt", direction, confirmations
    if confirmations >= 2:
        return "Teilweise bestätigt", direction, confirmations
    return "Daten unzureichend", direction, confirmations


def _date(value: Any) -> pd.Timestamp | None:
    if value is None or value == "" or pd.isna(value):
        return None
    result = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result).tz_localize(None) if getattr(result, "tzinfo", None) else pd.Timestamp(result)


def common_history_years(starts: list[Any], ends: list[Any]) -> float | None:
    start_values = [v for v in (_date(x) for x in starts) if v is not None]
    end_values = [v for v in (_date(x) for x in ends) if v is not None]
    if not start_values or not end_values:
        return None
    start, end = max(start_values), min(end_values)
    if end <= start:
        return None
    return float((end - start).days / 365.25)


def build_confluence_row(
    asset: str,
    category: str,
    horizon_weeks: int,
    cot: Mapping[str, Any],
    fundamental: Mapping[str, Any],
    seasonality: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    modules = {"COT": dict(cot), "Fundamental": dict(fundamental), "Seasonality": dict(seasonality)}
    status, direction, confirmations = classify_modules(modules)
    confirming = [m for m in modules.values() if m.get("available") and m.get("direction") == direction]
    lower_bounds = [float(m["confidence_low"]) for m in confirming if m.get("confidence_low") is not None]
    samples = [int(m["sample_size"]) for m in confirming if int(m.get("sample_size") or 0) > 0]
    moves = [abs(float(m["median_return"])) for m in confirming if m.get("median_return") is not None]
    meta = dict(metadata or {})

    row: dict[str, Any] = {
        "Asset": asset,
        "Kategorie": category,
        "Horizont": int(horizon_weeks),
        "Status": status,
        "Richtung": direction,
        "Bestätigungen": confirmations,
        "Verfügbare Module": sum(bool(m.get("available")) for m in modules.values()),
        "Robuste KI-Untergrenze": min(lower_bounds) if lower_bounds else None,
        "Schwächste Stichprobe": min(samples) if samples else 0,
        "Kleinste Medianbewegung": min(moves) if moves else None,
    }
    for name, module in modules.items():
        row.update({
            f"{name} Richtung": module.get("direction", "Nicht verfügbar"),
            f"{name} Fälle": int(module.get("sample_size") or 0),
            f"{name} Trefferquote": module.get("hit_rate"),
            f"{name} KI unten": module.get("confidence_low"),
            f"{name} KI oben": module.get("confidence_high"),
            f"{name} Median": module.get("median_return"),
            f"{name} Range unten": module.get("range_low"),
            f"{name} Range oben": module.get("range_high"),
            f"{name} Hinweis": module.get("reason", ""),
        })
    row.update(meta)
    row["Gemeinsamer Zeitraum Jahre"] = common_history_years(
        [meta.get("COT-Start"), meta.get("Fundamental-Start"), meta.get("Seasonality-Start")],
        [meta.get("COT-Stichtag"), meta.get("Fundamental-Snapshot"), meta.get("Preis-Stichtag")],
    )
    return row


def rank_confluence(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return frame
    out = frame.copy()
    out["_status_rank"] = out["Status"].map(STATUS_ORDER).fillna(99)
    out["_lower"] = pd.to_numeric(out["Robuste KI-Untergrenze"], errors="coerce").fillna(-1.0)
    out["_sample"] = pd.to_numeric(out["Schwächste Stichprobe"], errors="coerce").fillna(0)
    out["_move"] = pd.to_numeric(out["Kleinste Medianbewegung"], errors="coerce").fillna(-1.0)
    out = out.sort_values(
        ["_status_rank", "Bestätigungen", "_lower", "_sample", "_move", "Asset"],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    )
    return out.drop(columns=["_status_rank", "_lower", "_sample", "_move"]).reset_index(drop=True)
