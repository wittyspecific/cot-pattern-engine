from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class DataQuality:
    latest_date: pd.Timestamp | None
    age_days: int | None
    is_current: bool
    status: str
    message: str


def assess_freshness(frame: pd.DataFrame, date_column: str, max_age_days: int, today: pd.Timestamp | None = None) -> DataQuality:
    if frame.empty or date_column not in frame:
        return DataQuality(None, None, False, "missing", "Keine Daten vorhanden.")
    values = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if values.empty:
        return DataQuality(None, None, False, "invalid", "Kein gültiges Datum vorhanden.")
    latest = values.max().normalize()
    now = (today or pd.Timestamp.today()).normalize()
    age = max(0, int((now - latest).days))
    current = age <= max_age_days
    return DataQuality(
        latest,
        age,
        current,
        "current" if current else "stale",
        f"Aktuell: {latest:%d.%m.%Y}" if current else f"Veraltet: letzter Stichtag {latest:%d.%m.%Y} ({age} Tage)",
    )
