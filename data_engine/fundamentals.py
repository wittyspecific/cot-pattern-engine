from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .settings import DATA_DIR
from fundamental_engine.config import FundamentalSeriesSpec

FRED_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _cache_path(series_id: str) -> Path:
    return DATA_DIR / f"fundamental_fred_{series_id.lower()}.parquet"


def _cache_is_fresh(path: Path, max_age_days: int = 7) -> bool:
    if not path.exists():
        return False
    modified = pd.Timestamp(path.stat().st_mtime, unit="s")
    return modified >= pd.Timestamp.now() - pd.Timedelta(days=max_age_days)


def _read_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["date", "value"]).sort_values("date")


def fetch_fred_series(
    spec: FundamentalSeriesSpec,
    force: bool = False,
    start: str = "1980-01-01",
) -> tuple[pd.DataFrame, str]:
    """Load one FRED series without requiring a FRED API key.

    The public graph CSV endpoint is used.  Data are cached locally.  If the
    online request fails but a cache exists, stale cached data are returned and
    the source string contains the error.
    """
    cache = _cache_path(spec.fred_id)
    if not force and _cache_is_fresh(cache):
        return _read_cache(cache), "cache"

    try:
        response = requests.get(
            FRED_GRAPH_CSV,
            params={"id": spec.fred_id, "cosd": start},
            headers={"User-Agent": "cot-fundamental-research/1.0"},
            timeout=45,
        )
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
        if raw.shape[1] < 2:
            raise ValueError(f"Unerwartetes FRED-Format für {spec.fred_id}.")
        date_col = raw.columns[0]
        value_col = spec.fred_id if spec.fred_id in raw.columns else raw.columns[1]
        out = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "value": pd.to_numeric(raw[value_col].replace(".", pd.NA), errors="coerce"),
        }).dropna(subset=["date", "value"]).sort_values("date")
        if out.empty:
            raise ValueError(f"FRED lieferte keine verwertbaren Daten für {spec.fred_id}.")
        out.to_parquet(cache, index=False)
        return out, "online"
    except Exception as exc:
        if cache.exists():
            return _read_cache(cache), f"stale-cache: {exc}"
        raise RuntimeError(f"Fundamentaldaten {spec.label} ({spec.fred_id}) konnten nicht geladen werden: {exc}") from exc
