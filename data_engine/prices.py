from __future__ import annotations
import pandas as pd
from assets.registry import Asset
from .cache import price_cache
from .settings import MAX_PRICE_AGE_DAYS


def fetch_prices(asset: Asset, force: bool = False) -> tuple[pd.DataFrame, str]:
    cache = price_cache(asset.price_symbol)
    if cache.exists() and not force:
        cached = pd.read_parquet(cache)
        latest = pd.to_datetime(cached["date"], errors="coerce").max()
        if pd.notna(latest) and latest >= pd.Timestamp.today().normalize() - pd.Timedelta(days=MAX_PRICE_AGE_DAYS):
            return cached, "cache"
    try:
        import yfinance as yf
        raw = yf.download(asset.price_symbol, start="1980-01-01", progress=False, auto_adjust=False, threads=False)
        if raw.empty:
            raise ValueError(f"Keine Preisdaten für {asset.price_symbol} erhalten.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index().rename(columns=str.lower)
        raw = raw.rename(columns={raw.columns[0]: "date", "adj close": "adj_close"})
        raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_localize(None)
        keep = [column for column in ["date", "open", "high", "low", "close", "adj_close", "volume"] if column in raw]
        out = raw[keep].dropna(subset=["date", "close"]).sort_values("date")
        out.to_parquet(cache, index=False)
        return out, "online"
    except Exception as exc:
        if cache.exists():
            return pd.read_parquet(cache), f"stale-cache: {exc}"
        raise
