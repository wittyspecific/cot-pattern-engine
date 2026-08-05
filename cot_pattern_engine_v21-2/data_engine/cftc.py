from __future__ import annotations
from typing import Iterable
import pandas as pd
import requests
from assets.registry import Asset
from .cache import cot_cache
from .settings import CFTC_BASE, CFTC_DATASETS

ALIASES = {
    "date": ["report_date_as_yyyy_mm_dd", "report_date_as_mm_dd_yyyy", "as_of_date_in_form_yyyy_mm_dd"],
    "name": ["market_and_exchange_names", "contract_market_name"],
    "code": ["cftc_contract_market_code", "cftc_contract_market_code_quotes"],
    "oi": ["open_interest_all", "open_interest"],
    "noncomm_long": ["noncomm_positions_long_all", "noncommercial_positions_long_all"],
    "noncomm_short": ["noncomm_positions_short_all", "noncommercial_positions_short_all"],
    "comm_long": ["comm_positions_long_all", "commercial_positions_long_all"],
    "comm_short": ["comm_positions_short_all", "commercial_positions_short_all"],
    "nonrep_long": ["nonrept_positions_long_all", "nonreportable_positions_long_all"],
    "nonrep_short": ["nonrept_positions_short_all", "nonreportable_positions_short_all"],
}

def _first(columns: Iterable[str], aliases: list[str], required: bool = True) -> str | None:
    lower = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    if required:
        raise KeyError(f"Spalte fehlt: {aliases}")
    return None


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    mapping = {key: _first(raw.columns, aliases, required=(key != "code")) for key, aliases in ALIASES.items()}
    out = pd.DataFrame({key: raw[column] for key, column in mapping.items() if column is not None})
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for column in [c for c in out.columns if c not in ("date", "name", "code")]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["commercial_net"] = out["comm_long"] - out["comm_short"]
    out["noncommercial_net"] = out["noncomm_long"] - out["noncomm_short"]
    out["nonreportable_net"] = out["nonrep_long"] - out["nonrep_short"]
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def fetch_cot(asset: Asset, force: bool = False) -> tuple[pd.DataFrame, str]:
    cache = cot_cache(asset.name)
    if cache.exists() and not force:
        return pd.read_parquet(cache), "cache"

    dataset = CFTC_DATASETS[asset.cftc_report]
    endpoint = f"{CFTC_BASE}/{dataset}.json"
    # Primär wird der eindeutige CFTC-Code verwendet. Die Marktbezeichnung dient nur als
    # kontrollierter Fallback, falls ein Datensatz den Code nicht erwartungsgemäß akzeptiert.
    code = asset.cftc_code.replace("'", "''")
    params = {
        "$limit": 50000,
        "$order": "report_date_as_yyyy_mm_dd asc",
        "$where": f"cftc_contract_market_code='{code}'",
    }
    try:
        response = requests.get(endpoint, params=params, timeout=45)
        response.raise_for_status()
        raw = pd.DataFrame(response.json())
        if raw.empty:
            escaped = asset.cftc_market_name.upper().replace("'", "''")
            params["$where"] = f"upper(market_and_exchange_names)='{escaped}'"
            response = requests.get(endpoint, params=params, timeout=45)
            response.raise_for_status()
            raw = pd.DataFrame(response.json())
        if raw.empty:
            raise ValueError(f"Kein CFTC-Datensatz für {asset.name} gefunden.")
        out = _normalize(raw)
        out.to_parquet(cache, index=False)
        return out, "online"
    except Exception as exc:
        if cache.exists():
            return pd.read_parquet(cache), f"stale-cache: {exc}"
        raise
