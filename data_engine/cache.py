from __future__ import annotations
import re
from pathlib import Path
from .settings import DATA_DIR

def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

def cot_cache(asset_name: str) -> Path:
    return DATA_DIR / f"cot_{safe_name(asset_name)}.parquet"

def price_cache(symbol: str) -> Path:
    return DATA_DIR / f"price_{safe_name(symbol)}.parquet"
