from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Asset:
    name: str
    category: str
    exchange: str
    cftc_report: str
    cftc_market_name: str
    cftc_code: str
    price_symbol: str

@dataclass(frozen=True)
class Category:
    name: str
    icon: str
    order: int


def _load_raw() -> dict[str, Any]:
    with (ROOT / "assets" / "registry.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_assets() -> dict[str, Asset]:
    raw = _load_raw()["assets"]
    return {name: Asset(name=name, **cfg) for name, cfg in raw.items()}


def load_categories() -> dict[str, Category]:
    raw = _load_raw()["categories"]
    return {name: Category(name=name, **cfg) for name, cfg in raw.items()}


def grouped_assets() -> dict[str, list[Asset]]:
    categories = load_categories()
    assets = load_assets().values()
    result: dict[str, list[Asset]] = {}
    for category in sorted(categories.values(), key=lambda item: item.order):
        result[category.name] = sorted(
            [asset for asset in assets if asset.category == category.name],
            key=lambda item: item.name,
        )
    return result
