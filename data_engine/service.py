from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from assets.registry import Asset
from .cftc import fetch_cot
from .prices import fetch_prices
from .quality import DataQuality, assess_freshness
from .settings import MAX_COT_AGE_DAYS, MAX_PRICE_AGE_DAYS

@dataclass
class MarketData:
    asset: Asset
    cot: pd.DataFrame
    prices: pd.DataFrame
    cot_quality: DataQuality
    price_quality: DataQuality
    cot_source: str
    price_source: str

    @property
    def usable(self) -> bool:
        return self.cot_quality.is_current and self.price_quality.is_current


def load_market_data(asset: Asset, force: bool = False) -> MarketData:
    cot, cot_source = fetch_cot(asset, force=force)
    prices, price_source = fetch_prices(asset, force=force)
    return MarketData(
        asset=asset,
        cot=cot,
        prices=prices,
        cot_quality=assess_freshness(cot, "date", MAX_COT_AGE_DAYS),
        price_quality=assess_freshness(prices, "date", MAX_PRICE_AGE_DAYS),
        cot_source=cot_source,
        price_source=price_source,
    )
