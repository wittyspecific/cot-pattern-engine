from __future__ import annotations
from typing import Callable
import pandas as pd
from assets.registry import Asset
from data_engine.service import load_market_data
from pattern_engine.analysis import NET_COLS, OI_RATIO_COLS, net_similar_weeks, prepare_dataset, timing_analysis, validate_open_interest_context


def scan_assets(
    assets: dict[str, Asset],
    force: bool = False,
    n_neighbors: int = 30,
    move_threshold: float = 0.01,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(assets)
    for index, asset in enumerate(assets.values(), 1):
        if progress:
            progress(index, total, asset.name)
        try:
            market = load_market_data(asset, force=force)
            if not market.usable:
                raise ValueError(f"COT: {market.cot_quality.message}; Preis: {market.price_quality.message}")
            data = prepare_dataset(market.cot, market.prices, [8])
            valid = data.dropna(subset=NET_COLS + OI_RATIO_COLS)
            if len(valid) < 60:
                raise ValueError("Zu wenige historische Beobachtungen")
            matches = net_similar_weeks(valid, n_neighbors, 8)
            oi_matches = validate_open_interest_context(matches, valid, 1.0)
            result = timing_analysis(matches, market.prices, 8, move_threshold)
            oi_result = timing_analysis(oi_matches, market.prices, 8, move_threshold)
            bias = result["bias"]
            hit_rate = float(result["positive_rate"] if bias == "bullish" else result["negative_rate"])
            rows.append({
                "Asset": asset.name,
                "Kategorie": asset.category,
                "Bias": "Bullish" if bias == "bullish" else "Bearish" if bias == "bearish" else "Neutral",
                "Trefferquote": hit_rate,
                "Fälle": int(result["count"]),
                "OI bestätigt": int(oi_result["count"]) >= 8 and oi_result["bias"] == bias and bias != "neutral",
                "COT-Stichtag": market.cot_quality.latest_date.strftime("%Y-%m-%d") if market.cot_quality.latest_date is not None else "",
                "Fehler": "",
            })
        except Exception as exc:
            rows.append({"Asset": asset.name, "Kategorie": asset.category, "Bias": "Nicht verfügbar", "Trefferquote": None, "Fälle": 0, "OI bestätigt": False, "COT-Stichtag": "", "Fehler": str(exc)})
    return rows


def top_five(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    usable = frame[frame["Bias"].isin(["Bullish", "Bearish"]) & frame["Trefferquote"].notna() & (frame["Fälle"] >= 15)].copy()
    if usable.empty:
        return usable
    usable["OI-Rang"] = usable["OI bestätigt"].astype(int)
    return usable.sort_values(["Trefferquote", "OI-Rang", "Fälle"], ascending=[False, False, False]).head(5).drop(columns=["OI-Rang"])
