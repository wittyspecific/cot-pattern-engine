from __future__ import annotations
from typing import Callable
import pandas as pd
import numpy as np
from assets.registry import Asset
from data_engine.service import load_market_data
from pattern_engine.analysis import NET_COLS, OI_RATIO_COLS, net_similar_weeks, prepare_dataset, timing_analysis, validate_open_interest_context
from pattern_engine.divergence import analyze_noncommercial_divergence
from pattern_engine.clusters import analyze_cluster_timing, analyze_current_cluster
from pattern_engine.extremes import analyze_position_extremes
from pattern_engine.cot_index import analyze_cot_index_pattern
from pattern_engine.net_levels import analyze_pure_net_pattern


def _dominant_range(returns: pd.Series) -> tuple[float | None, float | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    positive = values[values > 0]
    negative = values[values < 0]
    if len(positive) > len(negative):
        selected = positive
    elif len(negative) > len(positive):
        selected = negative
    else:
        return None, None
    if selected.empty:
        return None, None
    return float(selected.quantile(0.25)), float(selected.quantile(0.75))


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
            cot_dates = pd.to_datetime(market.cot.get("date"), errors="coerce").dropna()
            price_dates = pd.to_datetime(market.prices.get("date"), errors="coerce").dropna()
            data = prepare_dataset(market.cot, market.prices, [4, 8, 12])
            valid = data.dropna(subset=NET_COLS + OI_RATIO_COLS)
            if len(valid) < 60:
                raise ValueError("Zu wenige historische Beobachtungen")

            pure_net_results: dict[int, object] = {}
            pure_net_timings: dict[int, object | None] = {}
            for pure_horizon in (4, 8, 12):
                pure_result = analyze_pure_net_pattern(
                    valid, horizon_weeks=pure_horizon, n_neighbors=n_neighbors
                )
                pure_net_results[pure_horizon] = pure_result
                if pure_result.available and pure_result.dominant_direction != "neutral":
                    pure_result.matches.attrs["current_report_date"] = pd.Timestamp(valid.iloc[-1]["date"])
                    pure_net_timings[pure_horizon] = analyze_cluster_timing(
                        pure_result.matches,
                        market.prices,
                        pure_result.dominant_direction,
                        observation_weeks=pure_horizon,
                    )
                else:
                    pure_net_timings[pure_horizon] = None

            matches = net_similar_weeks(valid, n_neighbors, 8)
            oi_matches = validate_open_interest_context(matches, valid, 1.0)
            result = timing_analysis(matches, market.prices, 8, move_threshold)
            oi_result = timing_analysis(oi_matches, market.prices, 8, move_threshold)
            divergence = analyze_noncommercial_divergence(valid, horizons=(1, 2, 4, 8))
            extremes = analyze_position_extremes(valid, extreme_cutoff=90.0)
            cluster = analyze_current_cluster(valid, horizon_weeks=8)
            cot_index = analyze_cot_index_pattern(valid, horizon_weeks=8)
            if cot_index.available:
                cot_index.matches.attrs["current_report_date"] = pd.Timestamp(valid.iloc[-1]["date"])
                cot_index_timing_bias = "bullish" if (cot_index.median_return or 0) > 0 else "bearish" if (cot_index.median_return or 0) < 0 else cot_index.bias
                cot_index_timing = analyze_cluster_timing(cot_index.matches, market.prices, cot_index_timing_bias, observation_weeks=8)
            else:
                cot_index_timing = None
            if cluster.available:
                cluster.matches.attrs["current_report_date"] = pd.Timestamp(valid.iloc[-1]["date"])
                cluster_timing_bias = "bullish" if (cluster.median_return or 0) > 0 else "bearish" if (cluster.median_return or 0) < 0 else cluster.bias
                cluster_timing = analyze_cluster_timing(cluster.matches, market.prices, cluster_timing_bias, observation_weeks=8)
            else:
                cluster_timing = None
            bias = result["bias"]
            hit_rate = float(result["positive_rate"] if bias == "bullish" else result["negative_rate"])
            extreme_groups = {group.label: group for group in extremes.get("groups", [])}
            commercial_extreme = extreme_groups.get("Commercials")
            noncommercial_extreme = extreme_groups.get("Non-Commercials")
            retail_extreme = extreme_groups.get("Retail / Non-Reportables")
            extreme_distance = max(
                [abs(float(g.combined_percentile) - 50.0) * 2.0 for g in extreme_groups.values()] or [0.0]
            )
            horizon_rows = divergence.get("horizons", [])
            strongest_flow = max(horizon_rows, key=lambda x: float(x.get("score", 0.0)), default={})
            flow_score = float(strongest_flow.get("score", 0.0))
            flow_label_map = {
                "long_building": "Long Building",
                "short_building": "Short Building",
                "none": "Kein dominanter Aufbau",
            }
            flow_label = flow_label_map.get(str(strongest_flow.get("dominant_flow", "none")), "Kein dominanter Aufbau")
            cluster_returns = cluster.matches.get("return_8w", pd.Series(dtype=float)).dropna().astype(float) if cluster.available else pd.Series(dtype=float)
            cluster_up = int((cluster_returns > 0).sum())
            cluster_down = int((cluster_returns < 0).sum())
            cluster_direction_rate = max(cluster_up, cluster_down) / len(cluster_returns) if len(cluster_returns) else None
            cluster_range_low, cluster_range_high = _dominant_range(cluster_returns)
            index_returns = cot_index.matches.get("return_8w", pd.Series(dtype=float)).dropna().astype(float) if cot_index.available else pd.Series(dtype=float)
            index_up = int((index_returns > 0).sum())
            index_down = int((index_returns < 0).sum())
            index_direction_rate = max(index_up, index_down) / len(index_returns) if len(index_returns) else None
            index_range_low, index_range_high = _dominant_range(index_returns)

            pure_fields: dict[str, object] = {}
            for pure_horizon in (4, 8, 12):
                pure_result = pure_net_results[pure_horizon]
                pure_timing = pure_net_timings[pure_horizon]
                prefix = f"Reines Netto {pure_horizon}W"
                pure_fields.update({
                    f"{prefix} verfügbar": bool(pure_result.available),
                    f"{prefix} Fälle": int(pure_result.sample_size),
                    f"{prefix} Gestiegen": int(pure_result.up_count),
                    f"{prefix} Gefallen": int(pure_result.down_count),
                    f"{prefix} Richtung": (
                        "Bullish" if pure_result.dominant_direction == "bullish"
                        else "Bearish" if pure_result.dominant_direction == "bearish"
                        else "Neutral"
                    ),
                    f"{prefix} Trefferquote": pure_result.hit_rate,
                    f"{prefix} Median": pure_result.median_return,
                    f"{prefix} Range unten": pure_result.range_low,
                    f"{prefix} Range oben": pure_result.range_high,
                    f"{prefix} Beginnfenster": (
                        f"{pure_timing.onset_window_start}–{pure_timing.onset_window_end}"
                        if pure_timing and pure_timing.available and pure_timing.onset_window_start is not None
                        else None
                    ),
                })

            rows.append({
                **pure_fields,
                "Asset": asset.name,
                "Kategorie": asset.category,
                "Bias": "Bullish" if bias == "bullish" else "Bearish" if bias == "bearish" else "Neutral",
                "Trefferquote": hit_rate,
                "Fälle": int(result["count"]),
                "OI bestätigt": int(oi_result["count"]) >= 8 and oi_result["bias"] == bias and bias != "neutral",
                "NC-Divergenz": "Bullisch" if divergence["signal"] == "bullish" else ("Bearisch" if divergence["signal"] == "bearish" else "–"),
                "NC-Divergenz 1W": next(("Bullisch" if x["signal"] == "bullish" else "Bearisch" if x["signal"] == "bearish" else "–" for x in divergence.get("horizons", []) if int(x["weeks"]) == 1), "–"),
                "NC-Divergenz 2W": next(("Bullisch" if x["signal"] == "bullish" else "Bearisch" if x["signal"] == "bearish" else "–" for x in divergence.get("horizons", []) if int(x["weeks"]) == 2), "–"),
                "NC-Divergenz 4W": next(("Bullisch" if x["signal"] == "bullish" else "Bearisch" if x["signal"] == "bearish" else "–" for x in divergence.get("horizons", []) if int(x["weeks"]) == 4), "–"),
                "NC-Divergenz 8W": next(("Bullisch" if x["signal"] == "bullish" else "Bearisch" if x["signal"] == "bearish" else "–" for x in divergence.get("horizons", []) if int(x["weeks"]) == 8), "–"),
                "Extrem verfügbar": bool(extremes.get("available", False)),
                "Extrem Signal": str(extremes.get("title", "Nicht verfügbar")),
                "Extrem Richtung": str(extremes.get("absolute_signal", "neutral")),
                "Extrem + OI": bool(extremes.get("confirmed_by_oi", False)),
                "Extremstärke": extreme_distance,
                "Commercial Perzentil": float(commercial_extreme.combined_percentile) if commercial_extreme else None,
                "NonCommercial Perzentil": float(noncommercial_extreme.combined_percentile) if noncommercial_extreme else None,
                "Retail Perzentil": float(retail_extreme.combined_percentile) if retail_extreme else None,
                "Flow verfügbar": bool(divergence.get("available", False)),
                "Flow Richtung": "Bullisch" if divergence.get("signal") == "bullish" else "Bearisch" if divergence.get("signal") == "bearish" else "Neutral",
                "Flow Stärke": str(divergence.get("strength", "keine")),
                "Flow Score": flow_score,
                "Dominanter Flow": flow_label,
                "Flow Horizont": int(strongest_flow.get("weeks", 0)) if strongest_flow else 0,
                "Flow Dominanz": float(strongest_flow.get("dominance", 0.0)) if strongest_flow else 0.0,
                "Cluster verfügbar": bool(cluster.available),
                "Cluster Bias": "Bullish" if cluster.bias == "bullish" else "Bearish" if cluster.bias == "bearish" else "Neutral",
                "Cluster Trefferquote": cluster.hit_rate,
                "Cluster Konfidenz unten": cluster.confidence_low,
                "Cluster Fälle": int(cluster.sample_size),
                "Cluster Gestiegen": cluster_up,
                "Cluster Gefallen": cluster_down,
                "Cluster Richtungsanteil": cluster_direction_rate,
                "Cluster Median": cluster.median_return,
                "Cluster Q25": cluster.q25_return,
                "Cluster Q75": cluster.q75_return,
                "Cluster Range unten": cluster_range_low,
                "Cluster Range oben": cluster_range_high,
                "Cluster Qualität": cluster.quality_score,
                "Cluster Silhouette": cluster.silhouette,
                "Cluster Beginn": cluster_timing.onset_day if cluster_timing and cluster_timing.available else None,
                "Cluster Beginnfenster": (f"{cluster_timing.onset_window_start}–{cluster_timing.onset_window_end}" if cluster_timing and cluster_timing.available and cluster_timing.onset_window_start is not None else None),
                "Cluster Maximum Tag": cluster_timing.peak_day if cluster_timing and cluster_timing.available else None,
                "Cluster Timing Status": cluster_timing.status_text if cluster_timing and cluster_timing.available else "",
                "Index verfügbar": bool(cot_index.available),
                "Commercial COT Index": cot_index.commercial_index,
                "Commercial Netto-Perzentil": cot_index.commercial_net_percentile,
                "Retail COT Index": cot_index.retail_index,
                "Retail Netto-Perzentil": cot_index.retail_net_percentile,
                "Index doppelt bestätigt": bool(cot_index.commercial_validated and cot_index.retail_validated),
                "Index Bias": "Bullish" if cot_index.bias == "bullish" else "Bearish" if cot_index.bias == "bearish" else "Neutral",
                "Index Trefferquote": cot_index.hit_rate,
                "Index Konfidenz unten": cot_index.confidence_low,
                "Index Fälle": int(cot_index.sample_size),
                "Index Gestiegen": index_up,
                "Index Gefallen": index_down,
                "Index Richtungsanteil": index_direction_rate,
                "Index Median": cot_index.median_return,
                "Index Range unten": index_range_low,
                "Index Range oben": index_range_high,
                "Index Qualität": cot_index.quality_score,
                "Index Beginnfenster": (f"Tag {cot_index_timing.onset_window_start}–{cot_index_timing.onset_window_end}" if cot_index_timing and cot_index_timing.available and cot_index_timing.onset_window_start is not None else "–"),
                "COT-Start": cot_dates.min().strftime("%Y-%m-%d") if not cot_dates.empty else "",
                "COT-Stichtag": market.cot_quality.latest_date.strftime("%Y-%m-%d") if market.cot_quality.latest_date is not None else "",
                "Preis-Start": price_dates.min().strftime("%Y-%m-%d") if not price_dates.empty else "",
                "Preis-Stichtag": market.price_quality.latest_date.strftime("%Y-%m-%d") if market.price_quality.latest_date is not None else "",
                "Fehler": "",
            })
        except Exception as exc:
            pure_error_fields = {}
            for pure_horizon in (4, 8, 12):
                prefix = f"Reines Netto {pure_horizon}W"
                pure_error_fields.update({
                    f"{prefix} verfügbar": False, f"{prefix} Fälle": 0,
                    f"{prefix} Gestiegen": 0, f"{prefix} Gefallen": 0,
                    f"{prefix} Richtung": "Nicht verfügbar", f"{prefix} Trefferquote": None,
                    f"{prefix} Median": None, f"{prefix} Range unten": None,
                    f"{prefix} Range oben": None, f"{prefix} Beginnfenster": None,
                })
            rows.append({**pure_error_fields, "Asset": asset.name, "Kategorie": asset.category, "Bias": "Nicht verfügbar", "Trefferquote": None, "Fälle": 0, "OI bestätigt": False, "NC-Divergenz": "–", "NC-Divergenz 1W": "–", "NC-Divergenz 2W": "–", "NC-Divergenz 4W": "–", "NC-Divergenz 8W": "–", "Extrem verfügbar": False, "Extrem Signal": "Nicht verfügbar", "Extrem Richtung": "neutral", "Extrem + OI": False, "Extremstärke": 0.0, "Commercial Perzentil": None, "NonCommercial Perzentil": None, "Retail Perzentil": None, "Flow verfügbar": False, "Flow Richtung": "Neutral", "Flow Stärke": "keine", "Flow Score": 0.0, "Dominanter Flow": "Nicht verfügbar", "Flow Horizont": 0, "Flow Dominanz": 0.0, "Cluster verfügbar": False, "Cluster Bias": "Nicht verfügbar", "Cluster Trefferquote": None, "Cluster Konfidenz unten": None, "Cluster Fälle": 0, "Cluster Gestiegen": 0, "Cluster Gefallen": 0, "Cluster Richtungsanteil": None, "Cluster Median": None, "Cluster Q25": None, "Cluster Q75": None, "Cluster Range unten": None, "Cluster Range oben": None, "Cluster Qualität": None, "Cluster Silhouette": None, "Cluster Beginn": None, "Cluster Beginnfenster": None, "Cluster Maximum Tag": None, "Cluster Timing Status": "", "Index verfügbar": False, "Commercial COT Index": None, "Commercial Netto-Perzentil": None, "Retail COT Index": None, "Retail Netto-Perzentil": None, "Index doppelt bestätigt": False, "Index Bias": "Nicht verfügbar", "Index Trefferquote": None, "Index Konfidenz unten": None, "Index Fälle": 0, "Index Gestiegen": 0, "Index Gefallen": 0, "Index Richtungsanteil": None, "Index Median": None, "Index Range unten": None, "Index Range oben": None, "Index Qualität": None, "Index Beginnfenster": "–", "COT-Start": "", "COT-Stichtag": "", "Preis-Start": "", "Preis-Stichtag": "", "Fehler": str(exc)})
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



def top_five_pure_net_patterns(
    rows: list[dict[str, object]],
    horizon_weeks: int,
    min_hit_rate: float = 0.70,
    min_episodes: int = 25,
    min_median_move: float = 0.02,
) -> pd.DataFrame:
    """Top-5 der reinen absoluten Nettopositionsmuster für einen Horizont."""
    if horizon_weeks not in (4, 8, 12):
        raise ValueError("Unterstützte Horizonte sind 4, 8 und 12 Wochen.")
    frame = pd.DataFrame(rows)
    prefix = f"Reines Netto {horizon_weeks}W"
    required = [
        f"{prefix} verfügbar", f"{prefix} Fälle", f"{prefix} Trefferquote",
        f"{prefix} Median", f"{prefix} Range unten", f"{prefix} Range oben",
    ]
    if frame.empty or any(column not in frame.columns for column in required):
        return pd.DataFrame()
    valid = frame.loc[
        frame[f"{prefix} verfügbar"].fillna(False)
        & frame[f"{prefix} Trefferquote"].notna()
        & frame[f"{prefix} Median"].notna()
        & (frame[f"{prefix} Trefferquote"] >= min_hit_rate)
        & (frame[f"{prefix} Fälle"] >= min_episodes)
        & (frame[f"{prefix} Median"].abs() >= min_median_move)
    ].copy()
    if valid.empty:
        return valid
    valid["Historischer Edge"] = (
        valid[f"{prefix} Median"].abs()
        * valid[f"{prefix} Trefferquote"]
        * np.log1p(valid[f"{prefix} Fälle"])
    )
    columns = [
        "Asset", "Kategorie", f"{prefix} Richtung", f"{prefix} Fälle",
        f"{prefix} Gestiegen", f"{prefix} Gefallen", f"{prefix} Trefferquote",
        f"{prefix} Median", f"{prefix} Range unten", f"{prefix} Range oben",
        f"{prefix} Beginnfenster", "COT-Stichtag",
    ]
    return valid.sort_values(
        ["Historischer Edge", f"{prefix} Fälle"], ascending=[False, False]
    ).head(5)[columns]


def top_five_clusters(
    rows: list[dict[str, object]],
    min_hit_rate: float = 0.70,
    min_episodes: int = 25,
    min_median_move: float = 0.02,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or "Cluster verfügbar" not in frame.columns:
        return pd.DataFrame()
    valid = frame.loc[
        frame["Cluster verfügbar"].fillna(False)
        & frame["Cluster Median"].notna()
        & frame["Cluster Richtungsanteil"].notna()
        & (frame["Cluster Richtungsanteil"] >= min_hit_rate)
        & (frame["Cluster Fälle"] >= min_episodes)
        & (frame["Cluster Median"].abs() >= min_median_move)
    ].copy()
    if valid.empty:
        return valid
    valid["Historischer Edge"] = (
        valid["Cluster Median"].abs()
        * valid["Cluster Richtungsanteil"]
        * np.log1p(valid["Cluster Fälle"])
    )
    columns = [
        "Asset", "Kategorie", "Cluster Fälle", "Cluster Gestiegen", "Cluster Gefallen",
        "Cluster Richtungsanteil", "Cluster Median", "Cluster Range unten", "Cluster Range oben", "Cluster Beginnfenster", "COT-Stichtag"
    ]
    return valid.sort_values(
        ["Historischer Edge", "Cluster Fälle"], ascending=[False, False]
    ).head(5)[columns]


def top_five_extremes(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or "Extrem verfügbar" not in frame.columns:
        return pd.DataFrame()
    valid = frame[(frame["Fehler"] == "") & frame["Extrem verfügbar"].fillna(False)].copy()
    if valid.empty:
        return valid
    valid["OI Rang"] = valid["Extrem + OI"].astype(int)
    columns = [
        "Asset", "Kategorie", "Extrem Signal", "Extremstärke",
        "Commercial Perzentil", "NonCommercial Perzentil", "Retail Perzentil",
        "Extrem + OI", "COT-Stichtag"
    ]
    return valid.sort_values(["Extremstärke", "OI Rang"], ascending=[False, False]).head(5)[columns]


def top_five_flows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or "Flow verfügbar" not in frame.columns:
        return pd.DataFrame()
    valid = frame[(frame["Fehler"] == "") & frame["Flow verfügbar"].fillna(False) & (frame["Flow Score"] > 0)].copy()
    if valid.empty:
        return valid
    columns = [
        "Asset", "Kategorie", "Flow Richtung", "Dominanter Flow",
        "Flow Stärke", "Flow Score", "Flow Horizont", "Flow Dominanz",
        "NC-Divergenz 1W", "NC-Divergenz 2W", "NC-Divergenz 4W", "NC-Divergenz 8W",
        "COT-Stichtag"
    ]
    return valid.sort_values(["Flow Score", "Flow Dominanz", "Flow Horizont"], ascending=[False, False, False]).head(5)[columns]



# Backward compatibility for previous deployments/tests. The new UI does not use this ranking.
def top_five_transitions(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or "Transition Phase" not in frame.columns:
        return pd.DataFrame()
    usable = frame[(frame.get("Fehler", "") == "") & (frame["Transition Phase"] >= 1)].copy()
    if usable.empty:
        return usable
    usable["Bestätigungen"] = (
        usable.get("Extrem + OI", False).astype(int)
        + usable.get("Retail unterstützt", False).astype(int)
        + usable.get("NC-Konvergenz", False).astype(int)
        + usable.get("Preisbestätigung", False).astype(int)
    )
    columns = [
        "Asset", "Kategorie", "Transition Phase", "Transition Score",
        "Transition Richtung", "Transition Status", "Extrem + OI",
        "Retail unterstützt", "NC-Konvergenz", "Preisbestätigung",
        "NC-Divergenz 1W", "NC-Divergenz 2W", "NC-Divergenz 4W", "NC-Divergenz 8W",
        "COT-Stichtag",
    ]
    return usable.sort_values(
        ["Transition Phase", "Transition Score", "Bestätigungen"],
        ascending=[False, False, False],
    ).head(5)[columns]


def top_five_cot_index(
    rows: list[dict[str, object]],
    min_hit_rate: float = 0.70,
    min_episodes: int = 25,
    min_median_move: float = 0.02,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or "Index verfügbar" not in frame.columns:
        return pd.DataFrame()
    valid = frame.loc[
        frame["Index verfügbar"].fillna(False)
        & frame["Index Median"].notna()
        & frame["Index Richtungsanteil"].notna()
        & (frame["Index Richtungsanteil"] >= min_hit_rate)
        & (frame["Index Fälle"] >= min_episodes)
        & (frame["Index Median"].abs() >= min_median_move)
    ].copy()
    if valid.empty:
        return valid
    valid["Historischer Edge"] = (
        valid["Index Median"].abs()
        * valid["Index Richtungsanteil"]
        * np.log1p(valid["Index Fälle"])
    )
    columns = [
        "Asset", "Kategorie", "Commercial COT Index", "Commercial Netto-Perzentil",
        "Retail COT Index", "Retail Netto-Perzentil", "Index doppelt bestätigt",
        "Index Fälle", "Index Gestiegen", "Index Gefallen", "Index Richtungsanteil",
        "Index Median", "Index Range unten", "Index Range oben", "Index Beginnfenster", "COT-Stichtag"
    ]
    return valid.sort_values(
        ["Historischer Edge", "Index Fälle"], ascending=[False, False]
    ).head(5)[columns]

