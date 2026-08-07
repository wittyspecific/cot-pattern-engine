from __future__ import annotations

from assets.registry import Asset
from data_engine.fundamentals import fetch_fred_series

from .analysis import FundamentalResearchResult, analyze_fundamental_regime, transformed_series
from .config import SERIES, coverage_for_asset, profile_for_asset, isolated_currency_profile


def load_fundamental_research(
    asset: Asset,
    prices,
    horizon_weeks: int = 8,
    n_neighbors: int = 20,
    force: bool = False,
) -> tuple[FundamentalResearchResult, tuple[str, str]]:
    """Load and analyse fundamentals without loading any COT data.

    Currency futures use only macroeconomic data from their own home economy.
    No US-relative spread and no COT feature enters the analog search.
    """
    isolated = isolated_currency_profile(asset) if asset.category == "Währungen" else None
    if isolated:
        country, specs, weights = isolated
        coverage = ("mittel bis hoch", f"Isoliertes Makroprofil für {country}; keine US-Gegenrechnung und keine COT-Daten.")
    else:
        country = None
        specs = SERIES
        weights = profile_for_asset(asset)
        coverage = coverage_for_asset(asset)

    transformed = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for key in weights:
        spec = specs[key]
        try:
            raw, source = fetch_fred_series(spec, force=force)
            transformed[key] = transformed_series(raw, spec)
            sources[key] = source
        except Exception as exc:
            missing.append(f"{spec.label}: {exc}")

    result = analyze_fundamental_regime(
        transformed,
        specs,
        weights,
        prices,
        horizon_weeks=horizon_weeks,
        n_neighbors=n_neighbors,
        sources=sources,
        missing_series=missing,
    )
    if country:
        result.reason = result.reason or f"Isolierte Makroanalyse für {country}."
    return result, coverage
