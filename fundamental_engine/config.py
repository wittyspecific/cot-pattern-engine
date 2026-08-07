from __future__ import annotations

from dataclasses import dataclass

from assets.registry import Asset


@dataclass(frozen=True)
class FundamentalSeriesSpec:
    key: str
    fred_id: str
    label: str
    group: str
    frequency: str
    transform: str
    release_lag_days: int
    display_unit: str
    description: str


SERIES: dict[str, FundamentalSeriesSpec] = {
    "cpi": FundamentalSeriesSpec(
        "cpi", "CPIAUCSL", "CPI-Inflation", "Inflation", "monthly", "yoy_pct", 14, "% YoY",
        "Veränderung des US-Verbraucherpreisindex gegenüber dem Vorjahr.",
    ),
    "core_cpi": FundamentalSeriesSpec(
        "core_cpi", "CPILFESL", "Kern-CPI", "Inflation", "monthly", "yoy_pct", 14, "% YoY",
        "US-Verbraucherpreise ohne Lebensmittel und Energie gegenüber dem Vorjahr.",
    ),
    "core_pce": FundamentalSeriesSpec(
        "core_pce", "PCEPILFE", "Kern-PCE", "Inflation", "monthly", "yoy_pct", 30, "% YoY",
        "Kernrate des von der Federal Reserve bevorzugten PCE-Preisindex.",
    ),
    "gdp": FundamentalSeriesSpec(
        "gdp", "GDPC1", "Reales GDP", "Wachstum", "quarterly", "yoy_pct", 30, "% YoY",
        "Wachstum des realen US-Bruttoinlandsprodukts gegenüber dem Vorjahr.",
    ),
    "industrial_production": FundamentalSeriesSpec(
        "industrial_production", "INDPRO", "Industrieproduktion", "Wachstum", "monthly", "yoy_pct", 18, "% YoY",
        "Veränderung der US-Industrieproduktion gegenüber dem Vorjahr.",
    ),
    "retail_sales": FundamentalSeriesSpec(
        "retail_sales", "RSAFS", "Einzelhandelsumsätze", "Wachstum", "monthly", "yoy_pct", 16, "% YoY",
        "Veränderung der US-Einzelhandelsumsätze gegenüber dem Vorjahr.",
    ),
    "unemployment": FundamentalSeriesSpec(
        "unemployment", "UNRATE", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 7, "%",
        "Saisonbereinigte US-Arbeitslosenquote.",
    ),
    "nfp": FundamentalSeriesSpec(
        "nfp", "PAYEMS", "NFP / Payroll-Veränderung", "Arbeitsmarkt", "monthly", "diff", 7, "Tsd.",
        "Monatliche Veränderung der gesamten US-Nonfarm-Payroll-Beschäftigung.",
    ),
    "initial_claims": FundamentalSeriesSpec(
        "initial_claims", "ICSA", "Erstanträge Arbeitslosenhilfe", "Arbeitsmarkt", "weekly", "rolling4", 2, "4W-Mittel",
        "Vier-Wochen-Mittel der wöchentlichen Erstanträge auf Arbeitslosenhilfe.",
    ),
    "fed_funds": FundamentalSeriesSpec(
        "fed_funds", "FEDFUNDS", "Federal Funds Rate", "Geldpolitik", "monthly", "level", 5, "%",
        "Monatsdurchschnitt der effektiven Federal Funds Rate.",
    ),
    "yield_2y": FundamentalSeriesSpec(
        "yield_2y", "DGS2", "US-Rendite 2 Jahre", "Geldpolitik", "daily", "level", 0, "%",
        "Rendite zweijähriger US-Staatsanleihen.",
    ),
    "yield_10y": FundamentalSeriesSpec(
        "yield_10y", "DGS10", "US-Rendite 10 Jahre", "Geldpolitik", "daily", "level", 0, "%",
        "Rendite zehnjähriger US-Staatsanleihen.",
    ),
    "real_yield_10y": FundamentalSeriesSpec(
        "real_yield_10y", "DFII10", "Reale US-Rendite 10 Jahre", "Geldpolitik", "daily", "level", 0, "%",
        "Inflationsindexierte reale Rendite zehnjähriger US-Staatsanleihen.",
    ),
    "breakeven_10y": FundamentalSeriesSpec(
        "breakeven_10y", "T10YIE", "Inflationserwartung 10 Jahre", "Inflation", "daily", "level", 0, "%",
        "Zehnjährige US-Breakeven-Inflationsrate.",
    ),
    "dollar": FundamentalSeriesSpec(
        "dollar", "DTWEXBGS", "Breiter US-Dollar-Index", "Finanzbedingungen", "daily", "yoy_pct", 0, "% YoY",
        "Veränderung des handelsgewichteten breiten US-Dollar-Index gegenüber dem Vorjahr.",
    ),
    "m2": FundamentalSeriesSpec(
        "m2", "M2SL", "US-Geldmenge M2", "Liquidität", "monthly", "yoy_pct", 30, "% YoY",
        "Veränderung der US-Geldmenge M2 gegenüber dem Vorjahr.",
    ),
    "financial_stress": FundamentalSeriesSpec(
        "financial_stress", "STLFSI4", "US-Finanzstress", "Finanzbedingungen", "weekly", "level", 7, "Index",
        "St. Louis Fed Financial Stress Index.",
    ),
}


# A positive weight means that a higher standardized value is historically and
# economically treated as supportive for the FUTURES PRICE.  A negative weight
# means the opposite.  These are transparent research priors, not fitted COT
# weights and not trading recommendations.
CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "Aktienindizes": {
        "core_cpi": -0.65, "core_pce": -0.55, "gdp": 0.80,
        "industrial_production": 0.65, "retail_sales": 0.45,
        "unemployment": -0.60, "nfp": 0.45, "fed_funds": -0.70,
        "real_yield_10y": -0.85, "dollar": -0.30, "m2": 0.55,
        "financial_stress": -0.85,
    },
    "Anleihen": {
        "cpi": -0.75, "core_pce": -0.85, "gdp": -0.65,
        "industrial_production": -0.45, "unemployment": 0.70,
        "nfp": -0.55, "initial_claims": 0.45, "fed_funds": -0.80,
        "yield_2y": -0.65, "yield_10y": -0.95,
        "breakeven_10y": -0.75, "financial_stress": 0.45,
    },
    "Energie": {
        "cpi": 0.25, "gdp": 0.75, "industrial_production": 0.90,
        "retail_sales": 0.35, "unemployment": -0.35, "nfp": 0.35,
        "fed_funds": -0.25, "dollar": -0.80, "m2": 0.25,
        "financial_stress": -0.45,
    },
    "Kryptowährungen": {
        "core_cpi": -0.20, "gdp": 0.20, "unemployment": -0.20,
        "fed_funds": -0.85, "real_yield_10y": -0.90,
        "dollar": -0.85, "m2": 1.00, "financial_stress": -0.70,
    },
    "Agrarrohstoffe": {
        "cpi": 0.25, "gdp": 0.20, "industrial_production": 0.15,
        "fed_funds": -0.15, "dollar": -0.70, "m2": 0.20,
        "financial_stress": -0.20,
    },
    "Soft Commodities": {
        "cpi": 0.25, "gdp": 0.20, "fed_funds": -0.15,
        "dollar": -0.75, "m2": 0.20, "financial_stress": -0.20,
    },
    "Vieh": {
        "cpi": 0.20, "gdp": 0.25, "retail_sales": 0.25,
        "unemployment": -0.20, "dollar": -0.45,
        "financial_stress": -0.25,
    },
}


METAL_WEIGHTS: dict[str, dict[str, float]] = {
    "Gold": {
        "cpi": 0.25, "core_pce": 0.20, "gdp": -0.35,
        "unemployment": 0.35, "nfp": -0.25, "fed_funds": -0.75,
        "real_yield_10y": -1.00, "breakeven_10y": 0.55,
        "dollar": -1.00, "m2": 0.65, "financial_stress": 0.55,
    },
    "Silber": {
        "cpi": 0.20, "gdp": 0.35, "industrial_production": 0.65,
        "unemployment": -0.20, "fed_funds": -0.55,
        "real_yield_10y": -0.75, "dollar": -0.90,
        "m2": 0.50, "financial_stress": 0.10,
    },
    "Kupfer": {
        "gdp": 0.85, "industrial_production": 1.00,
        "retail_sales": 0.30, "unemployment": -0.35, "nfp": 0.25,
        "fed_funds": -0.30, "real_yield_10y": -0.25,
        "dollar": -0.85, "m2": 0.30, "financial_stress": -0.55,
    },
    "Platin": {
        "gdp": 0.55, "industrial_production": 0.85,
        "unemployment": -0.30, "fed_funds": -0.30,
        "real_yield_10y": -0.30, "dollar": -0.75,
        "m2": 0.25, "financial_stress": -0.35,
    },
    "Palladium": {
        "gdp": 0.55, "industrial_production": 0.90,
        "unemployment": -0.30, "fed_funds": -0.25,
        "dollar": -0.75, "financial_stress": -0.40,
    },
}


USD_WEIGHTS = {
    "core_cpi": 0.20, "gdp": 0.65, "industrial_production": 0.45,
    "unemployment": -0.60, "nfp": 0.50, "initial_claims": -0.35,
    "fed_funds": 0.85, "yield_2y": 1.00, "yield_10y": 0.45,
    "financial_stress": 0.25,
}


COVERAGE_BY_CATEGORY = {
    "Aktienindizes": ("hoch", "US-Makro, Zinsen, Liquidität und Finanzstress sind direkt abgedeckt."),
    "Anleihen": ("hoch", "Inflation, Wachstum, Arbeitsmarkt und Zinsstruktur sind direkt abgedeckt."),
    "Edelmetalle": ("mittel bis hoch", "Zinsen, Realrenditen, Dollar, Inflation und Wachstum sind abgedeckt; physische Nachfrage ist noch nicht enthalten."),
    "Kryptowährungen": ("mittel", "Liquidität, Dollar, Realzinsen und Finanzstress sind abgedeckt; On-Chain-Daten fehlen noch."),
    "Energie": ("mittel", "US-Makro und Dollar sind abgedeckt; EIA-Lager, Produktion, Raffinerien und OPEC fehlen noch."),
    "Währungen": ("niedrig bis mittel", "Aktuell wird nur die US-Seite analysiert; die Daten der Gegenwährung fehlen noch."),
    "Agrarrohstoffe": ("niedrig", "US-Makro und Dollar sind abgedeckt; USDA, Wetter, Erträge und Lager fehlen noch."),
    "Soft Commodities": ("niedrig", "US-Makro und Dollar sind abgedeckt; Wetter und Produzentenländer-Daten fehlen noch."),
    "Vieh": ("niedrig", "US-Makro ist abgedeckt; Tierbestände, Futtermittel und USDA-Berichte fehlen noch."),
}


def profile_for_asset(asset: Asset) -> dict[str, float]:
    if asset.name in METAL_WEIGHTS:
        return dict(METAL_WEIGHTS[asset.name])
    if asset.category == "Währungen":
        if asset.name == "US Dollar Index":
            return dict(USD_WEIGHTS)
        # Currency futures are quoted as foreign currency per USD.  Until the
        # foreign macro side is added, the US profile is inverted.
        return {key: -weight for key, weight in USD_WEIGHTS.items()}
    weights = CATEGORY_WEIGHTS.get(asset.category)
    if weights:
        adjusted = dict(weights)
        if asset.name == "Nasdaq 100 E-mini":
            adjusted["real_yield_10y"] = -1.00
            adjusted["fed_funds"] = -0.85
        return adjusted
    return {"gdp": 0.30, "core_cpi": -0.20, "fed_funds": -0.30, "dollar": -0.30}


def coverage_for_asset(asset: Asset) -> tuple[str, str]:
    return COVERAGE_BY_CATEGORY.get(asset.category, ("niedrig", "Nur ein kleines US-Makro-Basisset ist verfügbar."))

# ---------------------------------------------------------------------------
# Isolated macro profiles for currency futures
# ---------------------------------------------------------------------------
# The FX laboratory evaluates the home economy of each currency in isolation.
# It does not subtract US data and it does not mix COT information into the
# analog search. Most international series are harmonised OECD series
# distributed through FRED's free CSV endpoint, which keeps one consistent
# loader for the first research version.

CURRENCY_COUNTRY = {
    "US Dollar Index": "USA",
    "Euro FX": "Eurozone",
    "Britisches Pfund": "Großbritannien",
    "Japanischer Yen": "Japan",
    "Schweizer Franken": "Schweiz",
    "Kanadischer Dollar": "Kanada",
    "Australischer Dollar": "Australien",
    "Neuseeland-Dollar": "Neuseeland",
    "Mexikanischer Peso": "Mexiko",
}


def _s(key, fred_id, label, group, frequency, transform, lag, unit, description):
    return FundamentalSeriesSpec(key, fred_id, label, group, frequency, transform, lag, unit, description)


# Series identifiers are intentionally kept in configuration so individual
# sources can be replaced without changing the research engine.
COUNTRY_SERIES: dict[str, dict[str, FundamentalSeriesSpec]] = {
    "USA": {k: SERIES[k] for k in ("cpi", "core_cpi", "gdp", "industrial_production", "retail_sales", "unemployment", "nfp", "fed_funds", "yield_2y", "yield_10y")},
    "Eurozone": {
        "cpi": _s("cpi", "CP0000EZ19M086NEST", "HVPI", "Inflation", "monthly", "yoy_pct", 18, "% YoY", "Harmonisierter Verbraucherpreisindex der Eurozone."),
        "gdp": _s("gdp", "CLVMNACSCAB1GQEA19", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 45, "% YoY", "Reales Bruttoinlandsprodukt der Eurozone."),
        "industrial_production": _s("industrial_production", "PRMNTO01EZQ661S", "Industrieproduktion", "Wachstum", "quarterly", "yoy_pct", 45, "% YoY", "Industrieproduktion der Eurozone."),
        "unemployment": _s("unemployment", "LRHUTTTTEZM156S", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 25, "%", "Harmonisierte Arbeitslosenquote der Eurozone."),
        "policy_rate": _s("policy_rate", "ECBDFR", "EZB-Einlagensatz", "Geldpolitik", "daily", "level", 0, "%", "Einlagensatz der Europäischen Zentralbank."),
        "yield_10y": _s("yield_10y", "IRLTLT01EZM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Harmonisierter langfristiger Zinssatz der Eurozone."),
    },
    "Großbritannien": {
        "cpi": _s("cpi", "GBRCPIALLMINMEI", "CPI", "Inflation", "monthly", "yoy_pct", 18, "% YoY", "Verbraucherpreisindex Großbritanniens."),
        "gdp": _s("gdp", "CLVMNACSCAB1GQGB", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 45, "% YoY", "Reales Bruttoinlandsprodukt Großbritanniens."),
        "industrial_production": _s("industrial_production", "GBRPROINDMISMEI", "Industrieproduktion", "Wachstum", "monthly", "yoy_pct", 45, "% YoY", "Industrieproduktion Großbritanniens."),
        "unemployment": _s("unemployment", "LRHUTTTTGBM156S", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 45, "%", "Harmonisierte Arbeitslosenquote Großbritanniens."),
        "policy_rate": _s("policy_rate", "IUDERB6", "Bank Rate", "Geldpolitik", "daily", "level", 0, "%", "Offizieller Leitzins der Bank of England."),
        "yield_10y": _s("yield_10y", "IRLTLT01GBM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Langfristiger britischer Staatsanleihezins."),
    },
    "Japan": {
        "cpi": _s("cpi", "JPNCPIALLMINMEI", "CPI", "Inflation", "monthly", "yoy_pct", 25, "% YoY", "Verbraucherpreisindex Japans."),
        "gdp": _s("gdp", "JPNRGDPEXP", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 55, "% YoY", "Reales Bruttoinlandsprodukt Japans."),
        "industrial_production": _s("industrial_production", "JPNPROINDMISMEI", "Industrieproduktion", "Wachstum", "monthly", "yoy_pct", 35, "% YoY", "Industrieproduktion Japans."),
        "unemployment": _s("unemployment", "LRUNTTTTJPM156S", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 35, "%", "Arbeitslosenquote Japans."),
        "policy_rate": _s("policy_rate", "IRSTCI01JPM156N", "Kurzfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Kurzfristiger japanischer Zinssatz als geldpolitischer Proxy."),
        "yield_10y": _s("yield_10y", "IRLTLT01JPM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Langfristiger japanischer Staatsanleihezins."),
    },
    "Schweiz": {
        "cpi": _s("cpi", "CHECPIALLMINMEI", "CPI", "Inflation", "monthly", "yoy_pct", 18, "% YoY", "Verbraucherpreisindex der Schweiz."),
        "gdp": _s("gdp", "CLVMNACSCAB1GQCH", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 55, "% YoY", "Reales Bruttoinlandsprodukt der Schweiz."),
        "industrial_production": _s("industrial_production", "CHEPROINDMISMEI", "Industrieproduktion", "Wachstum", "monthly", "yoy_pct", 45, "% YoY", "Industrieproduktion der Schweiz."),
        "unemployment": _s("unemployment", "LRUNTTTTCHM156S", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 20, "%", "Harmonisierte Arbeitslosenquote der Schweiz."),
        "policy_rate": _s("policy_rate", "IRSTCI01CHM156N", "Kurzfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Kurzfristiger Schweizer Zinssatz als geldpolitischer Proxy."),
        "yield_10y": _s("yield_10y", "IRLTLT01CHM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Langfristiger Schweizer Staatsanleihezins."),
    },
    "Kanada": {
        "cpi": _s("cpi", "CANCPIALLMINMEI", "CPI", "Inflation", "monthly", "yoy_pct", 20, "% YoY", "Verbraucherpreisindex Kanadas."),
        "gdp": _s("gdp", "CLVMNACSCAB1GQCA", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 45, "% YoY", "Reales Bruttoinlandsprodukt Kanadas."),
        "industrial_production": _s("industrial_production", "CANPROINDMISMEI", "Industrieproduktion", "Wachstum", "monthly", "yoy_pct", 45, "% YoY", "Industrieproduktion Kanadas."),
        "unemployment": _s("unemployment", "LRUNTTTTCAM156S", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 12, "%", "Arbeitslosenquote Kanadas."),
        "policy_rate": _s("policy_rate", "IRSTCI01CAM156N", "Kurzfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Kurzfristiger kanadischer Zinssatz als geldpolitischer Proxy."),
        "yield_10y": _s("yield_10y", "IRLTLT01CAM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Langfristiger kanadischer Staatsanleihezins."),
    },
    "Australien": {
        "cpi": _s("cpi", "AUSCPIALLMINMEI", "CPI", "Inflation", "monthly", "yoy_pct", 30, "% YoY", "Verbraucherpreisindex Australiens."),
        "gdp": _s("gdp", "CLVMNACSCAB1GQAU", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 60, "% YoY", "Reales Bruttoinlandsprodukt Australiens."),
        "industrial_production": _s("industrial_production", "AUSPROINDQISMEI", "Industrieproduktion", "Wachstum", "quarterly", "yoy_pct", 60, "% YoY", "Industrieproduktion Australiens."),
        "unemployment": _s("unemployment", "LRUNTTTTAUM156S", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 20, "%", "Arbeitslosenquote Australiens."),
        "policy_rate": _s("policy_rate", "IRSTCI01AUM156N", "Kurzfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Kurzfristiger australischer Zinssatz als geldpolitischer Proxy."),
        "yield_10y": _s("yield_10y", "IRLTLT01AUM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Langfristiger australischer Staatsanleihezins."),
    },
    "Neuseeland": {
        "cpi": _s("cpi", "NZLCPIALLQINMEI", "CPI", "Inflation", "quarterly", "yoy_pct", 30, "% YoY", "Verbraucherpreisindex Neuseelands."),
        "gdp": _s("gdp", "CLVMNACSCAB1GQNZ", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 60, "% YoY", "Reales Bruttoinlandsprodukt Neuseelands."),
        "unemployment": _s("unemployment", "LRUNTTTTNZQ156S", "Arbeitslosenquote", "Arbeitsmarkt", "quarterly", "level", 45, "%", "Arbeitslosenquote Neuseelands."),
        "policy_rate": _s("policy_rate", "IRSTCI01NZM156N", "Kurzfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Kurzfristiger neuseeländischer Zinssatz als geldpolitischer Proxy."),
        "yield_10y": _s("yield_10y", "IRLTLT01NZM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Langfristiger neuseeländischer Staatsanleihezins."),
    },
    "Mexiko": {
        "cpi": _s("cpi", "MEXCPIALLMINMEI", "CPI", "Inflation", "monthly", "yoy_pct", 18, "% YoY", "Verbraucherpreisindex Mexikos."),
        "gdp": _s("gdp", "CLVMNACSCAB1GQMX", "Reales BIP", "Wachstum", "quarterly", "yoy_pct", 55, "% YoY", "Reales Bruttoinlandsprodukt Mexikos."),
        "industrial_production": _s("industrial_production", "MEXPROINDMISMEI", "Industrieproduktion", "Wachstum", "monthly", "yoy_pct", 45, "% YoY", "Industrieproduktion Mexikos."),
        "unemployment": _s("unemployment", "LRUNTTTTMXM156S", "Arbeitslosenquote", "Arbeitsmarkt", "monthly", "level", 20, "%", "Arbeitslosenquote Mexikos."),
        "policy_rate": _s("policy_rate", "IRSTCI01MXM156N", "Kurzfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Kurzfristiger mexikanischer Zinssatz als geldpolitischer Proxy."),
        "yield_10y": _s("yield_10y", "IRLTLT01MXM156N", "Langfristzins", "Geldpolitik", "monthly", "level", 5, "%", "Langfristiger mexikanischer Staatsanleihezins."),
    },
}

COUNTRY_WEIGHTS = {
    "cpi": 0.45,
    "core_cpi": 0.25,
    "gdp": 0.85,
    "industrial_production": 0.65,
    "retail_sales": 0.45,
    "unemployment": -0.75,
    "nfp": 0.65,
    "policy_rate": 0.75,
    "fed_funds": 0.75,
    "yield_2y": 0.75,
    "yield_10y": 0.40,
}


def isolated_currency_profile(asset: Asset):
    country = CURRENCY_COUNTRY.get(asset.name)
    if not country:
        return None
    specs = COUNTRY_SERIES[country]
    weights = {key: COUNTRY_WEIGHTS[key] for key in specs if key in COUNTRY_WEIGHTS}
    return country, specs, weights
