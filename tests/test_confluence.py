import pandas as pd

from dashboard.confluence import build_confluence_row, module_evidence, rank_confluence, wilson_interval


def ev(direction, up, down, median):
    return module_evidence(direction, up + down, up, down, median)


def test_wilson_penalizes_small_samples():
    low_small, _ = wilson_interval(8, 10)
    low_large, _ = wilson_interval(80, 100)
    assert low_large > low_small


def test_full_confirmation_without_weighted_score():
    row = build_confluence_row(
        "Gold", "Edelmetalle", 8,
        ev("Bullish", 24, 6, 0.04),
        ev("Bullish", 16, 4, 0.03),
        ev("Bullish", 15, 5, 0.02),
    )
    assert row["Status"] == "Vollständig bestätigt"
    assert row["Bestätigungen"] == 3
    assert row["Richtung"] == "Bullish"
    assert "Gesamtscore" not in row
    assert row["Robuste KI-Untergrenze"] == min(
        row["COT KI unten"], row["Fundamental KI unten"], row["Seasonality KI unten"]
    )


def test_conflict_is_not_hidden_by_strong_module():
    row = build_confluence_row(
        "WTI", "Energie", 8,
        ev("Bullish", 29, 1, 0.08),
        ev("Bearish", 5, 20, -0.05),
        ev("Bullish", 14, 6, 0.03),
    )
    assert row["Status"] == "Widersprüchlich"


def test_ranking_uses_weakest_evidence_lexicographically():
    a = build_confluence_row("A", "X", 8, ev("Bullish", 8, 2, .02), ev("Bullish", 8, 2, .02), ev("Bullish", 8, 2, .02))
    b = build_confluence_row("B", "X", 8, ev("Bullish", 80, 20, .02), ev("Bullish", 80, 20, .02), ev("Bullish", 80, 20, .02))
    ranked = rank_confluence([a, b])
    assert ranked.iloc[0]["Asset"] == "B"


def test_common_history_uses_overlap_only():
    row = build_confluence_row(
        "Gold", "Edelmetalle", 8, ev("Bullish", 8, 2, .02), ev("Bullish", 8, 2, .02), ev("Bullish", 8, 2, .02),
        {"COT-Start": "2000-01-01", "Fundamental-Start": "2010-01-01", "Seasonality-Start": "2005-01-01",
         "COT-Stichtag": "2025-01-01", "Fundamental-Snapshot": "2024-12-01", "Preis-Stichtag": "2025-02-01"},
    )
    assert 14.8 < row["Gemeinsamer Zeitraum Jahre"] < 15.1
