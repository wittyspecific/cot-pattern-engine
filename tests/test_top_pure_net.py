import pandas as pd

from scanner.market_scanner import top_five_pure_net_patterns


def test_top_five_pure_net_patterns_uses_selected_horizon_and_filters():
    rows = []
    for idx in range(7):
        row = {"Asset": f"A{idx}", "Kategorie": "Test", "COT-Stichtag": "2026-08-04"}
        for horizon in (4, 8, 12):
            prefix = f"Reines Netto {horizon}W"
            row.update({
                f"{prefix} verfügbar": True,
                f"{prefix} Fälle": 30 + idx,
                f"{prefix} Gestiegen": 24 + idx,
                f"{prefix} Gefallen": 6,
                f"{prefix} Richtung": "Bullish",
                f"{prefix} Trefferquote": 0.80,
                f"{prefix} Median": 0.03 + idx * 0.01 if horizon == 4 else 0.02,
                f"{prefix} Range unten": 0.02,
                f"{prefix} Range oben": 0.06,
                f"{prefix} Beginnfenster": "4–8",
            })
        rows.append(row)

    result = top_five_pure_net_patterns(rows, 4, 0.70, 25, 0.02)
    assert len(result) == 5
    assert result.iloc[0]["Asset"] == "A6"
    assert "Reines Netto 4W Median" in result.columns


def test_top_five_pure_net_patterns_rejects_invalid_horizon():
    try:
        top_five_pure_net_patterns([], 6)
    except ValueError as exc:
        assert "4, 8 und 12" in str(exc)
    else:
        raise AssertionError("ValueError erwartet")
