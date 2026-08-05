from scanner.market_scanner import top_five_transitions


def test_top_five_transitions_prioritizes_phase_then_score():
    rows = [
        {"Asset": "A", "Kategorie": "X", "Transition Phase": 1, "Transition Score": 99, "Transition Richtung": "Bullisch", "Transition Status": "Extreme", "Extrem + OI": True, "Retail unterstützt": True, "NC-Konvergenz": False, "Preisbestätigung": False, "NC-Divergenz 1W": "–", "NC-Divergenz 2W": "–", "NC-Divergenz 4W": "–", "NC-Divergenz 8W": "–", "COT-Stichtag": "2026-01-01", "Fehler": ""},
        {"Asset": "B", "Kategorie": "X", "Transition Phase": 2, "Transition Score": 70, "Transition Richtung": "Bearisch", "Transition Status": "Flow", "Extrem + OI": True, "Retail unterstützt": False, "NC-Konvergenz": True, "Preisbestätigung": False, "NC-Divergenz 1W": "–", "NC-Divergenz 2W": "Bearisch", "NC-Divergenz 4W": "–", "NC-Divergenz 8W": "–", "COT-Stichtag": "2026-01-01", "Fehler": ""},
        {"Asset": "C", "Kategorie": "X", "Transition Phase": 0, "Transition Score": 100, "Transition Richtung": "Neutral", "Transition Status": "None", "Extrem + OI": False, "Retail unterstützt": False, "NC-Konvergenz": False, "Preisbestätigung": False, "NC-Divergenz 1W": "–", "NC-Divergenz 2W": "–", "NC-Divergenz 4W": "–", "NC-Divergenz 8W": "–", "COT-Stichtag": "2026-01-01", "Fehler": ""},
    ]
    result = top_five_transitions(rows)
    assert list(result["Asset"]) == ["B", "A"]
