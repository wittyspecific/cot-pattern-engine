# COT Pattern Engine – modularer Neuaufbau

Diese Version trennt die Anwendung in sechs Bereiche:

- `assets/`: zentrale Asset Registry mit Kategorien, Börsen, CFTC-Codes und Preissymbolen
- `data_engine/`: CFTC-Abruf, Preisabruf, Cache und Aktualitätskontrolle
- `pattern_engine/`: Nettovergleich und Timinganalyse
- `validation/`: unabhängige Open-Interest-Prüfung
- `scanner/`: marktübergreifender Top-5-Scanner
- `dashboard/` bzw. `app.py`: reine Darstellung

## Zentrale Sicherheitsregel

Ein Future wird nicht analysiert, wenn der jüngste COT-Stichtag älter als 14 Tage oder der jüngste Preistag älter als 7 Tage ist. Ein vorhandener Cache wird bei einem Abruffehler zwar diagnostisch geladen, aber nicht als aktuell ausgegeben.

## Kategorien

Die Asset-Auswahl ist nach Aktienindizes, Edelmetallen, Energie, Anleihen, Währungen, Agrarrohstoffen, Soft Commodities und Vieh organisiert.

## Start unter macOS

```bash
cd /Users/kevinbusch/Downloads/cot_pattern_engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Hinweis zur Registry

Die CFTC-Codes sind als zentrale Zuordnungsfelder angelegt. Da in der isolierten Build-Umgebung kein Livezugriff auf die CFTC-API bestand, muss die Anwendung beim ersten Onlineabruf die Codes gegen die aktuelle CFTC-Ausgabe validieren. Schlägt ein Code fehl, wird kontrolliert auf den exakten Marktnamen zurückgefallen; unscharfe Teilstrings werden nicht mehr verwendet.


## Sichtbare Analyseeinstellungen

In der Seitenleiste können nun zwei zentrale Parameter festgelegt werden:

- **Historische Netto-Vergleichsfälle:** 20, 30, 40 oder 50; Standardwert 30.
- **Mindestbewegung für den Bewegungsbeginn:** 0,5 %, 1,0 %, 1,5 % oder 2,0 %; Standardwert 1,0 %.

Die Anzahl der Vergleichsfälle beeinflusst Stichprobengröße und Ähnlichkeit. Die Mindestbewegung beeinflusst primär das abgeleitete Setup-Zeitfenster. Dieselben Einstellungen werden sowohl in der Einzelanalyse als auch im Top-5-Scanner verwendet.
