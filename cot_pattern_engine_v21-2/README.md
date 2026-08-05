# COT Pattern Research – Version 20

Die Oberfläche wurde auf die Kernfrage reduziert:

1. Wie oft trat ein ähnliches COT-Muster historisch auf?
2. Wie oft stieg bzw. fiel der Markt danach und wie groß war die typische Bewegung?
3. Wann begann diese Bewegung typischerweise ab dem COT-Stichtag?

Enthalten sind zwei getrennte Musteranalysen:

- Muster aus absoluten Nettopositionen
- Muster aus Commercial-/Retail-COT-Indexwerten

Die Top-5-Rankings werden anhand der Kombination aus typischer Bewegungsgröße, Richtungshäufigkeit und Anzahl unabhängiger historischer Episoden sortiert.

## Start macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Qualitätsfilter (Version 21)

Die Top-5-Listen und die Asset-Detailansicht verwenden drei einstellbare Mindestanforderungen:

- minimale Richtungsquote (Standard: 70 %)
- minimale Zahl unabhängiger historischer Episoden (Standard: 25)
- minimale absolute Medianbewegung (Standard: 2 %)

Ein Muster gilt nur dann als qualifiziert, wenn alle drei Bedingungen erfüllt sind. Die Detailansicht nennt ausdrücklich, welche Bedingung gegebenenfalls nicht erfüllt wurde.
