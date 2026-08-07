# COT Institutional Level Research

Die Anwendung arbeitet in drei getrennten Schritten:

1. **Extremscanner** – zeigt Assets, bei denen Commercials oder Retail im COT-Index oberhalb von 80 oder unterhalb von 20 liegen.
2. **Historical Net Position Research** – wertet ähnliche absolute Nettopositionen historisch aus: Vorkommen, gestiegen/gefallen, typischer Bewegungsbereich und Beginnfenster.
3. **Institutional Flow** – zeigt den aktuellen Non-Commercial-Long-/Short-Aufbau über 1, 2, 4 und 8 Wochen.

Der COT-Index dient nur als Scanner. Er wird nicht mehr direkt mit der Nettopositionsanalyse vermischt.

## Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```


## Version 27 – getrennte Netto-Level-Analyse

- Commercial-Nettopositionen werden zuerst isoliert gegen die spätere Preisentwicklung untersucht.
- Retail-/Non-Reportable-Nettopositionen werden separat ausgewertet.
- Erst danach werden historische Zeitpunkte gesucht, an denen beide Netto-Level gleichzeitig ähnlich waren.
- Wiederkehrende Commercial- und Retail-Nettozonen werden als institutionelle Positionierungsreferenzen gespeichert und statistisch beschrieben.
- Open Interest kann optional als Zusatzkontext zugeschaltet werden; die Primäranalyse bleibt auf absoluten Nettopositionen.


## Version 29 – Reine Nettopositionsmuster

Die Asset-Analyse enthält nun einen eigenen Bereich „Reines Gesamtmuster“. Dort werden ausschließlich die absoluten Nettopositionen von Commercials, Non-Commercials und Retail verglichen. COT-Index und Open Interest beeinflussen diese Mustersuche nicht.


## Version 29

Die Startseite zeigt die Top-5 der reinen absoluten Nettopositionsmuster getrennt für 4, 8 und 12 Wochen. Die bisherige Top-5-Liste des Non-Commercial-Flows wurde entfernt.

## Version 30 – isoliertes Fundamental-Labor

Die Navigation enthält den neuen Bereich **„Fundamental-Labor (isoliert)“**.

- Der Bereich lädt keine COT-Daten und verändert keine bestehende COT-Auswertung.
- Analysiert werden grundlegende US-Fundamentaldaten wie CPI, Kern-CPI, Kern-PCE, reales GDP, Arbeitslosenquote, NFP/Payroll-Veränderung, Federal Funds Rate, US-Renditen, Realrendite, Inflationserwartungen, Industrieproduktion, Einzelhandelsumsätze, M2, US-Dollar und Finanzstress.
- Jedes Asset erhält ein transparentes Fundamentalprofil mit separaten Treibern und Gewichten.
- Historisch ähnliche Fundamentalregime werden ausschließlich aus den standardisierten Fundamentaldaten gesucht. Erst danach werden Preisreaktionen nach 4, 8 und 12 Wochen ausgewertet.
- Für monatliche und quartalsweise Reihen werden angenäherte Veröffentlichungslags berücksichtigt.

### Wichtige Grenze der Testversion

Die FRED-Daten sind in dieser Version überwiegend heutige, teilweise revidierte Daten. Das reduziert die Aussagekraft eines historischen Echtzeit-Backtests. Außerdem fehlen in der ersten Stufe noch physische und länderspezifische Daten, unter anderem EIA-Lagerdaten, USDA/Wetterdaten, Daten der Gegenwährungen und On-Chain-Daten. Die Oberfläche weist die Datenabdeckung je Asset deshalb ausdrücklich aus.

## Version 31 – isolierte Länder-Makromuster

Für Währungsfutures werden die Makrodaten der jeweiligen Heimatwirtschaft getrennt ausgewertet. EUR, GBP, JPY, CHF, CAD, AUD, NZD, MXN und USD besitzen eigene Datenprofile. Es werden keine US-Differenziale und keine COT-Daten in die fundamentale Analogsuche aufgenommen. Die Preisreaktion des jeweiligen Futures wird nach 4, 8 oder 12 Wochen gemessen.


## Version 38

- Price Seasonality als durchschnittliche Jahreskurve über Kalenderwoche 1 bis 52.
- Aktueller Jahresverlauf wird anhand rollierender Zwei-Wochen-Bewegungen zeitlich kalibriert.
- Anzeige, ob der Markt saisonal vorausgelaufen ist, hinterherhinkt oder synchron verläuft.
- Objektive Konfluenzübersicht für COT, isolierte Fundamental-Analogien und kalibrierte Price Seasonality ohne gewichteten Gesamtscore.
- COT, Fundamentals und Seasonality bleiben methodisch getrennte Analysen.


## Price Seasonality v38

Die Seasonality nutzt einen robusten Median-Jahrespfad, ein 25.–75.-Perzentilband, 2W/4W/8W-DTW-Mustervergleich, Phasenkalibrierung, Seasonal Progress und Stabilitätsmessung.


## Objektive Konfluenzübersicht v40

Die Übersicht berechnet COT, Fundamental-Analogien und die statistische Fortsetzung ab der kalibrierten saisonalen Phase unabhängig für einen gemeinsamen Horizont von 4, 8 oder 12 Wochen. Eine manuelle Gewichtung findet nicht statt. Die Sortierung ist lexikografisch: Bestätigungsstatus, Anzahl bestätigender Module, niedrigste Wilson-Konfidenzuntergrenze, schwächste Stichprobe und kleinste Medianbewegung. Konflikte und unzureichende Daten werden separat angezeigt. Ein gemeinsamer Point-in-Time-Dreifach-Backtest ist ausdrücklich noch nicht Bestandteil dieser Version.
