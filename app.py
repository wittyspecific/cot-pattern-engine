from __future__ import annotations
from datetime import date

import pandas as pd
import streamlit as st
from assets.registry import load_assets, load_categories, grouped_assets
from data_engine.service import load_market_data
from pattern_engine.analysis import NET_COLS, OI_RATIO_COLS, net_similar_weeks, prepare_dataset, timing_analysis, validate_open_interest_context
from scanner.market_scanner import scan_assets, top_five

st.set_page_config(page_title="COT Pattern Engine", page_icon="📈", layout="wide")
assets = load_assets()
categories = load_categories()
groups = grouped_assets()

st.title("COT Pattern Engine")
st.caption("Modularer Neuaufbau · eindeutige Asset Registry · Datenaktualität wird vor jeder Analyse geprüft")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Bereich", ["Startseite", "Asset-Analyse", "Datenstatus"], label_visibility="collapsed")
    st.divider()
    category_name = st.selectbox(
        "Kategorie",
        list(groups),
        format_func=lambda name: f"{categories[name].icon} {name}",
    )
    asset_names = [asset.name for asset in groups[category_name]]
    asset_name = st.selectbox("Future", asset_names)
    st.divider()
    st.subheader("Analyseeinstellungen")
    n_neighbors = st.selectbox(
        "Historische Netto-Vergleichsfälle",
        [20, 30, 40, 50],
        index=1,
        help=(
            "Bestimmt, wie viele historische COT-Konstellationen anhand der absoluten "
            "Nettopositionen verglichen werden. 30 ist der empfohlene Standardwert."
        ),
    )
    move_threshold_pct = st.selectbox(
        "Mindestbewegung für den Bewegungsbeginn",
        [0.5, 1.0, 1.5, 2.0],
        index=1,
        format_func=lambda value: f"ab {value:.1f} % Bewegung",
        help=(
            "Bestimmt, ab welcher Kursveränderung eine historische Bewegung als begonnen gilt. "
            "Diese Einstellung verändert vor allem das Setup-Zeitfenster, nicht die Richtungsquote."
        ),
    )
    move_threshold = float(move_threshold_pct) / 100.0
    st.caption("Empfohlen: 30 Vergleichsfälle und 1,0 % Mindestbewegung.")
    st.divider()
    force = st.button("Daten neu laden", use_container_width=True)

if page == "Startseite":
    st.subheader("Top 5 aktuelle Futures")
    st.write("Gerankt werden nur Märkte mit aktuellen COT- und Preisdaten sowie mindestens 15 vollständigen Vergleichsfällen.")
    if st.button("Top-5-Scan starten", type="primary"):
        progress = st.progress(0, text="Scan wird vorbereitet …")
        rows = scan_assets(assets, force=force, n_neighbors=n_neighbors, move_threshold=move_threshold, progress=lambda i, n, name: progress.progress(i/n, text=f"{name} ({i}/{n})"))
        progress.empty()
        leaders = top_five(rows)
        if leaders.empty:
            st.warning("Keine fünf belastbaren und aktuellen Märkte verfügbar.")
        else:
            shown = leaders.copy()
            shown["Trefferquote"] = shown["Trefferquote"].map(lambda value: f"{value:.1%}")
            shown["OI bestätigt"] = shown["OI bestätigt"].map(lambda value: "✅" if value else "–")
            st.dataframe(shown, use_container_width=True, hide_index=True)
        failures = [row for row in rows if row["Fehler"]]
        if failures:
            with st.expander(f"{len(failures)} Märkte wurden wegen Datenproblemen ausgeschlossen"):
                st.dataframe(pd.DataFrame(failures)[["Asset", "Kategorie", "Fehler"]], hide_index=True, use_container_width=True)
    else:
        st.info("Der vollständige Scan wird bewusst nur manuell gestartet, da er alle Märkte aktualisiert und mehrere Minuten dauern kann.")

elif page == "Datenstatus":
    st.subheader("Datenstatus")
    st.write("Veraltete Märkte werden nicht analysiert und gelangen nicht in das Top-5-Ranking.")
    asset = assets[asset_name]
    try:
        with st.spinner("Daten werden geprüft …"):
            market = load_market_data(asset, force=force)
        col1, col2 = st.columns(2)
        col1.metric("COT-Stichtag", market.cot_quality.latest_date.strftime("%d.%m.%Y") if market.cot_quality.latest_date is not None else "–")
        col2.metric("Letzter Preistag", market.price_quality.latest_date.strftime("%d.%m.%Y") if market.price_quality.latest_date is not None else "–")
        if market.usable:
            st.success("Daten sind aktuell und für die Analyse freigegeben.")
        else:
            st.error("Analyse gesperrt: Mindestens eine Datenquelle ist veraltet.")
        st.write(f"**CFTC-Zuordnung:** `{asset.cftc_code}` · {asset.cftc_market_name}")
        st.write(f"**Preissymbol:** `{asset.price_symbol}` · **Börse:** {asset.exchange}")
        st.write(f"**COT:** {market.cot_quality.message} · Quelle: {market.cot_source}")
        st.write(f"**Preis:** {market.price_quality.message} · Quelle: {market.price_source}")
    except Exception as exc:
        st.exception(exc)

else:
    asset = assets[asset_name]
    try:
        with st.spinner("COT- und Preisdaten werden geladen …"):
            market = load_market_data(asset, force=force)
        if not market.usable:
            st.error("Diese Analyse wurde gesperrt, weil die Daten nicht aktuell sind.")
            st.write(f"**COT:** {market.cot_quality.message}")
            st.write(f"**Preis:** {market.price_quality.message}")
            st.stop()

        observation_weeks = 8
        oi_threshold = 1.0
        data = prepare_dataset(market.cot, market.prices, [observation_weeks])
        valid = data.dropna(subset=NET_COLS + OI_RATIO_COLS).copy()
        if len(valid) < 3:
            st.warning("Für dieses Asset liegen nicht genügend auswertbare Daten vor.")
            st.stop()

        current = valid.iloc[-1]
        matches = net_similar_weeks(valid, n_neighbors, observation_weeks)
        oi_matches = validate_open_interest_context(matches, valid, oi_threshold)
        net = timing_analysis(matches, market.prices, observation_weeks, move_threshold)
        oi = timing_analysis(oi_matches, market.prices, observation_weeks, move_threshold)

        def bias_meta(result):
            count = int(result["count"])
            if not count:
                return "⚪", "Keine Aussage", None
            bias_value = str(result["bias"])
            if bias_value == "bullish":
                return "🟢", "Bullish", float(result["positive_rate"])
            if bias_value == "bearish":
                return "🔴", "Bearish", float(result["negative_rate"])
            return "⚪", "Neutral", max(float(result["positive_rate"]), float(result["negative_rate"]))

        net_icon, net_label, net_conf = bias_meta(net)
        oi_icon, oi_label, oi_conf = bias_meta(oi)
        net_count = int(net["count"])
        oi_count = int(oi["count"])

        if oi_count < 8:
            confirmation_icon = "⚪"
            confirmation = "Nicht belastbar prüfbar"
            confirmation_text = "Zu wenige Netto-Treffer besitzen zusätzlich einen ausreichend ähnlichen Open-Interest-Kontext."
        elif net["bias"] == oi["bias"] and net["bias"] != "neutral":
            confirmation_icon = "✅"
            if oi_conf is not None and net_conf is not None and oi_conf >= net_conf:
                confirmation = "Open Interest bestätigt und verstärkt"
                confirmation_text = "Richtung und Trefferquote werden durch den vergleichbaren Open-Interest-Kontext bestätigt."
            else:
                confirmation = "Open Interest bestätigt die Richtung"
                confirmation_text = "Die Richtung bleibt gleich, die Trefferquote wird jedoch nicht verbessert."
        elif oi["bias"] == "neutral":
            confirmation_icon = "⚠️"
            confirmation = "Open Interest bestätigt nicht eindeutig"
            confirmation_text = "Die Nettoanalyse zeigt einen Bias, im vergleichbaren Open-Interest-Kontext entsteht jedoch keine klare Mehrheit."
        else:
            confirmation_icon = "❌"
            confirmation = "Open Interest widerspricht"
            confirmation_text = "Die vergleichbaren Open-Interest-Fälle zeigen eine andere historische Richtung als die reine Nettoanalyse."

        use_oi_timing = (
            oi_count >= 8
            and oi["bias"] == net["bias"]
            and oi["bias"] != "neutral"
            and oi["window_start"] is not None
        )
        timing = oi if use_oi_timing else net
        bias = str(net["bias"])
        if bias == "bullish":
            setup_name, avoid_name, direction_word = "Demand- und Long-Setups", "aggressive Short-Setups", "Aufwärtsbewegung"
        elif bias == "bearish":
            setup_name, avoid_name, direction_word = "Supply- und Short-Setups", "aggressive Long-Setups", "Abwärtsbewegung"
        else:
            setup_name, avoid_name, direction_word = "bestätigte Setups", "Positionen ohne Bestätigung", "gerichtete Bewegung"

        report_date = pd.Timestamp(current["date"]).normalize()
        today = pd.Timestamp(date.today()).normalize()
        age_days = max(0, (today - report_date).days)
        release_estimate = report_date + pd.Timedelta(days=3)
        price_dates = pd.to_datetime(market.prices["date"]).dt.normalize().drop_duplicates().sort_values()
        current_trading_day = int(((price_dates > report_date) & (price_dates <= today)).sum())

        window_start = timing["window_start"]
        window_end = timing["window_end"]
        if window_start is not None and window_end is not None:
            window_start_date = report_date + pd.offsets.BDay(int(window_start))
            window_end_date = report_date + pd.offsets.BDay(int(window_end))
            if current_trading_day < window_start:
                status_icon, status = "🟡", "Noch warten"
                status_detail = f"Das historische Suchfenster beginnt voraussichtlich in etwa {int(window_start-current_trading_day)} Handelstag(en)."
                action = f"Bias beobachten, aber noch nicht aktiv nach {setup_name} suchen."
            elif current_trading_day <= window_end:
                status_icon, status = "🟢", "Jetzt aktiv suchen"
                status_detail = f"Das statistisch interessante Suchfenster läuft noch etwa {int(window_end-current_trading_day)} Handelstag(e)."
                action = f"Aktiv nach {setup_name} suchen und {avoid_name} vermeiden."
            else:
                status_icon, status = "🔴", "Zeitfenster abgelaufen"
                status_detail = f"Das typische Suchfenster endete vor etwa {int(current_trading_day-window_end)} Handelstag(en)."
                action = "Keinen neuen Einstieg allein auf Basis dieses COT-Musters planen."
        else:
            window_start_date = window_end_date = None
            status_icon, status = "⚪", "Kein belastbares Timing"
            status_detail = "Aus den historischen Treffern konnte kein stabiles Zeitfenster abgeleitet werden."
            action = "Zusätzliche Marktbestätigung abwarten."

        st.subheader(f"{categories[asset.category].icon} {asset.name}: verwendeter COT-Bericht")
        st.caption(f"Analyse mit {n_neighbors} historischen Netto-Vergleichsfällen und {move_threshold_pct:.1f} % Mindestbewegung.")
        date_cols = st.columns(3)
        date_cols[0].metric("COT-Stichtag", report_date.strftime("%d.%m.%Y"))
        date_cols[1].metric("Veröffentlichung ungefähr", release_estimate.strftime("%d.%m.%Y"))
        date_cols[2].metric("Heute seit Stichtag", f"Tag {current_trading_day}")
        st.caption(f"Die Positionen stammen vom Stichtag und sind heute {age_days} Kalendertage alt.")

        st.divider()
        st.subheader("1. Reines Muster der absoluten Nettopositionen")
        st.markdown(f"### {net_icon} {net_label}")
        cols = st.columns(3)
        cols[0].metric("Ähnliche Netto-Fälle", net_count)
        cols[1].metric("Gestiegen", f"{int(net['positive_count'])} · {float(net['positive_rate']):.1%}" if net_count else "–")
        cols[2].metric("Gefallen", f"{int(net['negative_count'])} · {float(net['negative_rate']):.1%}" if net_count else "–")
        st.caption("Diese erste Aussage basiert ausschließlich auf Commercial-, Non-Commercial- und Non-Reportable-Nettopositionen. Open Interest beeinflusst die Auswahl dieser Fälle nicht.")

        st.subheader("2. Unabhängige Open-Interest-Prüfung")
        st.markdown(f"### {confirmation_icon} {confirmation}")
        cols = st.columns(3)
        cols[0].metric("Davon OI-vergleichbar", f"{oi_count} von {net_count}")
        cols[1].metric("OI-Bias", f"{oi_icon} {oi_label}")
        cols[2].metric("Trefferquote", "–" if oi_conf is None else f"{oi_conf:.1%}")
        st.write(confirmation_text)
        st.caption("Es werden nur die zuvor gefundenen Netto-Fälle betrachtet. Anschließend wird geprüft, ob die drei Nettopositionen relativ zum jeweiligen Open Interest ebenfalls dem aktuellen Bericht ähneln.")

        st.divider()
        st.subheader("Gesamteinordnung und Setup-Zeitfenster")
        st.markdown(f"### {status_icon} {status}")
        st.write(status_detail)
        if window_start_date is not None and window_end_date is not None:
            source = "Netto + Open Interest" if use_oi_timing else "reine Nettoanalyse"
            st.write(f"Auf Basis von **{source}** begann die historische **{direction_word}** meist zwischen **Handelstag {window_start} und {window_end}** nach dem COT-Stichtag (**{window_start_date:%d.%m.%Y} bis {window_end_date:%d.%m.%Y}**).")
        st.info(f"**Handelsplan:** {action}")

        if net_count < 15:
            st.warning("Die Netto-Stichprobe ist begrenzt. Die Aussage sollte nicht als eigenständiges Einstiegssignal verwendet werden.")
        if 0 < oi_count < 8:
            st.warning("Die Open-Interest-Untergruppe ist zu klein für eine belastbare Bestätigung oder Widerlegung.")

        with st.expander("Aktuelle Positionierung und Assetdetails anzeigen"):
            labels = ["Commercials", "Non-Commercials", "Retail / Non-Reportables"]
            for box, label_name, net_col, oi_col in zip(st.columns(3), labels, NET_COLS, OI_RATIO_COLS):
                box.metric(label_name, f"{float(current[net_col]):,.0f} Kontrakte")
                box.caption(f"{float(current[oi_col]):.1%} des Open Interest")
            st.divider()
            st.write(f"**Kategorie:** {asset.category} · **Börse:** {asset.exchange}")
            st.write(f"**CFTC-Code:** `{asset.cftc_code}`")
            st.write(f"**CFTC-Markt:** {asset.cftc_market_name}")
            st.write(f"**Preissymbol:** `{asset.price_symbol}`")
            st.write(f"**Datenstatus:** COT {market.cot_quality.message}; Preis {market.price_quality.message}")

        with st.expander("Historische Vergleichsfälle im Detail"):
            details = net["details"].copy()
            if details.empty:
                st.info("Keine vollständigen historischen Treffer verfügbar.")
            else:
                extra = matches[["date", "net_similarity_score"]].copy()
                confirmed_dates = set(pd.to_datetime(oi_matches["date"])) if not oi_matches.empty else set()
                details = details.merge(extra, on="date", how="left")
                details["OI bestätigt"] = pd.to_datetime(details["date"]).isin(confirmed_dates)
                details["final_return"] *= 100
                details = details.rename(columns={
                    "date": "COT-Stichtag",
                    "start_price": "Preis am Stichtag",
                    "final_return": "Veränderung in 8 Wochen (%)",
                    "direction": "Richtung",
                    "onset_trading_day": "Bewegungsbeginn (Handelstag)",
                    "net_similarity_score": "Netto-Ähnlichkeit (%)",
                }).sort_values("COT-Stichtag", ascending=False)
                st.dataframe(details, use_container_width=True, hide_index=True)
                st.download_button("Treffer als CSV herunterladen", details.to_csv(index=False).encode("utf-8"), file_name=f"cot_faelle_{asset.name}.csv", mime="text/csv", use_container_width=True)

        st.caption(f"Die Auswertung verwendet {n_neighbors} historische Netto-Vergleichsfälle. Eine Bewegung gilt ab {move_threshold_pct:.1f} % als begonnen. Die Ergebnisse beschreiben historische Häufigkeiten und stellen kein eigenständiges Einstiegssignal dar.")
    except Exception as exc:
        st.exception(exc)
