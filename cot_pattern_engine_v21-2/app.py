from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from assets.registry import grouped_assets, load_assets, load_categories
from data_engine.service import load_market_data
from pattern_engine.analysis import NET_COLS, OI_RATIO_COLS, prepare_dataset
from pattern_engine.clusters import analyze_cluster_timing, analyze_current_cluster
from pattern_engine.cot_index import analyze_cot_index_pattern
from scanner.market_scanner import scan_assets, top_five_clusters, top_five_cot_index

st.set_page_config(page_title="COT Pattern Research", page_icon="📊", layout="wide")

assets = load_assets()
categories = load_categories()
groups = grouped_assets()

st.title("COT Pattern Research")
st.caption(
    "Die Anwendung beantwortet zwei Fragen: Wie oft trat ein ähnliches COT-Muster auf, "
    "und welche typische Preisbewegung folgte darauf? Das Beginnfenster wird ab dem COT-Stichtag gemessen."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Bereich", ["Übersicht", "Asset-Analyse", "Datenstatus"], label_visibility="collapsed")
    st.divider()
    category_name = st.selectbox(
        "Kategorie",
        list(groups),
        format_func=lambda name: f"{categories[name].icon} {name}",
    )
    asset_names = [asset.name for asset in groups[category_name]]
    asset_name = st.selectbox("Future", asset_names)
    st.divider()
    horizon = st.selectbox(
        "Beobachtungszeitraum",
        [4, 8, 12],
        index=1,
        format_func=lambda value: f"{value} Wochen",
        help="Die typische Bewegung wird vom COT-Stichtag bis zum Ende dieses Zeitraums gemessen.",
    )
    st.caption("Empfohlen: 8 Wochen.")
    st.divider()
    st.subheader("Qualitätsfilter")
    min_hit_rate_pct = st.slider(
        "Minimale Trefferquote",
        min_value=50,
        max_value=95,
        value=70,
        step=1,
        format="%d%%",
        help="Nur Muster mit mindestens dieser Richtungsquote werden in den Top-5-Listen angezeigt.",
    )
    min_hit_rate = min_hit_rate_pct / 100.0
    min_episodes = st.slider(
        "Minimale historische Episoden",
        min_value=5,
        max_value=60,
        value=25,
        step=1,
        help="Verhindert, dass kleine Stichproben das Ranking dominieren.",
    )
    min_median_move_pct = st.slider(
        "Minimale typische Bewegung",
        min_value=0.0,
        max_value=15.0,
        value=2.0,
        step=0.5,
        format="%.1f%%",
        help="Es zählt der Betrag der Medianbewegung, unabhängig von bullish oder bearish.",
    )
    min_median_move = min_median_move_pct / 100.0
    force = st.button("Daten neu laden", use_container_width=True)


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.{digits}%}"


def fmt_return(value: float | None) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:+.1%}"


def direction_counts(matches: pd.DataFrame, return_col: str) -> tuple[int, int, int]:
    if return_col not in matches:
        return 0, 0, 0
    returns = pd.to_numeric(matches[return_col], errors="coerce").dropna()
    return int(len(returns)), int((returns > 0).sum()), int((returns < 0).sum())


def direction_text(up: int, down: int) -> str:
    if up > down:
        return "überwiegend gestiegen"
    if down > up:
        return "überwiegend gefallen"
    return "ausgeglichen"


def simple_cluster_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    view = frame.copy()
    view["Vorkommen"] = view["Cluster Fälle"].astype(int)
    view["Gestiegen"] = view.apply(
        lambda row: f"{int(row['Cluster Gestiegen'])} ({fmt_pct(row['Cluster Gestiegen'] / row['Cluster Fälle'])})"
        if row["Cluster Fälle"] else "–", axis=1
    )
    view["Gefallen"] = view.apply(
        lambda row: f"{int(row['Cluster Gefallen'])} ({fmt_pct(row['Cluster Gefallen'] / row['Cluster Fälle'])})"
        if row["Cluster Fälle"] else "–", axis=1
    )
    view["Typische Bewegung"] = view["Cluster Median"].map(fmt_return)
    view["Beginn"] = view["Cluster Beginnfenster"].map(lambda x: "–" if pd.isna(x) or x is None else f"Tag {x}")
    return view[["Asset", "Kategorie", "Vorkommen", "Gestiegen", "Gefallen", "Typische Bewegung", "Beginn", "COT-Stichtag"]]


def simple_index_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    view = frame.copy()
    view["Commercial Index"] = view["Commercial COT Index"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")
    view["Retail Index"] = view["Retail COT Index"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")
    view["Vorkommen"] = view["Index Fälle"].astype(int)
    view["Gestiegen"] = view.apply(
        lambda row: f"{int(row['Index Gestiegen'])} ({fmt_pct(row['Index Gestiegen'] / row['Index Fälle'])})"
        if row["Index Fälle"] else "–", axis=1
    )
    view["Gefallen"] = view.apply(
        lambda row: f"{int(row['Index Gefallen'])} ({fmt_pct(row['Index Gefallen'] / row['Index Fälle'])})"
        if row["Index Fälle"] else "–", axis=1
    )
    view["Typische Bewegung"] = view["Index Median"].map(fmt_return)
    view["Beginn"] = view["Index Beginnfenster"].map(lambda x: "–" if pd.isna(x) or x is None else str(x))
    return view[["Asset", "Kategorie", "Commercial Index", "Retail Index", "Vorkommen", "Gestiegen", "Gefallen", "Typische Bewegung", "Beginn", "COT-Stichtag"]]


def show_pattern_summary(
    title: str,
    matches: pd.DataFrame,
    return_col: str,
    median_return: float | None,
    timing,
    min_hit_rate: float,
    min_episodes: int,
    min_median_move: float,
) -> None:
    total, up, down = direction_counts(matches, return_col)
    up_rate = up / total if total else None
    down_rate = down / total if total else None

    st.markdown(f"### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historische Vorkommen", total)
    c2.metric("Gestiegen", f"{up} · {fmt_pct(up_rate)}")
    c3.metric("Gefallen", f"{down} · {fmt_pct(down_rate)}")
    c4.metric("Typische Bewegung", fmt_return(median_return))

    dominant_rate = max(up_rate or 0.0, down_rate or 0.0)
    move_size = abs(float(median_return)) if median_return is not None and not pd.isna(median_return) else 0.0
    checks = {
        "Trefferquote": dominant_rate >= min_hit_rate,
        "Episoden": total >= min_episodes,
        "Bewegung": move_size >= min_median_move,
    }
    passed = all(checks.values())
    if passed:
        st.success(
            f"Qualitätsfilter erfüllt: {dominant_rate:.1%} Richtungsquote, "
            f"{total} Episoden und {move_size:.1%} typische Bewegung."
        )
    else:
        failed = []
        if not checks["Trefferquote"]:
            failed.append(f"Trefferquote unter {min_hit_rate:.0%}")
        if not checks["Episoden"]:
            failed.append(f"weniger als {min_episodes} Episoden")
        if not checks["Bewegung"]:
            failed.append(f"Medianbewegung unter {min_median_move:.1%}")
        st.warning("Qualitätsfilter nicht erfüllt: " + ", ".join(failed) + ".")

    st.write(
        f"Dieses Muster trat in **{total} unabhängigen historischen Episoden** auf. "
        f"Der Markt ist danach **{up}-mal gestiegen** und **{down}-mal gefallen**. "
        f"Die typische Bewegung nach dem gewählten Zeitraum betrug **{fmt_return(median_return)}**."
    )

    st.markdown("#### Wann begann die Bewegung typischerweise?")
    if timing is None or not timing.available or timing.onset_window_start is None:
        st.info(timing.reason if timing is not None else "Für dieses Muster ist noch kein belastbares Beginnfenster verfügbar.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Typisches Beginnfenster", f"Tag {timing.onset_window_start}–{timing.onset_window_end}")
        t2.metric("Median-Beginn", f"Tag {timing.onset_day}")
        t3.metric("Heute seit Stichtag", f"Tag {timing.current_trading_day}")
        st.info(timing.status_text)
        st.caption("Alle Tage werden als Handelstage ab dem COT-Stichtag (Dienstag) gezählt.")


if page == "Übersicht":
    st.subheader("Top 5 Muster nach historischer Häufigkeit und Bewegungsgröße")
    st.write(
        "Das Ranking berücksichtigt nur drei Dinge: typische Bewegungsgröße, Häufigkeit der dominanten Richtung "
        "und Anzahl unabhängiger historischer Episoden."
    )
    if st.button("Muster-Scan starten", type="primary"):
        progress = st.progress(0, text="Scan wird vorbereitet …")
        rows = scan_assets(
            assets,
            force=force,
            progress=lambda i, n, name: progress.progress(i / n, text=f"{name} ({i}/{n})"),
        )
        progress.empty()
        st.session_state["pattern_scan_rows"] = rows

    rows = st.session_state.get("pattern_scan_rows")
    if rows:
        st.markdown("### 1. Top 5 Muster aus absoluten Nettopositionen")
        clusters = top_five_clusters(
            rows,
            min_hit_rate=min_hit_rate,
            min_episodes=min_episodes,
            min_median_move=min_median_move,
        )
        if clusters.empty:
            st.info("Keine ausreichenden Muster gefunden.")
        else:
            st.dataframe(simple_cluster_table(clusters), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 2. Top 5 Commercial-/Retail-COT-Index-Muster")
        index_rows = top_five_cot_index(
            rows,
            min_hit_rate=min_hit_rate,
            min_episodes=min_episodes,
            min_median_move=min_median_move,
        )
        if index_rows.empty:
            st.info("Keine ausreichenden Indexmuster gefunden.")
        else:
            st.dataframe(simple_index_table(index_rows), use_container_width=True, hide_index=True)

        failures = [row for row in rows if row.get("Fehler")]
        if failures:
            with st.expander(f"{len(failures)} Märkte wegen Datenproblemen ausgeschlossen"):
                st.dataframe(pd.DataFrame(failures)[["Asset", "Kategorie", "Fehler"]], use_container_width=True, hide_index=True)
    else:
        st.info("Noch kein Scan in dieser Sitzung ausgeführt.")

elif page == "Datenstatus":
    st.subheader("Datenstatus")
    asset = assets[asset_name]
    try:
        with st.spinner("Daten werden geprüft …"):
            market = load_market_data(asset, force=force)
        c1, c2 = st.columns(2)
        c1.metric("COT-Stichtag", market.cot_quality.latest_date.strftime("%d.%m.%Y") if market.cot_quality.latest_date is not None else "–")
        c2.metric("Letzter Preistag", market.price_quality.latest_date.strftime("%d.%m.%Y") if market.price_quality.latest_date is not None else "–")
        if market.usable:
            st.success("Daten sind aktuell und freigegeben.")
        else:
            st.error("Analyse gesperrt: Mindestens eine Datenquelle ist veraltet.")
        st.write(f"**CFTC-Code:** `{asset.cftc_code}`")
        st.write(f"**CFTC-Markt:** {asset.cftc_market_name}")
        st.write(f"**Preissymbol:** `{asset.price_symbol}` · **Börse:** {asset.exchange}")
    except Exception as exc:
        st.exception(exc)

else:
    asset = assets[asset_name]
    try:
        with st.spinner("COT- und Preisdaten werden geladen …"):
            market = load_market_data(asset, force=force)
        if not market.usable:
            st.error("Diese Analyse wurde gesperrt, weil die Daten nicht aktuell sind.")
            st.stop()

        data = prepare_dataset(market.cot, market.prices, [horizon])
        valid = data.dropna(subset=NET_COLS + OI_RATIO_COLS).copy()
        if len(valid) < 80:
            st.warning("Für dieses Asset liegt nicht genügend vollständige Historie vor.")
            st.stop()

        current = valid.iloc[-1]
        report_date = pd.Timestamp(current["date"]).normalize()
        age_days = max(0, (pd.Timestamp(date.today()) - report_date).days)
        st.subheader(asset.name)
        st.caption(f"COT-Stichtag {report_date.strftime('%d.%m.%Y')} · Datenalter etwa {age_days} Kalendertage · {asset.category} · {asset.exchange}")

        cluster = analyze_current_cluster(valid, horizon_weeks=horizon)
        if cluster.available:
            cluster.matches.attrs["current_report_date"] = report_date
            timing_bias = "bullish" if (cluster.median_return or 0) > 0 else "bearish" if (cluster.median_return or 0) < 0 else cluster.bias
            cluster_timing = analyze_cluster_timing(cluster.matches, market.prices, timing_bias, observation_weeks=horizon)
        else:
            cluster_timing = None

        cot_index = analyze_cot_index_pattern(valid, horizon_weeks=horizon)
        if cot_index.available:
            cot_index.matches.attrs["current_report_date"] = report_date
            index_timing_bias = "bullish" if (cot_index.median_return or 0) > 0 else "bearish" if (cot_index.median_return or 0) < 0 else cot_index.bias
            cot_index_timing = analyze_cluster_timing(cot_index.matches, market.prices, index_timing_bias, observation_weeks=horizon)
        else:
            cot_index_timing = None

        tab_net, tab_index = st.tabs(["1 · Absolute Nettopositionen", "2 · Commercial-/Retail-COT-Index"])

        with tab_net:
            if not cluster.available:
                st.warning(cluster.reason)
            else:
                show_pattern_summary(
                    "Muster aus absoluten Nettopositionen",
                    cluster.matches,
                    f"return_{horizon}w",
                    cluster.median_return,
                    cluster_timing,
                    min_hit_rate,
                    min_episodes,
                    min_median_move,
                )
                with st.expander("Historische Fälle anzeigen"):
                    shown = cluster.matches[[c for c in ["date", f"return_{horizon}w"] if c in cluster.matches]].copy().sort_values("date", ascending=False)
                    shown["date"] = pd.to_datetime(shown["date"]).dt.strftime("%d.%m.%Y")
                    shown[f"return_{horizon}w"] = shown[f"return_{horizon}w"].map(fmt_return)
                    st.dataframe(shown, use_container_width=True, hide_index=True)

        with tab_index:
            st.caption(
                "Der Commercial- und Retail-COT-Index wird auf einer Skala von 0 bis 100 berechnet. "
                "Gesucht werden historisch ähnliche Kombinationen beider Werte."
            )
            if not cot_index.available:
                st.warning(cot_index.reason)
            else:
                i1, i2 = st.columns(2)
                i1.metric("Commercial COT-Index", f"{cot_index.commercial_index:.1f}")
                i2.metric("Retail COT-Index", f"{cot_index.retail_index:.1f}")
                show_pattern_summary(
                    "Commercial-/Retail-COT-Index-Muster",
                    cot_index.matches,
                    f"return_{horizon}w",
                    cot_index.median_return,
                    cot_index_timing,
                    min_hit_rate,
                    min_episodes,
                    min_median_move,
                )
                with st.expander("Historische Indexmuster anzeigen"):
                    cols = [c for c in ["date", "commercial_cot_index", "retail_cot_index", f"return_{horizon}w"] if c in cot_index.matches]
                    shown = cot_index.matches[cols].copy().sort_values("date", ascending=False)
                    shown["date"] = pd.to_datetime(shown["date"]).dt.strftime("%d.%m.%Y")
                    shown[f"return_{horizon}w"] = shown[f"return_{horizon}w"].map(fmt_return)
                    st.dataframe(shown, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.exception(exc)
