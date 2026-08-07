from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from assets.registry import grouped_assets, load_assets, load_categories
from data_engine.service import load_market_data
from data_engine.prices import fetch_prices
from dashboard.confluence import build_confluence_row, module_evidence, rank_confluence
from fundamental_engine.service import load_fundamental_research
from pattern_engine.analysis import NET_COLS, OI_RATIO_COLS, prepare_dataset
from pattern_engine.clusters import analyze_cluster_timing, analyze_current_cluster
from pattern_engine.cot_index import add_cot_indices
from pattern_engine.divergence import analyze_noncommercial_divergence
from pattern_engine.net_levels import analyze_group_net, analyze_overlap, analyze_pure_net_pattern, build_reference_zones
from scanner.market_scanner import scan_assets, top_five_pure_net_patterns
from seasonality_engine import annual_seasonality_curve, calibrate_seasonality

st.set_page_config(page_title="COT Institutional Level Research", page_icon="📊", layout="wide")

assets = load_assets()
categories = load_categories()
groups = grouped_assets()

st.title("COT Institutional Level Research")
st.caption(
    "COT-Research und ein getrenntes Fundamental-Labor: Der Fundamentalbereich nutzt keine COT-Werte und "
    "beeinflusst weder Nettopositionsmuster noch Rankings."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Bereich", ["Übersicht", "Asset-Analyse", "Fundamental-Labor (isoliert)", "Price Seasonality", "Datenstatus"], label_visibility="collapsed")
    st.divider()
    category_name = st.selectbox("Kategorie", list(groups), format_func=lambda x: f"{categories[x].icon} {x}")
    asset_name = st.selectbox("Future", [a.name for a in groups[category_name]])
    st.divider()
    horizon = st.selectbox("Beobachtungszeitraum", [4, 8, 12], index=1, format_func=lambda x: f"{x} Wochen")
    cases_label = "Ähnliche Fundamentalregime" if page == "Fundamental-Labor (isoliert)" else "Historische Vergleichsfälle"
    net_cases = st.selectbox(cases_label, [20, 30, 40, 50], index=1)
    if page == "Fundamental-Labor (isoliert)":
        include_oi_context = False
        extreme_cutoff = 80
        min_hit_rate_pct = 70
        min_episodes = 25
        min_move_pct = 2.0
        st.info("Dieser Bereich lädt keine COT-Daten und verändert keine COT-Auswertung.")
    elif page == "Übersicht":
        include_oi_context = False
        extreme_cutoff = st.slider("COT-Index-Extrembereich", 70, 95, 80, 5, format="%d")
        min_hit_rate_pct = 70
        min_episodes = 25
        min_move_pct = 2.0
        st.caption("Die Konfluenzübersicht nutzt keine manuellen Qualitätsgrenzen und keinen Gesamtscore.")
    else:
        include_oi_context = st.toggle("Open Interest als Zusatzkontext", value=False, help="Die Primäranalyse bleibt auf Nettopositionen fokussiert. Optional wird die Position relativ zum Open Interest zusätzlich in die Ähnlichkeit aufgenommen.")
        extreme_cutoff = st.slider("COT-Index-Extrembereich", 70, 95, 80, 5, format="%d")
        st.caption(f"Extrem hoch ≥ {extreme_cutoff}; extrem niedrig ≤ {100-extreme_cutoff}.")
        st.divider()
        st.subheader("Muster-Qualitätsfilter")
        min_hit_rate_pct = st.slider("Minimale Trefferquote", 50, 95, 70, 1, format="%d%%")
        min_episodes = st.slider("Minimale historische Episoden", 5, 60, 25, 1)
        min_move_pct = st.slider("Minimale typische Bewegung", 0.0, 15.0, 2.0, 0.5, format="%.1f%%")
    force = st.button("Daten neu laden", use_container_width=True)

min_hit_rate = min_hit_rate_pct / 100.0
min_move = min_move_pct / 100.0


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.{digits}%}"


def fmt_return(value: float | None) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:+.1%}"


def movement_range(matches: pd.DataFrame, return_col: str) -> tuple[float | None, float | None]:
    values = pd.to_numeric(matches.get(return_col), errors="coerce").dropna()
    pos, neg = values[values > 0], values[values < 0]
    selected = pos if len(pos) > len(neg) else neg if len(neg) > len(pos) else pd.Series(dtype=float)
    if selected.empty:
        return None, None
    return float(selected.quantile(.25)), float(selected.quantile(.75))


def fmt_range(low: float | None, high: float | None) -> str:
    return "–" if low is None or high is None else f"{low:+.1%} bis {high:+.1%}"


def index_zone(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Nicht verfügbar"
    if value >= extreme_cutoff:
        return "Extrem hoch"
    if value <= 100 - extreme_cutoff:
        return "Extrem niedrig"
    return "Neutral"


def show_net_research(cluster, timing, return_col: str) -> None:
    if not cluster.available:
        st.warning(cluster.reason)
        return
    returns = pd.to_numeric(cluster.matches[return_col], errors="coerce").dropna()
    total = len(returns)
    up, down = int((returns > 0).sum()), int((returns < 0).sum())
    rate = max(up, down) / total if total else 0.0
    low, high = movement_range(cluster.matches, return_col)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historische Vorkommen", total)
    c2.metric("Gestiegen", f"{up} · {up/total:.1%}" if total else "–")
    c3.metric("Gefallen", f"{down} · {down/total:.1%}" if total else "–")
    c4.metric("Typischer Bewegungsbereich", fmt_range(low, high))
    st.write(f"Medianbewegung: **{fmt_return(cluster.median_return)}** · dominante Richtungsquote: **{rate:.1%}**")
    if timing and timing.available and timing.onset_window_start is not None:
        t1, t2, t3 = st.columns(3)
        t1.metric("Typischer Beginn", f"Tag {timing.onset_window_start}–{timing.onset_window_end}")
        t2.metric("Median-Beginn", f"Tag {timing.onset_day}")
        t3.metric("Heute seit Stichtag", f"Tag {timing.current_trading_day}")
        st.info(timing.status_text)
    else:
        st.info("Für dieses Muster ist noch kein belastbares Beginnfenster verfügbar.")
    if rate >= min_hit_rate and total >= min_episodes and abs(cluster.median_return or 0) >= min_move:
        st.success("Dieses Muster erfüllt den eingestellten Qualitätsfilter.")
    else:
        st.caption("Das Muster wird vollständig angezeigt, auch wenn es den Top-5-Qualitätsfilter nicht erfüllt.")
    with st.expander("Historische Fälle"):
        cols = [c for c in ["date", return_col, "commercial_net", "noncommercial_net", "nonreportable_net"] if c in cluster.matches]
        shown = cluster.matches[cols].sort_values("date", ascending=False).copy()
        shown["date"] = pd.to_datetime(shown["date"]).dt.strftime("%d.%m.%Y")
        if return_col in shown:
            shown[return_col] = shown[return_col].map(fmt_return)
        st.dataframe(shown, use_container_width=True, hide_index=True)




def show_group_research(result, timing, return_col: str) -> None:
    if not result.available:
        st.warning(result.reason)
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historische Vorkommen", result.sample_size)
    c2.metric("Gestiegen", f"{result.up_count} · {result.up_count/result.sample_size:.1%}")
    c3.metric("Gefallen", f"{result.down_count} · {result.down_count/result.sample_size:.1%}")
    c4.metric("Typischer Bewegungsbereich", fmt_range(result.range_low, result.range_high))
    direction = {"bullish": "bullish", "bearish": "bearish", "neutral": "neutral"}.get(result.dominant_direction, "neutral")
    st.write(f"Dominante Richtung: **{direction}** · Trefferquote: **{fmt_pct(result.hit_rate)}** · Medianbewegung: **{fmt_return(result.median_return)}**")
    if result.current_net is not None:
        st.caption(f"Aktuelles Netto-Level: {result.current_net:+,.0f} Kontrakte · relativ zum Open Interest: {fmt_pct(result.current_net_oi)}")
    if timing and timing.available and timing.onset_window_start is not None:
        t1, t2, t3 = st.columns(3)
        t1.metric("Typischer Beginn", f"Tag {timing.onset_window_start}–{timing.onset_window_end}")
        t2.metric("Median-Beginn", f"Tag {timing.onset_day}")
        t3.metric("Heute seit Stichtag", f"Tag {timing.current_trading_day}")
        st.info(timing.status_text)
    else:
        st.info("Für diese Einzelanalyse ist noch kein belastbares Beginnfenster verfügbar.")
    with st.expander("Historische Vergleichsfälle"):
        cols = [c for c in ["date", return_col, "commercial_net", "nonreportable_net", "commercial_net_oi", "nonreportable_net_oi", "distance"] if c in result.matches]
        shown = result.matches[cols].sort_values("date", ascending=False).copy()
        shown["date"] = pd.to_datetime(shown["date"]).dt.strftime("%d.%m.%Y")
        if return_col in shown:
            shown[return_col] = shown[return_col].map(fmt_return)
        st.dataframe(shown, use_container_width=True, hide_index=True)


def timing_for_result(result, prices, report_date, horizon_weeks):
    if not result.available or result.matches.empty or result.dominant_direction == "neutral":
        return None
    result.matches.attrs["current_report_date"] = report_date
    return analyze_cluster_timing(result.matches, prices, result.dominant_direction, observation_weeks=horizon_weeks)


def reference_zone_frame(zones) -> pd.DataFrame:
    rows = []
    for z in zones:
        rows.append({
            "Aktuell": "●" if z.is_current else "",
            "Netto-Zone": f"{z.lower:+,.0f} bis {z.upper:+,.0f}",
            "Zentrum": f"{z.center:+,.0f}",
            "Besuche": z.visits,
            "Gestiegen": z.up_count,
            "Gefallen": z.down_count,
            "Richtung": z.dominant_direction,
            "Trefferquote": fmt_pct(z.hit_rate),
            "Bewegungsbereich": fmt_range(z.range_low, z.range_high),
        })
    return pd.DataFrame(rows)

def _date_min(frame: pd.DataFrame, column: str = "date"):
    if frame is None or frame.empty or column not in frame:
        return None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return None if values.empty else values.min()


def _date_max(frame: pd.DataFrame, column: str = "date"):
    if frame is None or frame.empty or column not in frame:
        return None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return None if values.empty else values.max()


def scan_confluence_assets(asset_map, cot_rows, horizon_weeks: int, n_neighbors: int = 20, force_reload: bool = False):
    """Run the three modules independently and assemble an unweighted overview."""
    rows = []
    cot_lookup = {row.get("Asset"): row for row in (cot_rows or [])}
    items = list(asset_map.values())
    progress = st.progress(0, text="Konfluenzanalyse wird vorbereitet …")
    for i, item in enumerate(items, start=1):
        progress.progress(i / len(items), text=f"COT · Fundamental · Seasonality: {item.name} ({i}/{len(items)})")
        cot_row = cot_lookup.get(item.name, {})
        prefix = f"Reines Netto {horizon_weeks}W"
        cot_available = bool(cot_row.get(f"{prefix} verfügbar", False))
        cot_direction = cot_row.get(f"{prefix} Richtung", "Nicht verfügbar")
        cot_evidence = module_evidence(
            cot_direction,
            cot_row.get(f"{prefix} Fälle", 0),
            cot_row.get(f"{prefix} Gestiegen", 0),
            cot_row.get(f"{prefix} Gefallen", 0),
            cot_row.get(f"{prefix} Median"),
            cot_row.get(f"{prefix} Range unten"),
            cot_row.get(f"{prefix} Range oben"),
            available=cot_available,
            reason=str(cot_row.get("Fehler", "")) if not cot_available else "",
        )

        try:
            prices, _ = fetch_prices(item, force=force_reload)
            price_start, price_end = _date_min(prices), _date_max(prices)
        except Exception as exc:
            prices = pd.DataFrame()
            price_start = price_end = None
            price_error = str(exc)
        else:
            price_error = ""

        if prices.empty:
            fundamental_evidence = module_evidence(None, 0, 0, 0, None, available=False, reason=price_error or "Keine Preisdaten")
            seasonality_evidence = module_evidence(None, 0, 0, 0, None, available=False, reason=price_error or "Keine Preisdaten")
            fundamental_start = fundamental_snapshot = seasonality_start = None
        else:
            try:
                fundamental, _ = load_fundamental_research(
                    item, prices, horizon_weeks=horizon_weeks, n_neighbors=n_neighbors, force=force_reload
                )
                fundamental_direction = (
                    "Bullish" if fundamental.up_count > fundamental.down_count
                    else "Bearish" if fundamental.down_count > fundamental.up_count
                    else "Neutral"
                )
                fundamental_evidence = module_evidence(
                    fundamental_direction,
                    fundamental.sample_size,
                    fundamental.up_count,
                    fundamental.down_count,
                    fundamental.median_return,
                    fundamental.range_low,
                    fundamental.range_high,
                    available=fundamental.available,
                    reason=fundamental.reason,
                )
                fundamental_start = _date_min(fundamental.score_history)
                fundamental_snapshot = fundamental.snapshot_date
            except Exception as exc:
                fundamental_evidence = module_evidence(None, 0, 0, 0, None, available=False, reason=str(exc))
                fundamental_start = fundamental_snapshot = None

            try:
                calibration = calibrate_seasonality(prices, window_years=15)
                forecast = calibration.forecast if calibration.available else pd.DataFrame()
                selected = forecast.loc[forecast["Horizont"].eq(f"{horizon_weeks}W")] if not forecast.empty else pd.DataFrame()
                if selected.empty:
                    seasonality_evidence = module_evidence(
                        None, 0, 0, 0, None, available=False,
                        reason=calibration.reason or "Keine Fortsetzung ab der kalibrierten Phase verfügbar.",
                    )
                else:
                    value = selected.iloc[0]
                    seasonality_evidence = module_evidence(
                        value.get("Richtung"), value.get("Fälle", 0), value.get("Gestiegen", 0), value.get("Gefallen", 0),
                        value.get("Median"), value.get("Q25"), value.get("Q75"), available=True,
                        reason=f"Kalibrierte Phase KW {calibration.calibrated_week}; Stabilität {fmt_pct(calibration.stability)}",
                    )
                seasonality_start = (
                    max(price_start, price_end - pd.DateOffset(years=int(calibration.years)))
                    if calibration.available and price_start is not None and price_end is not None and calibration.years
                    else price_start
                )
            except Exception as exc:
                seasonality_evidence = module_evidence(None, 0, 0, 0, None, available=False, reason=str(exc))
                seasonality_start = None

        metadata = {
            "COT-Start": cot_row.get("COT-Start"),
            "COT-Stichtag": cot_row.get("COT-Stichtag"),
            "Fundamental-Start": fundamental_start,
            "Fundamental-Snapshot": fundamental_snapshot,
            "Seasonality-Start": seasonality_start,
            "Preis-Start": price_start,
            "Preis-Stichtag": price_end if price_end is not None else cot_row.get("Preis-Stichtag"),
        }
        rows.append(build_confluence_row(
            item.name, item.category, horizon_weeks,
            cot_evidence, fundamental_evidence, seasonality_evidence, metadata,
        ))
    progress.empty()
    return rank_confluence(rows)


def _module_cell(row: pd.Series, module: str) -> str:
    direction = row.get(f"{module} Richtung", "Nicht verfügbar")
    n = int(row.get(f"{module} Fälle", 0) or 0)
    if direction not in ("Bullish", "Bearish") or n <= 0:
        return str(direction)
    return (
        f"{direction} · {fmt_pct(row.get(f'{module} Trefferquote'))} · n={n} · "
        f"KI≥{fmt_pct(row.get(f'{module} KI unten'))} · Median {fmt_return(row.get(f'{module} Median'))}"
    )


def confluence_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    view = frame.copy()
    for module in ("COT", "Fundamental", "Seasonality"):
        view[module] = view.apply(lambda row: _module_cell(row, module), axis=1)
    view["Bestätigung"] = view.apply(lambda row: f"{int(row['Bestätigungen'])}/3", axis=1)
    view["Robuste Untergrenze"] = view["Robuste KI-Untergrenze"].map(fmt_pct)
    view["Gemeinsame Historie"] = view["Gemeinsamer Zeitraum Jahre"].map(
        lambda x: "–" if pd.isna(x) else f"{x:.1f} Jahre"
    )
    view["Datenstände"] = view.apply(
        lambda row: (
            f"COT {pd.to_datetime(row.get('COT-Stichtag'), errors='coerce').strftime('%d.%m.%Y') if not pd.isna(pd.to_datetime(row.get('COT-Stichtag'), errors='coerce')) else '–'} · "
            f"Fund. {pd.to_datetime(row.get('Fundamental-Snapshot'), errors='coerce').strftime('%d.%m.%Y') if not pd.isna(pd.to_datetime(row.get('Fundamental-Snapshot'), errors='coerce')) else '–'} · "
            f"Preis {pd.to_datetime(row.get('Preis-Stichtag'), errors='coerce').strftime('%d.%m.%Y') if not pd.isna(pd.to_datetime(row.get('Preis-Stichtag'), errors='coerce')) else '–'}"
        ), axis=1
    )
    return view[[
        "Asset", "Kategorie", "Status", "Richtung", "Bestätigung", "COT", "Fundamental", "Seasonality",
        "Robuste Untergrenze", "Schwächste Stichprobe", "Gemeinsame Historie", "Datenstände",
    ]]


def extreme_scanner_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    required = ["Commercial COT Index", "Retail COT Index"]
    if any(c not in frame for c in required):
        return pd.DataFrame()
    lower = 100 - extreme_cutoff
    mask = (
        frame["Commercial COT Index"].ge(extreme_cutoff) | frame["Commercial COT Index"].le(lower) |
        frame["Retail COT Index"].ge(extreme_cutoff) | frame["Retail COT Index"].le(lower)
    ) & frame["Fehler"].eq("")
    out = frame.loc[mask, ["Asset", "Kategorie", "Commercial COT Index", "Retail COT Index", "COT-Stichtag"]].copy()
    if out.empty:
        return out
    out["Commercial Status"] = out["Commercial COT Index"].map(index_zone)
    out["Retail Status"] = out["Retail COT Index"].map(index_zone)
    out["Extremstärke"] = out[["Commercial COT Index", "Retail COT Index"]].apply(
        lambda r: max(abs(float(r.iloc[0]) - 50), abs(float(r.iloc[1]) - 50)), axis=1
    )
    return out.sort_values("Extremstärke", ascending=False).drop(columns="Extremstärke")


if page == "Übersicht":
    st.subheader("Objektive Konfluenzübersicht")
    st.caption(
        "COT-Nettopositionsmuster, isolierte Fundamental-Analogien und die statistische Fortsetzung ab der "
        "kalibrierten saisonalen Phase werden unabhängig berechnet. Es gibt keinen gewichteten Gesamtscore."
    )

    overview_horizon = st.selectbox(
        "Gemeinsamer Analysehorizont", [4, 8, 12], index=1,
        format_func=lambda value: f"{value} Wochen", key="objective_overview_horizon",
    )
    if st.button("Objektive Konfluenzanalyse starten", type="primary", key="run_objective_confluence"):
        cot_progress = st.progress(0, text="COT-Marktscan wird vorbereitet …")
        scan_rows = scan_assets(
            assets, force=force, n_neighbors=net_cases,
            progress=lambda i, n, name: cot_progress.progress(i/n, text=f"COT: {name} ({i}/{n})"),
        )
        cot_progress.empty()
        st.session_state["scan_rows"] = scan_rows
        st.session_state["confluence_rows"] = scan_confluence_assets(
            assets, scan_rows, horizon_weeks=overview_horizon, n_neighbors=net_cases, force_reload=force
        )
        st.session_state["confluence_horizon"] = overview_horizon
        st.session_state["confluence_timestamp"] = datetime.now()

    confluence = st.session_state.get("confluence_rows")
    used_horizon = st.session_state.get("confluence_horizon")
    if isinstance(confluence, pd.DataFrame) and not confluence.empty:
        full_data = int((confluence["Verfügbare Module"] == 3).sum())
        confirmed = confluence[confluence["Status"].isin(["Vollständig bestätigt", "Teilweise bestätigt"])]
        conflicts = confluence[confluence["Status"].eq("Widersprüchlich")]
        insufficient = confluence[confluence["Status"].eq("Daten unzureichend")]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gemeinsamer Horizont", f"{used_horizon} Wochen")
        c2.metric("Analysierte Assets", len(confluence))
        c3.metric("Alle 3 Module verfügbar", full_data)
        c4.metric("Bestätigte Konstellationen", len(confirmed))
        stamp = st.session_state.get("confluence_timestamp")
        if stamp:
            st.caption(f"Berechnet am {stamp:%d.%m.%Y um %H:%M Uhr}.")

        st.markdown("### Stärkste bestätigte historische Konfluenz")
        if confirmed.empty:
            st.info("Aktuell bestätigen sich bei keinem Asset mindestens zwei Richtungsmodelle.")
        else:
            st.dataframe(confluence_view(confirmed), use_container_width=True, hide_index=True)
        st.caption(
            "Sortierung: Bestätigungsstatus → Anzahl bestätigender Module → niedrigste Wilson-KI-Untergrenze → "
            "schwächste Stichprobengröße → kleinste Medianbewegung. Kein additiver oder gewichteter Score."
        )

        st.markdown("### Aktuelle Modulkonflikte")
        if conflicts.empty:
            st.success("Keine gegensätzlichen Richtungen zwischen den verfügbaren Modulen.")
        else:
            st.dataframe(confluence_view(conflicts), use_container_width=True, hide_index=True)

        with st.expander(f"Assets mit unzureichender Richtungsbasis ({len(insufficient)})"):
            if insufficient.empty:
                st.write("Keine.")
            else:
                st.dataframe(confluence_view(insufficient), use_container_width=True, hide_index=True)

        scan_rows = st.session_state.get("scan_rows")
        with st.expander("COT-Diagnostik und aktuelle Index-Extreme"):
            if scan_rows:
                extreme_rows = extreme_scanner_table(scan_rows)
                if extreme_rows.empty:
                    st.info("Kein Asset liegt im eingestellten COT-Index-Extrembereich.")
                else:
                    st.dataframe(extreme_rows, use_container_width=True, hide_index=True)
            else:
                st.info("Keine COT-Scanergebnisse vorhanden.")

        with st.expander("Methodik und Grenzen"):
            st.markdown(
                """
- **COT:** historische Fortsetzung ähnlicher reiner Commercial-, Non-Commercial- und Retail-Nettopositionen.
- **Fundamental:** historische Preisreaktion nach ähnlichen isolierten Fundamentalregimen; keine COT-Daten im Modell.
- **Seasonality:** historische Fortsetzung ab der aktuell kalibrierten saisonalen Phase.
- **Unsicherheit:** Die Wilson-Untergrenze berücksichtigt Trefferquote und Stichprobengröße gemeinsam. Für die robuste Untergrenze zählt das schwächste bestätigende Modul.
- **Datenqualität:** Angezeigt wird nur der zeitliche Überschneidungsbereich der drei Datenhistorien.
- **Noch nicht enthalten:** ein echter gemeinsamer Point-in-Time-Backtest, bei dem alle drei damaligen Modellsignale gleichzeitig rekonstruiert werden. Die aktuelle Ansicht zeigt parallele Evidenz, keine validierte Dreifach-Strategie.
                """
            )
    else:
        st.info("Die Konfluenzanalyse wurde in dieser Sitzung noch nicht ausgeführt.")

elif page == "Price Seasonality":
    asset = assets[asset_name]
    st.subheader(f"Price Seasonality · {asset.name}")
    st.caption(
        "Robuste historische Jahres-Seasonality mit Medianpfad, Streuungsband und zeitlicher Kalibrierung. "
        "3D, 10D, 2W, 4W und 8W werden unabhängig mit früheren saisonalen Strukturen verglichen. Es werden keine manuell festgelegten Gewichte verwendet."
    )
    try:
        prices, price_source = fetch_prices(asset, force=force)
        latest = pd.to_datetime(prices["date"]).max()
        current_week = int(latest.isocalendar().week)
        history_window = st.selectbox(
            "Historie für die saisonale Referenzkurve",
            [5, 10, 15, None],
            index=2,
            format_func=lambda value: "Gesamte verfügbare Historie" if value is None else f"Letzte {value} Jahre",
        )
        calibration = calibrate_seasonality(prices, window_years=history_window, max_shift_weeks=8)
        st.write(
            f"Preisstand bis **{latest:%d.%m.%Y}** · Kalenderwoche **{current_week}** "
            f"· Quelle: `{price_source}`"
        )
        if not calibration.available:
            st.info(calibration.reason)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Historische Jahre", calibration.years)
            c2.metric("Pattern Similarity", fmt_pct(calibration.similarity))
            c3.metric("Kalibrierte Phase", f"KW {calibration.calibrated_week}", delta=calibration.status)
            c4.metric(
                "Seasonal Progress",
                f"{calibration.seasonal_progress * 100:.0f} %" if calibration.seasonal_progress is not None else "–",
                delta=calibration.progress_status,
            )

            st.markdown("### Unabhängige Pattern-Fenster")
            s1,s2,s3,s4,s5=st.columns(5)
            s1.metric("3D",fmt_pct(calibration.similarity_3d),delta=f"Phase {calibration.phase_3d:+d}W" if calibration.phase_3d is not None else None)
            s2.metric("10D",fmt_pct(calibration.similarity_10d),delta=f"Phase {calibration.phase_10d:+d}W" if calibration.phase_10d is not None else None)
            s3.metric("2W",fmt_pct(calibration.similarity_2w),delta=f"Phase {calibration.phase_2w:+d}W" if calibration.phase_2w is not None else None)
            s4.metric("4W",fmt_pct(calibration.similarity_4w),delta=f"Phase {calibration.phase_4w:+d}W" if calibration.phase_4w is not None else None)
            s5.metric("8W",fmt_pct(calibration.similarity_8w),delta=f"Phase {calibration.phase_8w:+d}W" if calibration.phase_8w is not None else None)
            st.metric("Seasonality-Stabilität",fmt_pct(calibration.stability))
            st.caption("Keine Gewichtung: Jedes Fenster bestimmt seine beste Phase separat. Die Konsensphase ist der Median der fünf unabhängigen Phasenschätzungen.")

            chart_cols = [
                "Median-Seasonality", "25. Perzentil", "75. Perzentil",
                "Kalibrierte Seasonality", "Aktueller Verlauf"
            ]
            chart = calibration.chart.set_index("Kalenderwoche")[chart_cols] * 100.0
            st.line_chart(
                chart,
                use_container_width=True,
                height=560,
                x_label="Kalenderwoche",
                y_label="Kumulierte Preisentwicklung seit Jahresbeginn (%)",
            )
            st.caption(
                "Der Medianpfad bildet die robuste historische Jahreskurve. Das 25.–75.-Perzentil zeigt die normale Streuung. "
                "Die kalibrierte Kurve verwendet die Konsensphase aus 3D, 10D, 2W, 4W und 8W. Kein Fenster erhält ein manuell festgelegtes Gewicht."
            )

            if calibration.lead_lag_weeks is not None:
                if calibration.lead_lag_weeks > 1:
                    st.info(
                        f"Der aktuelle Kursverlauf ähnelt derzeit eher der historischen **KW {calibration.calibrated_week}** "
                        f"und ist damit ungefähr **{calibration.lead_lag_weeks} Wochen vorausgelaufen**. "
                        "Die weitere saisonale Erwartung wird ab dieser kalibrierten Phase gelesen."
                    )
                elif calibration.lead_lag_weeks < -1:
                    st.info(
                        f"Der aktuelle Kursverlauf ähnelt derzeit eher der historischen **KW {calibration.calibrated_week}** "
                        f"und liegt ungefähr **{abs(calibration.lead_lag_weeks)} Wochen hinter der üblichen Seasonality**."
                    )
                else:
                    st.success("Der aktuelle Kursverlauf liegt zeitlich nahe an seiner historischen saisonalen Phase.")

            st.markdown("### Statistische Fortsetzung ab der kalibrierten Phase")
            if calibration.forecast.empty:
                st.info("Für die kalibrierte Phase ist keine vollständige 4W-/8W-/12W-Fortsetzung verfügbar.")
            else:
                forecast = calibration.forecast.copy()
                forecast["Trefferquote"] = forecast["Trefferquote"].map(fmt_pct)
                forecast["Median"] = forecast["Median"].map(fmt_return)
                forecast["Typischer Bewegungsbereich"] = calibration.forecast.apply(
                    lambda row: fmt_range(row["Q25"], row["Q75"]), axis=1
                )
                st.dataframe(
                    forecast[["Horizont", "Fälle", "Richtung", "Trefferquote", "Median", "Typischer Bewegungsbereich"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Die Fortsetzung beginnt nicht bei der aktuellen Kalenderwoche, sondern bei der ermittelten kalibrierten saisonalen Phase."
                )

            st.warning(
                "Die Kalibrierung ist eine historische Struktur- und Timinganalyse. COT- und Fundamentalresultate bleiben getrennt "
                "und werden noch nicht automatisch in die saisonale Erwartung eingerechnet."
            )
    except Exception as exc:
        st.error(f"Seasonality konnte nicht berechnet werden: {exc}")

elif page == "Fundamental-Labor (isoliert)":
    asset = assets[asset_name]
    st.subheader(f"Fundamental-Labor · {asset.name}")
    if asset.category == "Währungen":
        st.info("Währungsmodus: Es werden ausschließlich die historischen Makrodaten der Heimatwirtschaft dieser Währung analysiert. US-Daten, relative Spreads und COT-Daten fließen nicht ein.")
    st.warning(
        "Isolierter Testbetrieb: Hier werden ausschließlich Fundamentaldaten der jeweiligen Heimatwirtschaft und historische Preisreaktionen analysiert. "
        "COT-Positionen, COT-Index, Nettopositionsmuster und Positionsfluss werden weder geladen noch verwendet."
    )
    try:
        with st.spinner("Fundamentaldaten und Preishistorie werden geladen …"):
            prices, price_source = fetch_prices(asset, force=force)
            fundamental, coverage = load_fundamental_research(
                asset,
                prices,
                horizon_weeks=horizon,
                n_neighbors=net_cases,
                force=force,
            )
        coverage_level, coverage_text = coverage
        st.caption(
            f"Datenabdeckung für diese Testversion: **{coverage_level}** · {coverage_text} "
            f"· Preisquelle: {price_source}"
        )
        if not fundamental.available:
            st.error(fundamental.reason)
            if fundamental.missing_series:
                with st.expander("Fehlende oder nicht ladbare Datenreihen"):
                    for item in fundamental.missing_series:
                        st.write(f"- {item}")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset-Fundamental-Score", f"{fundamental.score:+.0f} / 100")
        c2.metric("Einordnung", fundamental.label)
        c3.metric("Modellvertrauen", fundamental.confidence)
        c4.metric(
            "Fundamental-Snapshot",
            fundamental.snapshot_date.strftime("%d.%m.%Y") if fundamental.snapshot_date is not None else "–",
        )

        st.markdown("### Makroregime")
        regime_view = fundamental.regime_table.copy()
        regime_view["Z-Score"] = regime_view["Z-Score"].map(lambda value: "–" if pd.isna(value) else f"{value:+.2f}")
        st.dataframe(regime_view, use_container_width=True, hide_index=True)

        st.markdown("### Fundamentale Treiber für dieses Asset")
        drivers = fundamental.driver_table.copy()
        drivers["Aktueller Messwert"] = drivers["Aktueller Messwert"].map(lambda value: f"{value:,.2f}")
        drivers["Z-Score"] = drivers["Z-Score"].map(lambda value: f"{value:+.2f}")
        drivers["Gewicht"] = drivers["Gewicht"].map(lambda value: f"{value:+.2f}")
        drivers["Beitrag"] = drivers["Beitrag"].map(lambda value: f"{value:+.2f}")
        for column in ["Beobachtungsperiode", "Als verfügbar behandelt ab"]:
            drivers[column] = pd.to_datetime(drivers[column], errors="coerce").dt.strftime("%d.%m.%Y")
        st.dataframe(
            drivers[[
                "Bereich", "Datenreihe", "Aktueller Messwert", "Einheit", "Z-Score",
                "Gewicht", "Beitrag", "Asset-Wirkung", "Beobachtungsperiode",
                "Als verfügbar behandelt ab", "FRED-ID",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(f"### Historisch ähnliche Fundamentalregime · Preis nach {horizon} Wochen")
        if fundamental.sample_size:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Historische Fälle", fundamental.sample_size)
            h2.metric("Gestiegen", f"{fundamental.up_count} · {fundamental.up_count/fundamental.sample_size:.1%}")
            h3.metric("Gefallen", f"{fundamental.down_count} · {fundamental.down_count/fundamental.sample_size:.1%}")
            h4.metric("Dominante Richtungsquote", fmt_pct(fundamental.hit_rate))
            st.write(
                f"Medianbewegung: **{fmt_return(fundamental.median_return)}** · "
                f"typischer Bewegungsbereich: **{fmt_range(fundamental.range_low, fundamental.range_high)}**"
            )
            analogs = fundamental.analogs.copy()
            analogs["Datum"] = pd.to_datetime(analogs["date"]).dt.strftime("%d.%m.%Y")
            analogs["Fundamental-Score"] = analogs["fundamental_score"].map(lambda value: f"{value:+.0f}")
            analogs["Ähnlichkeit"] = analogs["similarity"].map(lambda value: f"{value:.1f}")
            for weeks in (4, 8, 12):
                col = f"return_{weeks}w"
                analogs[f"Preis {weeks}W"] = analogs[col].map(fmt_return)
            st.dataframe(
                analogs[["Datum", "Fundamental-Score", "Ähnlichkeit", "Preis 4W", "Preis 8W", "Preis 12W"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Noch nicht genügend vollständige historische Analogfälle mit späteren Preisdaten verfügbar.")

        st.markdown("### Verlauf des isolierten Fundamental-Scores")
        chart = fundamental.score_history[["date", "fundamental_score"]].dropna().set_index("date")
        st.line_chart(chart, y="fundamental_score", use_container_width=True)

        with st.expander("Methodik und wichtige Einschränkungen"):
            st.write(
                "Die Datenreihen werden zuerst unabhängig von COT transformiert: Inflation und Wachstum als "
                "Vorjahresrate, NFP als monatliche Payroll-Veränderung, Zinsen und Arbeitslosenquote als Niveau. "
                "Anschließend werden rollierende Z-Scores berechnet und mit transparenten asset-spezifischen "
                "Vorzeichen und Gewichten zu einem Score von −100 bis +100 zusammengeführt."
            )
            st.write(
                "Für monatliche und quartalsweise Reihen werden konservative, näherungsweise Veröffentlichungslags "
                "verwendet. Dadurch wird ein offensichtlicher Blick in die Zukunft reduziert. Die kostenlose FRED-CSV "
                "enthält jedoch überwiegend heute bekannte und teilweise revidierte Daten. Diese Version ist daher ein "
                "Research-MVP und noch kein vollständig point-in-time-fähiger Backtest."
            )
            st.write(
                "Die historischen Analogfälle werden ausschließlich aus Fundamental-Z-Scores gewählt. Erst danach "
                "werden die Preisänderungen nach 4, 8 und 12 Wochen gemessen. COT-Daten sind an keinem Schritt beteiligt."
            )
        if fundamental.missing_series:
            with st.expander("Teilweise fehlende Datenreihen"):
                for item in fundamental.missing_series:
                    st.write(f"- {item}")
    except Exception as exc:
        st.exception(exc)

elif page == "Datenstatus":
    asset = assets[asset_name]
    try:
        market = load_market_data(asset, force=force)
        c1, c2 = st.columns(2)
        c1.metric("COT-Stichtag", market.cot_quality.latest_date.strftime("%d.%m.%Y") if market.cot_quality.latest_date else "–")
        c2.metric("Letzter Preistag", market.price_quality.latest_date.strftime("%d.%m.%Y") if market.price_quality.latest_date else "–")
        st.success("Daten sind aktuell und freigegeben.") if market.usable else st.error("Analyse gesperrt: Daten sind veraltet.")
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
            st.error("Analyse gesperrt, weil die Daten nicht aktuell sind.")
            st.stop()
        data = prepare_dataset(market.cot, market.prices, [horizon])
        valid = data.dropna(subset=NET_COLS + OI_RATIO_COLS).copy()
        if len(valid) < 80:
            st.warning("Für dieses Asset liegt nicht genügend vollständige Historie vor.")
            st.stop()
        current = valid.iloc[-1]
        report_date = pd.Timestamp(current["date"]).normalize()
        st.subheader(asset.name)
        st.caption(f"COT-Stichtag {report_date.strftime('%d.%m.%Y')} · {asset.category} · {asset.exchange}")

        indexed = add_cot_indices(valid)
        idx_now = indexed.dropna(subset=["commercial_cot_index", "retail_cot_index"]).iloc[-1]
        pure_net_research = analyze_pure_net_pattern(valid, horizon_weeks=horizon, n_neighbors=net_cases)
        commercial_research = analyze_group_net(valid, "Commercials", horizon_weeks=horizon, n_neighbors=net_cases, include_oi_context=include_oi_context)
        retail_research = analyze_group_net(valid, "Retail", horizon_weeks=horizon, n_neighbors=net_cases, include_oi_context=include_oi_context)
        overlap_research = analyze_overlap(valid, horizon_weeks=horizon, n_neighbors=net_cases, include_oi_context=include_oi_context)
        pure_net_timing = timing_for_result(pure_net_research, market.prices, report_date, horizon)
        commercial_timing = timing_for_result(commercial_research, market.prices, report_date, horizon)
        retail_timing = timing_for_result(retail_research, market.prices, report_date, horizon)
        overlap_timing = timing_for_result(overlap_research, market.prices, report_date, horizon)
        commercial_zones = build_reference_zones(valid, "Commercials", horizon_weeks=horizon)
        retail_zones = build_reference_zones(valid, "Retail", horizon_weeks=horizon)
        divergence = analyze_noncommercial_divergence(valid)

        tab_extreme, tab_net, tab_flow = st.tabs(["1 · Extremscanner", "2 · Netto-Level-Research", "3 · Positionsfluss"])
        with tab_extreme:
            c1, c2 = st.columns(2)
            c1.metric("Commercial COT-Index", f"{idx_now['commercial_cot_index']:.1f}")
            c1.caption(index_zone(float(idx_now["commercial_cot_index"])))
            c2.metric("Retail COT-Index", f"{idx_now['retail_cot_index']:.1f}")
            c2.caption(index_zone(float(idx_now["retail_cot_index"])))
            is_extreme = (
                idx_now["commercial_cot_index"] >= extreme_cutoff or idx_now["commercial_cot_index"] <= 100-extreme_cutoff or
                idx_now["retail_cot_index"] >= extreme_cutoff or idx_now["retail_cot_index"] <= 100-extreme_cutoff
            )
            if is_extreme:
                st.success("Mindestens eine Händlergruppe befindet sich aktuell im eingestellten COT-Index-Extrembereich.")
            else:
                st.info("Aktuell liegt keine der beiden Gruppen im eingestellten COT-Index-Extrembereich.")
            st.caption("Dieser Schritt ist nur ein Scanner. Die absolute Nettoposition wird erst im nächsten Tab historisch ausgewertet.")

        with tab_net:
            st.caption("Commercials und Retail werden zuerst vollständig getrennt ausgewertet. Erst danach wird ihre zeitgleiche Überschneidung untersucht.")
            pure_tab, commercial_tab, retail_tab, overlap_tab, levels_tab = st.tabs([
                "Reines Gesamtmuster", "Commercials einzeln", "Retail einzeln", "Zeitgleiche Überschneidung", "Referenzlevel"
            ])
            return_col = f"return_{horizon}w"
            with pure_tab:
                st.markdown("### Reines Nettopositionsmuster")
                st.caption("Verglichen werden ausschließlich Commercial-, Non-Commercial- und Retail-Nettopositionen. COT-Index und Open Interest beeinflussen die Mustersuche nicht.")
                show_group_research(pure_net_research, pure_net_timing, return_col)
                if pure_net_research.available:
                    current_vals = valid.iloc[-1]
                    st.caption(
                        f"Aktuelle Konstellation: Commercials {current_vals['commercial_net']:+,.0f} · "
                        f"Non-Commercials {current_vals['noncommercial_net']:+,.0f} · "
                        f"Retail {current_vals['nonreportable_net']:+,.0f} Kontrakte"
                    )
            with commercial_tab:
                st.markdown("### Commercial-Nettoposition relativ zum Preis")
                show_group_research(commercial_research, commercial_timing, return_col)
            with retail_tab:
                st.markdown("### Retail-Nettoposition relativ zum Preis")
                show_group_research(retail_research, retail_timing, return_col)
            with overlap_tab:
                st.markdown("### Commercial-/Retail-Überschneidung")
                st.caption("Hier werden historische Zeitpunkte gesucht, an denen beide aktuellen Netto-Level gleichzeitig ähnlich waren.")
                show_group_research(overlap_research, overlap_timing, return_col)
            with levels_tab:
                st.markdown("### Wiederkehrende institutionelle Positionierungszonen")
                st.caption("Die Zonen werden aus wiederkehrenden absoluten Netto-Leveln gebildet. ● markiert die aktuell besuchte Zone. Sie sind Positionierungsreferenzen, keine horizontalen Preislevel.")
                ctab, rtab = st.tabs(["Commercial-Referenzzonen", "Retail-Referenzzonen"])
                with ctab:
                    cframe = reference_zone_frame(commercial_zones)
                    st.dataframe(cframe, use_container_width=True, hide_index=True) if not cframe.empty else st.info("Noch keine stabilen Commercial-Referenzzonen verfügbar.")
                with rtab:
                    rframe = reference_zone_frame(retail_zones)
                    st.dataframe(rframe, use_container_width=True, hide_index=True) if not rframe.empty else st.info("Noch keine stabilen Retail-Referenzzonen verfügbar.")

        with tab_flow:
            if not divergence.get("available"):
                st.warning(str(divergence.get("description", "Flow nicht verfügbar.")))
            else:
                st.markdown("### Preis-/Positionsfluss-Matrix")
                st.caption("Long- und Short-Aufbau werden getrennt bewertet. Gemischte Situationen bleiben sichtbar und werden nicht pauschal als ‚keine Divergenz‘ verworfen.")
                rows = []
                state_labels = {
                    "bullish_divergence": "Bullische aktive Divergenz",
                    "bearish_divergence": "Bearische aktive Divergenz",
                    "bullish_accumulation_mixed": "Bullische Akkumulation + Short-Aufbau",
                    "bearish_distribution_mixed": "Bearischer Gegenfluss + Long-Aufbau",
                    "bullish_accumulation": "Bullische Akkumulation",
                    "bearish_distribution": "Bearischer Gegenfluss",
                    "bearish_trend_confirmation": "Bearische Trendbestätigung",
                    "bullish_trend_confirmation": "Bullische Trendbestätigung",
                    "position_reduction": "Positionsabbau",
                    "neutral": "Kein eindeutiges Signal",
                }
                flow_labels = {
                    "long_building": "Long-Aufbau dominiert",
                    "short_building": "Short-Aufbau dominiert",
                    "long_liquidation": "Long-Liquidation",
                    "short_covering": "Short-Covering",
                    "position_reduction": "Beidseitiger Positionsabbau",
                    "none": "Kein dominanter Flow",
                }
                for item in divergence.get("horizons", []):
                    rows.append({
                        "Zeitraum": f"{item['weeks']} Woche(n)",
                        "Preis": fmt_return(float(item["price_change"])),
                        "Long-Veränderung": f"{float(item['long_change']):+,.0f}",
                        "Short-Veränderung": f"{float(item['short_change']):+,.0f}",
                        "Flow-Schwerpunkt": flow_labels.get(str(item.get("dominant_flow", "none")), str(item.get("dominant_flow", "none"))),
                        "Dominanz": fmt_pct(float(item.get("dominance", 0))),
                        "Interpretation": state_labels.get(str(item.get("state", "neutral")), str(item.get("state", "neutral"))),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                with st.expander("Interpretation je Zeitraum"):
                    for item in divergence.get("horizons", []):
                        st.markdown(f"**{item['weeks']} Woche(n):** {item.get('explanation', '')}")
                st.caption("Eine reine aktive Divergenz bleibt streng gefiltert. Zusätzlich werden Akkumulation, Distribution und Trendbestätigung separat ausgewiesen.")
    except Exception as exc:
        st.exception(exc)
