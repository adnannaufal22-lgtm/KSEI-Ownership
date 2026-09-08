from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

from charts import (
    composition_bar,
    flow_bar,
    foreign_share_history,
    monthly_movement_chart,
    movement_breakdown_chart,
    residency_donut,
    ticker_change_bar,
    trend_chart,
)
from data_loader import folder_signature, load_excel, load_excel_folder
from data_processing import (
    build_quality_report,
    category_flow,
    composition_share,
    concentration_metrics,
    entity_monthly_movement,
    movement_pivot_table,
    snapshot_metrics,
    sort_ticker_snapshot,
    standardize_dataframe,
    ticker_snapshot,
)
from toolbar import render_global_toolbar
from utils import dataframe_to_excel_bytes, display_number, exact_number, format_pct, metric_delta


APP_DIR = Path(__file__).resolve().parent
BEI_DATA_DIR = APP_DIR / "BEI_Data"
CONFIG_PATH = APP_DIR / "schema_mapping.json"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


st.set_page_config(
    page_title="KSEI Ownership Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(f"<style>{(APP_DIR / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=3)
def load_dashboard_data(workbook_bytes: bytes, config_text: str):
    config = json.loads(config_text)
    raw, metadata = load_excel(workbook_bytes, config)
    standardized, mapping = standardize_dataframe(raw, config)
    quality = build_quality_report(raw, standardized, mapping, metadata)
    return raw, standardized, quality


@st.cache_data(show_spinner=False, max_entries=3)
def load_folder_dashboard_data(
    data_folder: str,
    data_signature: tuple[tuple[str, int, int], ...],
    config_text: str,
):
    config = json.loads(config_text)
    raw, metadata = load_excel_folder(data_folder, config)
    standardized, mapping = standardize_dataframe(raw, config)
    quality = build_quality_report(raw, standardized, mapping, metadata)
    return raw, standardized, quality


def section(kicker: str, heading: str) -> None:
    st.markdown(f'<div class="section-kicker">{kicker}</div><div class="section-heading">{heading}</div>', unsafe_allow_html=True)


def percent_delta(current: float, previous: float | None) -> str | None:
    if previous is None or pd.isna(previous) or pd.isna(current):
        return None
    return f"{current - previous:+.1f} pp"


def source_controls() -> tuple[bytes | None, str, tuple[tuple[str, int, int], ...]]:
    with st.sidebar:
        st.markdown("### Data source")
        uploaded = st.file_uploader("Temporarily use one Excel workbook", type=["xlsx", "xlsm"])
        if uploaded is not None:
            st.success(f"Using {uploaded.name}")
            return uploaded.getvalue(), uploaded.name, ()
        signature = folder_signature(BEI_DATA_DIR)
        if not signature:
            st.error("BEI_Data is empty. Add at least one monthly .xlsx file to continue.")
            st.stop()
        latest_file = signature[-1][0]
        st.caption(f"Using {len(signature)} monthly files from BEI_Data. Latest: {latest_file}")
        return None, f"BEI_Data · {len(signature)} files", signature


config_text = CONFIG_PATH.read_text(encoding="utf-8")
workbook_bytes, source_name, data_signature = source_controls()

try:
    with st.spinner("Reading and validating ownership data…"):
        if workbook_bytes is not None:
            raw_data, data, quality = load_dashboard_data(workbook_bytes, config_text)
        else:
            raw_data, data, quality = load_folder_dashboard_data(
                str(BEI_DATA_DIR),
                data_signature,
                config_text,
            )
except Exception as error:
    st.error(f"The ownership data could not be loaded: {error}")
    st.stop()

periods = sorted(pd.Timestamp(value) for value in data["date"].dropna().unique())
if not periods:
    st.error("No valid reporting dates were found in the workbook.")
    st.stop()
latest_period = periods[-1]

holder_link_target = st.query_params.get("holder")
if holder_link_target:
    holder_link_target = str(holder_link_target)
    if holder_link_target in set(data["investor_name"].dropna().unique()):
        st.session_state["ticker_filter"] = []
        st.session_state["holder_filter"] = [holder_link_target]
        st.session_state["movement_entity_type"] = "Holder"
        st.session_state["movement_holder"] = holder_link_target
        st.session_state["focus_movement_tab"] = True
        st.session_state.pop("active_filter_chips", None)
    st.query_params.clear()

toolbar_state = render_global_toolbar(data, periods, source_name)
selected_period = toolbar_state.selected_period
comparison_mode = toolbar_state.comparison_mode
comparison_period = toolbar_state.comparison_period
ticker_filter = toolbar_state.ticker_filter
holder_filter = toolbar_state.holder_filter
category_filter = toolbar_state.category_filter
residency_filter = toolbar_state.residency_filter
institution_filter = toolbar_state.institution_filter
sector_filter = toolbar_state.sector_filter
minimum_holding_pct = toolbar_state.minimum_holding_pct
display_mode = toolbar_state.display_mode
filtered_history = toolbar_state.filtered_history
current = toolbar_state.current
previous = toolbar_state.previous

if current.empty:
    st.warning("No observations match the selected period and filters. Broaden the filters or choose another report date.")
    st.stop()

format_number = lambda value: display_number(value, display_mode)
current_metrics = snapshot_metrics(current)
previous_metrics = snapshot_metrics(previous) if not previous.empty else None
comparison_label = comparison_period.strftime("%d %b %Y") if comparison_period is not None else "not available"

st.markdown(
    f'<div class="method-note">Comparison: <strong>{comparison_label}</strong> · <strong>{len(current):,}</strong> filtered records · Composition uses reported stake points and excludes unclassified rows. The workbook contains share counts, not market values.</div>',
    unsafe_allow_html=True,
)

if st.session_state.get("focus_movement_tab", False):
    movement_tab, overview_tab, flow_tab, securities_tab, detail_tab = st.tabs(
        ["Movement explorer", "Overview", "Flows & concentration", "Securities", "Detailed data"]
    )
else:
    overview_tab, flow_tab, securities_tab, movement_tab, detail_tab = st.tabs(
        ["Overview", "Flows & concentration", "Securities", "Movement explorer", "Detailed data"]
    )

with overview_tab:
    section("01 · Key metrics", "Ownership snapshot")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    prev = previous_metrics or {}
    k1.metric("Reported shares", format_number(current_metrics["reported_units"]), metric_delta(current_metrics["reported_units"], prev.get("reported_units"), display_mode))
    k2.metric("Foreign share", format_pct(current_metrics["foreign_share"]), percent_delta(current_metrics["foreign_share"], prev.get("foreign_share")))
    k3.metric("Domestic share", format_pct(current_metrics["domestic_share"]), percent_delta(current_metrics["domestic_share"], prev.get("domestic_share")))
    k4.metric("Institutional share", format_pct(current_metrics["institutional_share"]), percent_delta(current_metrics["institutional_share"], prev.get("institutional_share")))
    k5.metric("Distinct holders", f"{current_metrics['investors']:,.0f}", metric_delta(current_metrics["investors"], prev.get("investors"), "Exact"))
    k6.metric("Individual share", format_pct(current_metrics["individual_share"]), percent_delta(current_metrics["individual_share"], prev.get("individual_share")))

    st.markdown("<br>", unsafe_allow_html=True)
    section("02–03 · Trend & composition", "How reported ownership is distributed")
    chart_left, chart_right = st.columns([1.65, 1])
    with chart_left:
        trend_mode = st.radio("Trend view", ["Reported holdings", "Domestic vs Foreign", "Institutional vs Individual", "Investor categories"], horizontal=True, label_visibility="collapsed")
        st.plotly_chart(trend_chart(filtered_history, trend_mode), width="stretch", config=PLOT_CONFIG, key="ownership_trend")
    with chart_right:
        investor_composition = composition_share(current, "investor_category")
        st.plotly_chart(composition_bar(investor_composition.rename(columns={"investor_category": "investor_category"})), width="stretch", config=PLOT_CONFIG, key="investor_composition")

    section("04 · Residency", "Domestic versus foreign")
    r1, r2 = st.columns([.9, 1.6])
    with r1:
        residency_composition = composition_share(current, "domestic_foreign")
        st.plotly_chart(residency_donut(residency_composition), width="stretch", config=PLOT_CONFIG, key="residency_split")
    with r2:
        st.plotly_chart(foreign_share_history(filtered_history), width="stretch", config=PLOT_CONFIG, key="foreign_history")

with flow_tab:
    section("05 · Ownership change", f"Movement versus {comparison_label}")
    if previous.empty:
        st.info("No earlier comparison period is available for the selected date and comparison rule.")
    else:
        flow_mode = st.radio("Change metric", ["Reported shares", "Stake points"], horizontal=True)
        flow_metric = "ownership_units" if flow_mode == "Reported shares" else "ownership_pct"
        flow = category_flow(current, previous, flow_metric)
        st.plotly_chart(flow_bar(flow, flow_mode), width="stretch", config=PLOT_CONFIG, key="flow_chart")
        flow_table = flow[["investor_category", "previous", "current", "change", "pct_change", "share_of_total"]].rename(
            columns={"investor_category": "Investor category", "previous": "Previous", "current": "Current", "change": "Change", "pct_change": "% change", "share_of_total": "Share of total (%)"}
        )
        st.dataframe(
            flow_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Previous": st.column_config.NumberColumn(format="%.2f" if flow_mode == "Stake points" else "localized"),
                "Current": st.column_config.NumberColumn(format="%.2f" if flow_mode == "Stake points" else "localized"),
                "Change": st.column_config.NumberColumn(format="%+.2f" if flow_mode == "Stake points" else "localized"),
                "% change": st.column_config.NumberColumn(format="%+.1f%%"),
                "Share of total (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            },
        )

        section("07 · Rankings", "Largest security-level changes")
        snapshot = ticker_snapshot(current, previous)
        rank_by = st.radio("Rank by", ["Absolute change", "Percentage change"], horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ticker_change_bar(snapshot, "increase", rank_by), width="stretch", config=PLOT_CONFIG, key="ticker_increases")
        with c2:
            st.plotly_chart(ticker_change_bar(snapshot, "decrease", rank_by), width="stretch", config=PLOT_CONFIG, key="ticker_decreases")

    section("08 · Concentration", "Reported-holder concentration")
    concentration = concentration_metrics(current)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top holder share", format_pct(concentration["top_1"]))
    c2.metric("Top 3 holders", format_pct(concentration["top_3"]))
    c3.metric("Top 5 holders", format_pct(concentration["top_5"]))
    c4.metric("Reported-holder HHI", f"{concentration['hhi']:,.0f}" if pd.notna(concentration["hhi"]) else "—")
    st.caption("HHI is the sum of squared ownership shares on a 0–10,000 scale. Here it describes concentration among holders reported in the workbook, not the full shareholder register; unreported stakes are excluded.")

with securities_tab:
    section("06 · Security analysis", "Cross-sectional ownership monitor")
    snapshot = sort_ticker_snapshot(
        ticker_snapshot(current, previous),
        toolbar_state.sort_mode,
        toolbar_state.sort_ascending,
    )
    security_table = snapshot.rename(columns={
        "ticker": "Ticker",
        "security_name": "Security",
        "current_total_units": "Reported shares",
        "current_domestic_points": "Domestic stake points",
        "current_foreign_points": "Foreign stake points",
        "current_foreign_share_pct": "Foreign share (%)",
        "current_institutional_points": "Institutional stake points",
        "current_individual_points": "Individual stake points",
        "current_investors": "Distinct holders",
        "period_change": "Period change",
        "period_change_pct": "Period change (%)",
    })
    display_columns = ["Ticker", "Security", "Reported shares", "Domestic stake points", "Foreign stake points", "Foreign share (%)", "Institutional stake points", "Individual stake points", "Distinct holders", "Period change", "Period change (%)"]
    st.dataframe(
        security_table[display_columns],
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "Ticker": st.column_config.TextColumn(pinned=True),
            "Reported shares": st.column_config.NumberColumn(format="localized"),
            "Foreign share (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Period change": st.column_config.NumberColumn(format="localized"),
            "Period change (%)": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

with movement_tab:
    section("10 · Monthly movement explorer", "Track any stock or disclosed holder through time")
    movement_base = filtered_history
    m1, m2, m3 = st.columns([1, 2.2, 1.3])
    with m1:
        entity_type = st.radio("Analyze by", ["Stock", "Holder"], horizontal=True, key="movement_entity_type")
    entity_column = "ticker" if entity_type == "Stock" else "investor_name"
    with m2:
        if entity_type == "Stock":
            entity_options = sorted(movement_base["ticker"].dropna().unique())
            security_names = (
                movement_base[["ticker", "security_name"]]
                .drop_duplicates("ticker")
                .set_index("ticker")["security_name"]
                .to_dict()
            )
            if entity_options:
                if st.session_state.get("movement_stock") not in entity_options:
                    st.session_state["movement_stock"] = entity_options[0]
                selected_entity = st.selectbox(
                    "Search stock",
                    entity_options,
                    format_func=lambda value: f"{value} · {security_names.get(value, '')}",
                    key="movement_stock",
                )
            else:
                selected_entity = None
                st.selectbox("Search stock", [], index=None, disabled=True, placeholder="No matching stocks")
        else:
            entity_options = sorted(movement_base["investor_name"].dropna().unique())
            if entity_options:
                if st.session_state.get("movement_holder") not in entity_options:
                    st.session_state["movement_holder"] = entity_options[0]
                selected_entity = st.selectbox("Search holder", entity_options, key="movement_holder")
            else:
                selected_entity = None
                st.selectbox("Search holder", [], index=None, disabled=True, placeholder="No matching holders")
    with m3:
        movement_metric = st.radio(
            "Movement metric",
            ["Reported shares", "Stake points"],
            horizontal=True,
            index=0 if entity_type == "Stock" else 1,
            key=f"movement_metric_{entity_type.lower()}",
        )

    if not entity_options:
        st.info("No stocks or holders match the active classification and minimum-holding filters.")
    else:
        monthly_movement, movement_breakdown, counterparty_label = entity_monthly_movement(
            movement_base,
            entity_column,
            selected_entity,
        )
        latest_movement = monthly_movement.iloc[-1]
        prior_movement = monthly_movement.iloc[-2] if len(monthly_movement) > 1 else None
        mm1, mm2, mm3, mm4, mm5 = st.columns(5)
        mm1.metric("Latest reported shares", exact_number(latest_movement["ownership_units"]))
        mm2.metric(
            "Monthly share movement",
            exact_number(latest_movement["change_units"]),
            f"{latest_movement['change_pct']:+.1f}%" if pd.notna(latest_movement["change_pct"]) else None,
        )
        mm3.metric("Latest stake points", f"{latest_movement['stake_points']:,.2f}")
        mm4.metric(
            "Monthly stake movement",
            f"{latest_movement['stake_change']:+,.2f} pp" if pd.notna(latest_movement["stake_change"]) else "—",
        )
        mm5.metric(f"Distinct {counterparty_label.lower()}s", f"{latest_movement['counterparties']:,.0f}")

        st.markdown(
            '<div class="method-note">Monthly movement compares each available reporting date with the immediately preceding workbook period. For a holder spanning several stocks, stake points are the more comparable metric because raw share units differ by issuer.</div>',
            unsafe_allow_html=True,
        )
        mc1, mc2 = st.columns([1.25, 1])
        with mc1:
            st.plotly_chart(
                monthly_movement_chart(monthly_movement, f"{selected_entity} monthly movement", movement_metric),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"movement_total_{entity_type}_{selected_entity}_{movement_metric}",
            )
        with mc2:
            st.plotly_chart(
                movement_breakdown_chart(movement_breakdown, counterparty_label, movement_metric),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"movement_breakdown_{entity_type}_{selected_entity}_{movement_metric}",
            )

        st.markdown("**Monthly movement pivot**")
        pv1, pv2 = st.columns([1.25, 3])
        with pv1:
            pivot_view = st.selectbox(
                "Pivot values",
                ["Holding level", "Monthly change", "Monthly % change"],
                key=f"pivot_view_{entity_type.lower()}",
            )
        with pv2:
            st.caption(
                f"Rows are {counterparty_label.lower()} names; columns are reporting months. "
                "Rows are ranked by the latest month, using absolute movement for change views."
            )

        movement_pivot = movement_pivot_table(movement_breakdown, movement_metric, pivot_view)
        movement_pivot = movement_pivot.rename(columns={"counterparty": counterparty_label})
        movement_pivot_display = movement_pivot.copy()
        if counterparty_label == "Holder":
            movement_pivot_display[counterparty_label] = movement_pivot_display[counterparty_label].map(
                lambda holder: f"?holder={quote(str(holder), safe='')}&view=movement#holder={holder}"
            )
        if pivot_view == "Monthly % change":
            pivot_number_format = "%+.1f%%"
        elif movement_metric == "Reported shares":
            pivot_number_format = "localized"
        else:
            pivot_number_format = "%+.2f" if pivot_view == "Monthly change" else "%.2f"
        if counterparty_label == "Holder":
            pivot_column_config = {
                counterparty_label: st.column_config.LinkColumn(
                    "Holder",
                    pinned=True,
                    display_text=r".*#holder=(.*)",
                    help="Open this holder's movement analysis.",
                )
            }
        else:
            pivot_column_config = {counterparty_label: st.column_config.TextColumn(pinned=True)}
        pivot_column_config.update(
            {
                column: st.column_config.NumberColumn(format=pivot_number_format)
                for column in movement_pivot.columns
                if column != counterparty_label
            }
        )
        st.dataframe(
            movement_pivot_display,
            width="stretch",
            hide_index=True,
            height=min(540, max(190, 42 + len(movement_pivot) * 35)),
            column_config=pivot_column_config,
        )
        st.download_button(
            "Export movement pivot CSV",
            movement_pivot.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"movement_pivot_{entity_type.lower()}_{selected_entity}_{pivot_view.lower().replace(' ', '_').replace('%', 'pct')}.csv",
            mime="text/csv",
        )

        with st.expander("Monthly totals and long-form detail"):
            monthly_table = monthly_movement.rename(columns={
                "date": "Month",
                "ownership_units": "Reported shares",
                "change_units": "Monthly change",
                "change_pct": "Monthly change (%)",
                "stake_points": "Stake points",
                "stake_change": "Stake change (pp)",
                "stake_change_pct": "Stake change (%)",
                "counterparties": f"Distinct {counterparty_label.lower()}s",
            })
            st.markdown("**Monthly totals**")
            st.dataframe(
                monthly_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Month": st.column_config.DateColumn(format="MMM YYYY"),
                    "Reported shares": st.column_config.NumberColumn(format="localized"),
                    "Monthly change": st.column_config.NumberColumn(format="localized"),
                    "Monthly change (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Stake points": st.column_config.NumberColumn(format="%.2f"),
                    "Stake change (pp)": st.column_config.NumberColumn(format="%+.2f"),
                    "Stake change (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                    f"Distinct {counterparty_label.lower()}s": st.column_config.NumberColumn(format="localized"),
                },
            )

            st.markdown(f"**Monthly {counterparty_label.lower()} detail**")
            detail_movement = movement_breakdown.rename(columns={
                "date": "Month",
                "counterparty": counterparty_label,
                "ownership_units": "Reported shares",
                "change_units": "Monthly change",
                "change_pct": "Monthly change (%)",
                "stake_points": "Stake points",
                "stake_change": "Stake change (pp)",
                "stake_change_pct": "Stake change (%)",
            })
            detail_movement_display = detail_movement.copy()
            if counterparty_label == "Holder":
                detail_movement_display[counterparty_label] = detail_movement_display[counterparty_label].map(
                    lambda holder: f"?holder={quote(str(holder), safe='')}&view=movement#holder={holder}"
                )
                detail_holder_config = st.column_config.LinkColumn(
                    "Holder",
                    pinned=True,
                    display_text=r".*#holder=(.*)",
                    help="Open this holder's movement analysis.",
                )
            else:
                detail_holder_config = st.column_config.TextColumn(pinned=True)
            st.dataframe(
                detail_movement_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Month": st.column_config.DateColumn(format="MMM YYYY"),
                    counterparty_label: detail_holder_config,
                    "Reported shares": st.column_config.NumberColumn(format="localized"),
                    "Monthly change": st.column_config.NumberColumn(format="localized"),
                    "Monthly change (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Stake points": st.column_config.NumberColumn(format="%.2f"),
                    "Stake change (pp)": st.column_config.NumberColumn(format="%+.2f"),
                    "Stake change (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )

with detail_tab:
    section("09 · Detailed data", "Filtered ownership records")
    table_search = st.text_input("Search investor, ticker, or issuer", placeholder="Search within the selected report date")
    table_data = current.copy()
    if table_search:
        query = table_search.strip()
        mask = (
            table_data["investor_name"].astype(str).str.contains(query, case=False, na=False)
            | table_data["ticker"].astype(str).str.contains(query, case=False, na=False)
            | table_data["security_name"].astype(str).str.contains(query, case=False, na=False)
        )
        table_data = table_data[mask]
    table_columns = ["date", "ticker", "security_name", "investor_name", "investor_category", "domestic_foreign", "institutional_individual", "nationality", "domicile", "ownership_scripless", "ownership_scrip", "ownership_units", "ownership_pct"]
    export_data = table_data[table_columns].copy()
    st.caption(f"{len(export_data):,} filtered records · numerical precision is retained in downloads")
    st.dataframe(
        export_data,
        width="stretch",
        hide_index=True,
        height=540,
        column_config={
            "date": st.column_config.DateColumn(format="DD MMM YYYY", pinned=True),
            "ticker": st.column_config.TextColumn(pinned=True),
            "ownership_scripless": st.column_config.NumberColumn(format="localized"),
            "ownership_scrip": st.column_config.NumberColumn(format="localized"),
            "ownership_units": st.column_config.NumberColumn(format="localized"),
            "ownership_pct": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Export filtered CSV", export_data.to_csv(index=False).encode("utf-8-sig"), file_name=f"ownership_{pd.Timestamp(selected_period):%Y%m%d}.csv", mime="text/csv", width="stretch")
    d2.download_button("Export filtered Excel", dataframe_to_excel_bytes(export_data), file_name=f"ownership_{pd.Timestamp(selected_period):%Y%m%d}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

with st.expander("Data Quality", expanded=False):
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("Earliest date", quality["earliest_date"].strftime("%d %b %Y"))
    q2.metric("Latest date", quality["latest_date"].strftime("%d %b %Y"))
    q3.metric("Observations", f"{quality['observations']:,}")
    q4.metric("Securities", f"{quality['securities']:,}")
    q5.metric("Periods", f"{quality['periods']:,}")
    q6.metric("Duplicates removed", f"{quality['duplicates_removed']:,}")
    st.markdown(f"**Source interpretation.** `{quality['data_sheet']}` · {quality.get('source_files', 1)} source files · {quality['source_columns']} populated source columns · {quality['investors']:,} distinct holder names · minimum observed stake {quality['minimum_observed_stake']:.2f}% · {quality['component_mismatches']} rows where total holdings do not equal scripless plus scrip holdings.")
    st.caption(
        f"Holder identity normalization consolidated {quality['holder_aliases_merged']:,} name variants "
        f"across {quality['holder_alias_groups']:,} alias groups. "
        f"{quality['holder_continuity_pairs']:,} close-name pairs were confirmed by identical shares across adjacent months."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values in important standardized fields**")
        missing_table = pd.DataFrame([{"Field": key, "Missing rows": value} for key, value in quality["missing_important"].items()])
        st.dataframe(missing_table, width="stretch", hide_index=True)
    with right:
        st.markdown("**Inferred schema mapping**")
        mapping_table = pd.DataFrame([{"Standard field": key, "Workbook column": value} for key, value in quality["mapping"].items()])
        st.dataframe(mapping_table, width="stretch", hide_index=True)
    st.markdown("**Holder alias audit (sample)**")
    alias_table = pd.DataFrame(
        [
            {
                "Canonical holder": item["canonical"],
                "Merged variants": " | ".join(item["aliases"]),
                "Method": item["method"],
            }
            for item in quality["holder_alias_examples"]
        ]
    )
    st.dataframe(alias_table, width="stretch", hide_index=True)
    st.caption("KSEI type codes are normalized as ID Individual, CP Corporate, MF Mutual Funds, IB Financial Institution, IS Insurance, SC Securities Company, PF Pension Funds, FD Foundation, and OT Other. Both A and F are normalized to Foreign because the workbook changes coding convention after the first period.")
