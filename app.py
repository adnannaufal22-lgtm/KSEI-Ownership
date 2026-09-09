from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from charts import (
    movement_breakdown_chart,
    monthly_movement_chart,
    ownership_movement_lines,
    scrip_vs_scripless_chart,
    stacked_area_line_chart,
)
from data_loader import (
    folder_signature,
    load_classification_folder,
    load_excel_folder,
    load_type_folder,
)
from data_processing import (
    classification_stock_history,
    entity_monthly_movement,
    historical_pivot,
    monthly_percentage_change_pivot,
    selected_ownership_history,
    standardize_dataframe,
    type_residency_history,
    type_stock_summary,
)
from monthly_changes import (
    build_monthly_change_detail,
    owner_change_summary,
    stock_change_summary,
)
from utils import compact_number


APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = APP_DIR / "BEI_Data"
OWNERSHIP_DIR = DATA_ROOT / "1% Ownership"
CLASSIFICATION_DIR = DATA_ROOT / "Classification"
TYPE_DIR = DATA_ROOT / "Type"
CONFIG_PATH = APP_DIR / "schema_mapping.json"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}
LOGGER = logging.getLogger("ksei_dashboard")


st.set_page_config(
    page_title="KSEI Ownership Dashboard",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    f"<style>{(APP_DIR / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=4)
def load_ownership_dataset(
    folder: str,
    signature: tuple[tuple[str, int, int], ...],
    config_text: str,
) -> tuple[pd.DataFrame, dict]:
    config = json.loads(config_text)
    raw, metadata = load_excel_folder(folder, config)
    standardized, mapping = standardize_dataframe(raw, config)
    metadata["mapping"] = mapping
    metadata["holder_alias_audit"] = standardized.attrs.get("holder_alias_audit", {})
    return standardized, metadata


@st.cache_data(show_spinner=False, max_entries=4)
def load_classification_dataset(
    folder: str,
    signature: tuple[tuple[str, int, int], ...],
    config_text: str,
) -> tuple[pd.DataFrame, dict]:
    return load_classification_folder(folder, json.loads(config_text))


@st.cache_data(show_spinner=False, max_entries=4)
def load_type_dataset(
    folder: str,
    signature: tuple[tuple[str, int, int], ...],
    config_text: str,
) -> tuple[pd.DataFrame, dict]:
    return load_type_folder(folder, json.loads(config_text))


@st.cache_data(show_spinner=False, max_entries=4)
def monthly_change_analysis(
    ownership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive reusable monthly activity tables from normalized ownership data."""
    detail = build_monthly_change_detail(ownership)
    return detail, stock_change_summary(detail), owner_change_summary(detail)


def empty_ownership_data() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "security_name",
            "investor_name",
            "ownership_units",
            "ownership_pct",
        ]
    )


def load_sources() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict, pd.DataFrame, dict]:
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    ownership_metadata: dict = {"source_files": 0, "issues": []}
    classification_metadata: dict = {"source_files": 0, "issues": []}
    type_metadata: dict = {"source_files": 0, "issues": []}

    try:
        ownership_data, ownership_metadata = load_ownership_dataset(
            str(OWNERSHIP_DIR),
            folder_signature(OWNERSHIP_DIR),
            config_text,
        )
    except Exception as error:
        LOGGER.exception("Could not load 1%% Ownership data")
        ownership_data = empty_ownership_data()
        ownership_metadata["issues"] = [str(error)]

    try:
        classification_data, classification_metadata = load_classification_dataset(
            str(CLASSIFICATION_DIR),
            folder_signature(CLASSIFICATION_DIR),
            config_text,
        )
    except Exception as error:
        LOGGER.exception("Could not load Classification data")
        classification_data = pd.DataFrame()
        classification_metadata["issues"] = [str(error)]

    try:
        type_data, type_metadata = load_type_dataset(
            str(TYPE_DIR),
            folder_signature(TYPE_DIR),
            config_text,
        )
    except Exception as error:
        LOGGER.exception("Could not load Type data")
        type_data = pd.DataFrame()
        type_metadata["issues"] = [str(error)]

    for dataset_name, metadata in (
        ("1% Ownership", ownership_metadata),
        ("Classification", classification_metadata),
        ("Type", type_metadata),
    ):
        for issue in metadata.get("issues", []):
            LOGGER.warning("%s: %s", dataset_name, issue)

    return (
        ownership_data,
        ownership_metadata,
        classification_data,
        classification_metadata,
        type_data,
        type_metadata,
    )


def section(kicker: str, heading: str, note: str | None = None) -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div>'
        f'<div class="section-heading">{heading}</div>',
        unsafe_allow_html=True,
    )
    if note:
        st.caption(note)


def table_heading(heading: str, separated: bool = False) -> None:
    heading_class = "table-heading table-heading-separated" if separated else "table-heading"
    st.markdown(
        f'<div class="{heading_class}">{escape(heading)}</div>',
        unsafe_allow_html=True,
    )


def render_pivot(
    pivot: pd.DataFrame,
    source_row_column: str,
    row_label: str,
    number_format: str,
    key: str,
    holder_links: bool = False,
    compact: bool = False,
    heatmap: bool = False,
    append_delta: bool = False,
) -> None:
    if pivot.empty:
        st.info("No pivot data is available for this selection.")
        return
    view = pivot.rename(columns={source_row_column: row_label}).copy()
    value_columns = [column for column in view.columns if column != row_label]

    def display_value(value: object) -> str:
        if pd.isna(value):
            return ""
        numeric = float(value)
        if number_format == "comma":
            return f"{numeric:,.0f}"
        if number_format == "signed_comma":
            return "0" if abs(numeric) < 0.5 else f"{numeric:+,.0f}"
        if number_format == "signed_pct":
            return "0.00%" if abs(numeric) <= 1e-9 else f"{numeric:+.2f}%"
        if number_format == "%.2f%%":
            return f"{numeric:.2f}%"
        if number_format == "%+.2f":
            return f"{numeric:+.2f}"
        return str(value)

    def movement_class(current: object, previous: object) -> str:
        if pd.isna(current) or pd.isna(previous):
            return "change-missing"
        current_number = float(current)
        previous_number = float(previous)
        tolerance = 1e-9 * max(1.0, abs(current_number), abs(previous_number))
        if abs(current_number - previous_number) <= tolerance:
            return "change-flat"
        return "change-up" if current_number > previous_number else "change-down"

    numeric_values = pd.to_numeric(
        pd.Series(view[value_columns].to_numpy().ravel()),
        errors="coerce",
    ).dropna()
    heatmap_scale = float(numeric_values.abs().max()) if not numeric_values.empty else 0.0

    def heatmap_style(value: object) -> str:
        if pd.isna(value):
            return ""
        numeric = float(value)
        if abs(numeric) <= 1e-9:
            return "background-color:rgba(245,166,35,.075);color:#8998A8;"
        ratio = min(abs(numeric) / heatmap_scale, 1.0) if heatmap_scale else 0.0
        opacity = 0.07 + 0.16 * ratio**0.5
        rgb = "0,199,129" if numeric > 0 else "255,90,82"
        color = "#00C781" if numeric > 0 else "#FF5A52"
        return f"background-color:rgba({rgb},{opacity:.3f});color:{color};"

    header_columns = [row_label, *value_columns]
    if append_delta:
        header_columns.extend(["Δ Shares", "Δ %"])
    header_cells = "".join(f"<th>{escape(str(column))}</th>" for column in header_columns)
    body_rows: list[str] = []
    for _, row in view.iterrows():
        raw_label = str(row[row_label])
        label_text = escape(raw_label)
        if holder_links:
            holder_query = escape(quote(raw_label, safe=""), quote=True)
            label_text = (
                f'<a href="/?holder={holder_query}" target="_top" '
                f'title="Analyze {escape(raw_label, quote=True)}">{label_text}</a>'
            )
        cells = [f'<td class="pivot-row-label">{label_text}</td>']
        previous: object = pd.NA
        for column in value_columns:
            current = row[column]
            css_class = "heatmap-cell" if heatmap else movement_class(current, previous)
            inline_style = heatmap_style(current) if heatmap else ""
            cells.append(
                f'<td class="pivot-value {css_class}" style="{inline_style}">'
                f"{escape(display_value(current))}</td>"
            )
            previous = current
        if append_delta:
            current = row[value_columns[-1]] if value_columns else pd.NA
            prior = row[value_columns[-2]] if len(value_columns) > 1 else pd.NA
            if pd.isna(current) or pd.isna(prior):
                delta_shares: object = pd.NA
                delta_pct: object = pd.NA
                delta_class = "change-missing"
            else:
                delta_shares = float(current) - float(prior)
                delta_pct = (
                    delta_shares / abs(float(prior)) * 100
                    if abs(float(prior)) > 1e-9
                    else pd.NA
                )
                delta_class = movement_class(delta_shares, 0.0)
            cells.append(
                f'<td class="pivot-value pivot-delta {delta_class}">'
                f"{escape(display_value(delta_shares) if pd.isna(delta_shares) else ('0' if abs(float(delta_shares)) < .5 else f'{float(delta_shares):+,.0f}'))}</td>"
            )
            cells.append(
                f'<td class="pivot-value pivot-delta {delta_class}">'
                f"{'' if pd.isna(delta_pct) else ('0.00%' if abs(float(delta_pct)) <= 1e-9 else f'{float(delta_pct):+.2f}%')}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    container_class = "pivot-scroll compact-pivot" if compact else "pivot-scroll"
    table_html = (
        f'<div class="{container_class}" data-pivot-key="{escape(key, quote=True)}">'
        '<table class="movement-pivot">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def selection_metrics(
    data: pd.DataFrame,
    analyze_by: str,
    selected_entity: str,
) -> dict[str, object]:
    """Build presentation-only summary metrics from the active ownership slice."""
    entity_column = "ticker" if analyze_by == "Stock" else "investor_name"
    counterparty_column = "investor_name" if analyze_by == "Stock" else "ticker"
    scoped = data[data[entity_column].eq(selected_entity)].dropna(subset=["date"]).copy()
    if scoped.empty:
        return {
            "reported_shares": 0.0,
            "delta": None,
            "delta_pct": None,
            "counterparties": 0,
            "largest_stake": None,
            "months": 0,
            "latest_period": "No data",
        }

    dates = sorted(pd.Timestamp(value) for value in scoped["date"].unique())
    latest = scoped[scoped["date"].eq(dates[-1])]
    monthly_totals = scoped.groupby("date", observed=True)["ownership_units"].sum()
    current_total = float(monthly_totals.loc[dates[-1]])
    previous_total = float(monthly_totals.loc[dates[-2]]) if len(dates) > 1 else None
    delta = current_total - previous_total if previous_total is not None else None
    delta_pct = (
        delta / abs(previous_total) * 100
        if delta is not None and previous_total not in (None, 0)
        else None
    )
    largest_stake = pd.to_numeric(latest["ownership_pct"], errors="coerce").max()
    return {
        "reported_shares": current_total,
        "delta": delta,
        "delta_pct": delta_pct,
        "counterparties": int(latest[counterparty_column].nunique()),
        "largest_stake": float(largest_stake) if pd.notna(largest_stake) else None,
        "months": len(dates),
        "latest_period": dates[-1].strftime("%b-%y").upper(),
    }


def metric_delta_label(delta: object, delta_pct: object) -> str | None:
    if delta is None or pd.isna(delta):
        return None
    numeric_delta = float(delta)
    sign = "+" if numeric_delta > 0 else "−" if numeric_delta < 0 else ""
    compact_delta = compact_number(abs(numeric_delta), decimals=1)
    if delta_pct is None or pd.isna(delta_pct):
        return f"{sign}{compact_delta}"
    return f"{sign}{compact_delta} ({float(delta_pct):+.2f}%)"


def terminal_number(value: object, decimals: int = 2) -> str:
    return (
        compact_number(value, decimals=decimals)
        .replace(" tn", "T")
        .replace(" bn", "B")
        .replace(" mn", "M")
        .replace(" k", "K")
    )


def render_terminal_header(
    ticker: str,
    company_name: str,
    analyze_by: str,
    metrics: dict[str, object],
) -> None:
    counterparty_label = "HOLDERS" if analyze_by == "Stock" else "STOCKS"
    largest_value = metrics["largest_stake"]
    delta = metrics["delta"]
    delta_class = "terminal-positive" if delta is not None and float(delta) > 0 else "terminal-negative" if delta is not None and float(delta) < 0 else "terminal-flat"
    delta_label = metric_delta_label(delta, metrics["delta_pct"]) or "—"
    st.markdown(
        '<div class="terminal-header">'
        '<div class="terminal-identity">'
        f'<span class="terminal-ticker">{escape(str(ticker))}</span>'
        f'<span class="terminal-company">{escape(str(company_name).upper())}</span>'
        f'<span class="terminal-mode">{escape(analyze_by.upper())}</span>'
        '</div>'
        '<div class="terminal-stats">'
        f'<span><b>SHARES</b><strong>{escape(terminal_number(metrics["reported_shares"]))}</strong></span>'
        f'<span><b>{counterparty_label}</b><strong>{int(metrics["counterparties"]):,}</strong></span>'
        f'<span><b>MAX</b><strong>{"—" if largest_value is None else f"{float(largest_value):.2f}%"}</strong></span>'
        f'<span><b>PERIOD</b><strong>{escape(str(metrics["latest_period"]))}</strong></span>'
        f'<span><b>HISTORY</b><strong>{int(metrics["months"])}M</strong></span>'
        f'<span><b>Δ SHARES</b><strong class="{delta_class}">{escape(delta_label)}</strong></span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_latest_snapshot(
    history: pd.DataFrame,
    row_label: str,
    analyze_by: str,
) -> None:
    dates = sorted(pd.Timestamp(value) for value in history["date"].dropna().unique())
    latest_date = dates[-1]
    latest_all = history[history["date"].eq(latest_date)]
    latest = latest_all.nlargest(5, "ownership_units")
    total = float(latest["ownership_units"].sum())
    snapshot_label = row_label.lower() if row_label.lower().endswith("s") else f"{row_label.lower()}s"

    def linked_label(label: str) -> str:
        label_html = escape(label)
        if analyze_by == "Stock":
            holder_query = escape(quote(label, safe=""), quote=True)
            return f'<a href="/?holder={holder_query}" target="_top">{label_html}</a>'
        return label_html

    rows: list[str] = []
    for position, (_, row) in enumerate(latest.iterrows(), start=1):
        label = str(row["series_label"])
        ownership_pct = row.get("ownership_pct")
        percentage = "" if pd.isna(ownership_pct) else f"{float(ownership_pct):.2f}%"
        rows.append(
            '<div class="snapshot-row">'
            f'<span class="snapshot-rank">{position}</span>'
            f'<span class="snapshot-name" title="{escape(label, quote=True)}">{linked_label(label)}</span>'
            '<span class="snapshot-value">'
            f'<strong>{terminal_number(row["ownership_units"])}</strong>'
            f'<small>{percentage}</small></span></div>'
        )

    mover_rows: list[str] = []
    if len(dates) > 1:
        previous_date = dates[-2]
        current_values = latest_all.groupby("series_label", observed=True)["ownership_units"].sum(min_count=1)
        previous_values = history[history["date"].eq(previous_date)].groupby("series_label", observed=True)["ownership_units"].sum(min_count=1)
        movement = pd.concat([current_values.rename("current"), previous_values.rename("previous")], axis=1).dropna()
        movement["delta"] = movement["current"] - movement["previous"]
        movement = movement[movement["delta"].abs().gt(.5)].assign(abs_delta=lambda frame: frame["delta"].abs()).nlargest(3, "abs_delta")
        for label, mover in movement.iterrows():
            delta_value = float(mover["delta"])
            direction = "↑" if delta_value > 0 else "↓"
            direction_class = "mover-up" if delta_value > 0 else "mover-down"
            mover_rows.append(
                f'<div class="mover-row {direction_class}"><span class="mover-arrow">{direction}</span>'
                f'<span class="mover-name" title="{escape(str(label), quote=True)}">{linked_label(str(label))}</span>'
                f'<strong>{"+" if delta_value > 0 else "−"}{terminal_number(abs(delta_value))}</strong></div>'
            )
    if not mover_rows:
        mover_rows.append('<div class="no-movers">NO REPORTED CHANGE</div>')

    st.markdown(
        '<div class="snapshot-card">'
        '<div class="snapshot-card-header">'
        f'<div><span class="period-chip">{latest_date:%b-%y}</span><h3>TOP {escape(snapshot_label.upper())}</h3></div>'
        f'<span class="snapshot-total">TOP 5&nbsp;&nbsp;{terminal_number(total)}</span></div>'
        f'<div class="snapshot-list">{"".join(rows)}</div>'
        '<div class="movers-header">MONTHLY MOVERS</div>'
        f'<div class="movers-list">{"".join(mover_rows)}</div></div>',
        unsafe_allow_html=True,
    )


def render_dashboard_masthead(
    analyze_by: str,
    selected_entity: str,
    ownership_metadata: dict,
    classification_metadata: dict,
    type_metadata: dict,
    latest_period: object,
) -> None:
    st.markdown(
        '<div class="dashboard-masthead">'
        '<div class="dashboard-heading">'
        '<span>KSEI OWNERSHIP</span>'
        f'<strong>— {escape(str(selected_entity))}</strong>'
        '</div>'
        '<div class="dashboard-coverage">'
        f'<span>MODE <b>{escape(analyze_by.upper())}</b></span>'
        f'<span>OWN <b>{int(ownership_metadata.get("source_files", 0))}</b></span>'
        f'<span>CLASS <b>{int(classification_metadata.get("source_files", 0))}</b></span>'
        f'<span>TYPE <b>{int(type_metadata.get("source_files", 0))}</b></span>'
        f'<span>UPDATED <b>{escape(str(latest_period))}</b></span>'
        '<span class="coverage-connected"><i></i>DATA CONNECTED</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def movement_cell_style(value: object) -> str:
    if pd.isna(value):
        return "color:#8998A8;"
    numeric = float(value)
    if numeric > 1e-9:
        return "color:#00C781;background-color:rgba(0,199,129,.075);"
    if numeric < -1e-9:
        return "color:#FF5A52;background-color:rgba(255,90,82,.075);"
    return "color:#8998A8;"


def render_activity_dataframe(
    data: pd.DataFrame,
    key: str,
    formatters: dict[str, object],
    directional_columns: list[str],
    name_columns: list[str],
    selectable: bool = False,
    max_height: int = 460,
) -> list[int]:
    """Render a compact terminal table and return any selected row positions."""
    if data.empty:
        st.info("No ownership changes are available for this comparison.")
        return []
    view = data.reset_index(drop=True)
    styled = view.style.format(formatters, na_rep="—")
    for column in directional_columns:
        if column in view:
            styled = styled.map(movement_cell_style, subset=[column])
    for column in name_columns:
        if column in view:
            styled = styled.set_properties(
                subset=[column],
                **{"color": "#3182F6", "font-weight": "650"},
            )
    height = min(max_height, 39 + 35 * len(view))
    if selectable:
        event = st.dataframe(
            styled,
            width="stretch",
            height=height,
            hide_index=True,
            row_height=34,
            key=key,
            on_select="rerun",
            selection_mode="single-row",
        )
        return list(event.selection.rows)
    st.dataframe(
        styled,
        width="stretch",
        height=height,
        hide_index=True,
        row_height=34,
        key=key,
    )
    return []


def render_monthly_change_kpis(
    stock_summary: pd.DataFrame,
    detail: pd.DataFrame,
    current_date: pd.Timestamp,
    previous_date: pd.Timestamp,
) -> None:
    increases = stock_summary[stock_summary["net_change"].gt(0.5)]
    decreases = stock_summary[stock_summary["net_change"].lt(-0.5)]
    largest_increase = (
        increases.loc[increases["net_change"].idxmax()] if not increases.empty else None
    )
    largest_decrease = (
        decreases.loc[decreases["net_change"].idxmin()] if not decreases.empty else None
    )

    def change_card(label: str, row: pd.Series | None, css_class: str) -> str:
        if row is None:
            return (
                '<div class="monthly-kpi"><span>' + escape(label) + '</span>'
                '<strong>—</strong><small>NO NET CHANGE</small></div>'
            )
        value = float(row["net_change"])
        sign = "+" if value > 0 else "−"
        return (
            '<div class="monthly-kpi"><span>' + escape(label) + '</span>'
            f'<strong>{escape(str(row["stock"]))}</strong>'
            f'<small class="{css_class}">{sign}{terminal_number(abs(value))} SHARES</small></div>'
        )

    st.markdown(
        '<div class="monthly-period-strip">'
        f'<strong>{current_date:%B %Y}</strong>'
        f'<span>COMPARED WITH {previous_date:%B %Y}</span>'
        '<em>REPORTED KSEI SNAPSHOT CHANGES</em></div>'
        '<div class="monthly-kpi-grid">'
        '<div class="monthly-kpi"><span>STOCKS CHANGED</span>'
        f'<strong>{stock_summary["stock"].nunique():,}</strong>'
        '<small>WITH REPORTED MOVEMENT</small></div>'
        '<div class="monthly-kpi"><span>ACTIVE OWNERS</span>'
        f'<strong>{detail["owner"].nunique():,}</strong>'
        '<small>ACROSS ALL STOCKS</small></div>'
        f'{change_card("LARGEST INCREASE", largest_increase, "terminal-positive")}'
        f'{change_card("LARGEST DECREASE", largest_decrease, "terminal-negative")}'
        '</div>',
        unsafe_allow_html=True,
    )


with st.spinner("Loading monthly ownership history…"):
    (
        ownership_data,
        ownership_metadata,
        classification_data,
        classification_metadata,
        type_data,
        type_metadata,
    ) = load_sources()


stock_names: dict[str, str] = {}
if not ownership_data.empty:
    stock_names.update(
        ownership_data[["ticker", "security_name"]]
        .dropna(subset=["ticker"])
        .drop_duplicates("ticker")
        .set_index("ticker")["security_name"]
        .fillna("")
        .astype(str)
        .to_dict()
    )
if not classification_data.empty:
    stock_names.update(
        classification_data[["ticker", "security_name"]]
        .dropna(subset=["ticker"])
        .drop_duplicates("ticker")
        .set_index("ticker")["security_name"]
        .fillna("")
        .astype(str)
        .to_dict()
    )
stock_options = sorted(
    set(stock_names)
    | (set(classification_data["ticker"].dropna().astype(str)) if "ticker" in classification_data else set())
    | (set(type_data["ticker"].dropna().astype(str)) if "ticker" in type_data else set())
)
owner_options = (
    sorted(ownership_data["investor_name"].dropna().astype(str).unique())
    if "investor_name" in ownership_data
    else []
)

holder_link_target = st.query_params.get("holder")
if holder_link_target:
    holder_link_target = str(holder_link_target)
    if holder_link_target in owner_options:
        st.session_state["analysis_mode"] = "Owner"
        st.session_state["selected_owner"] = holder_link_target
    st.query_params.clear()

st.markdown(
    '<div class="dashboard-brandbar">'
    '<span class="dashboard-wordmark">KSEI</span>'
    '<span>OWNERSHIP DASHBOARD</span>'
    '<em>INDONESIA CAPITAL MARKET MONITOR</em>'
    '</div>',
    unsafe_allow_html=True,
)

selector_mode_column, selector_entity_column = st.columns([1, 4], gap="medium")
with selector_mode_column:
    analyze_by = st.selectbox(
        "Analyze By",
        ["Stock", "Owner"],
        key="analysis_mode",
    )

with selector_entity_column:
    if analyze_by == "Stock":
        if not stock_options:
            st.error("No stock codes were found in the ownership folders.")
            st.stop()
        if st.session_state.get("selected_stock") not in stock_options:
            st.session_state["selected_stock"] = "AADI" if "AADI" in stock_options else stock_options[0]
        selected_entity = st.selectbox(
            "Stock",
            stock_options,
            key="selected_stock",
            format_func=lambda ticker: f"{ticker} · {stock_names.get(ticker, '')}".rstrip(" ·"),
        )
    else:
        if not owner_options:
            st.error("No individual owners were found in the 1% Ownership data.")
            st.stop()
        if st.session_state.get("selected_owner") not in owner_options:
            st.session_state["selected_owner"] = owner_options[0]
        selected_entity = st.selectbox(
            "Owner",
            owner_options,
            key="selected_owner",
        )

active_metrics = selection_metrics(ownership_data, analyze_by, selected_entity)
render_dashboard_masthead(
    analyze_by,
    selected_entity,
    ownership_metadata,
    classification_metadata,
    type_metadata,
    active_metrics["latest_period"],
)


if analyze_by == "Stock":
    context_name = f"{selected_entity} · {stock_names.get(selected_entity, '')}".rstrip(" ·")
    available_dates = pd.concat(
        [
            frame.loc[frame["ticker"].eq(selected_entity), ["date"]]
            for frame in (ownership_data, classification_data, type_data)
            if not frame.empty and "ticker" in frame and "date" in frame
        ],
        ignore_index=True,
    )
else:
    context_name = selected_entity
    available_dates = ownership_data.loc[
        ownership_data["investor_name"].eq(selected_entity),
        ["date"],
    ]
if available_dates.empty:
    coverage_label = "No history available"
else:
    coverage_label = (
        f"{available_dates['date'].min():%b-%y} to {available_dates['date'].max():%b-%y}"
    )

identity_code = selected_entity if analyze_by == "Stock" else "OWNER"
identity_name = stock_names.get(selected_entity, "") if analyze_by == "Stock" else selected_entity
render_terminal_header(identity_code, identity_name, analyze_by, active_metrics)


ownership_tab, classification_tab, type_tab, entity_movement_tab, monthly_changes_tab = st.tabs(
    ["1% Ownership", "Classification", "Type", "Entity Movement", "Monthly Changes"]
)


with ownership_tab:
    history, row_label = selected_ownership_history(
        ownership_data,
        analyze_by,
        selected_entity,
    )
    if history.empty:
        st.info(f"No 1% Ownership history is available for {selected_entity}.")
    else:
        ownership_dates = sorted(pd.Timestamp(value) for value in ownership_data["date"].dropna().unique())
        section(
            "1% Ownership",
            f"{context_name} ownership movement",
            None,
        )
        ownership_chart_column, snapshot_column = st.columns(
            [3, 1],
            gap="small",
        )
        with ownership_chart_column:
            st.plotly_chart(
                ownership_movement_lines(
                    history,
                    "Ownership movement",
                    ownership_dates,
                ),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"ownership_lines_{analyze_by}_{selected_entity}",
            )
        with snapshot_column:
            render_latest_snapshot(history, row_label, analyze_by)

        shares_pivot = historical_pivot(
            history,
            "series_label",
            "ownership_units",
            ownership_dates,
        )
        ownership_change = monthly_percentage_change_pivot(
            shares_pivot,
            "series_label",
        )
        table_heading("Number of shares")
        render_pivot(
            shares_pivot,
            "series_label",
            row_label,
            "comma",
            f"ownership_shares_pivot_{analyze_by}_{selected_entity}",
            holder_links=analyze_by == "Stock",
            append_delta=True,
        )

        table_heading("Monthly change (%)", separated=True)
        render_pivot(
            ownership_change,
            "series_label",
            row_label,
            "signed_pct",
            f"ownership_change_pivot_{analyze_by}_{selected_entity}",
            holder_links=analyze_by == "Stock",
            heatmap=True,
        )

        table_heading("Ownership percentage", separated=True)
        percentage_pivot = historical_pivot(
            history,
            "series_label",
            "ownership_pct",
            ownership_dates,
        )
        render_pivot(
            percentage_pivot,
            "series_label",
            row_label,
            "%.2f%%",
            f"ownership_pct_pivot_{analyze_by}_{selected_entity}",
            holder_links=analyze_by == "Stock",
        )
        st.caption("Blank cells represent missing observations, not zero ownership.")


with classification_tab:
    if analyze_by == "Owner":
        st.info(
            "This dataset is aggregated by stock and investor classification and does not contain individual owner-level information."
        )
    else:
        classification_history = classification_stock_history(
            classification_data,
            selected_entity,
        )
        if classification_history.empty:
            st.info(f"No Classification ownership history is available for {selected_entity}.")
        else:
            classification_dates = sorted(
                pd.Timestamp(value) for value in classification_data["date"].dropna().unique()
            )
            section(
                "Classification",
                f"{context_name} classification ownership movement",
                "Area shows composition; line boundaries show changes by source classification.",
            )
            st.plotly_chart(
                stacked_area_line_chart(
                    classification_history,
                    "classification",
                    f"{context_name} · Classification ownership",
                    "Number of scripless shares",
                    classification_dates,
                ),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"classification_chart_{selected_entity}",
            )

            classification_shares = historical_pivot(
                classification_history,
                "classification",
                "ownership_units",
                classification_dates,
            )
            classification_change = monthly_percentage_change_pivot(
                classification_shares,
                "classification",
            )
            table_heading("Number of shares by classification")
            render_pivot(
                classification_shares,
                "classification",
                "Classification",
                "comma",
                f"classification_shares_{selected_entity}",
            )

            table_heading("Monthly change (%)", separated=True)
            render_pivot(
                classification_change,
                "classification",
                "Classification",
                "signed_pct",
                f"classification_change_{selected_entity}",
                heatmap=True,
            )

            if classification_history["ownership_pct"].notna().any():
                table_heading("Percentage of scripless ownership", separated=True)
                classification_pct = historical_pivot(
                    classification_history,
                    "classification",
                    "ownership_pct",
                    classification_dates,
                )
                render_pivot(
                    classification_pct,
                    "classification",
                    "Classification",
                    "%.2f%%",
                    f"classification_pct_{selected_entity}",
                )
                st.caption("Percentages use Total Scripless from the Classification source as the denominator.")


with type_tab:
    if analyze_by == "Owner":
        st.info(
            "This dataset is aggregated by stock, residency, investor type, and holding band and does not contain individual owner-level information."
        )
    else:
        type_summary = type_stock_summary(type_data, selected_entity)
        residency_history = type_residency_history(type_data, selected_entity)
        if type_summary.empty:
            st.info(f"No Type ownership history is available for {selected_entity}.")
        else:
            type_dates = sorted(pd.Timestamp(value) for value in type_data["date"].dropna().unique())
            section(
                "Type · Share form",
                f"{context_name} scrip versus scripless movement",
                "Scrip shares equal Number of Shares minus Total Scripless.",
            )
            if type_summary["scrip_shares"].isna().any():
                st.warning(
                    "Scrip shares are unavailable for one or more periods because Total Scripless exceeds Number of Shares in the source."
                )
            share_form_history = type_summary.melt(
                id_vars=["date"],
                value_vars=["total_scripless", "scrip_shares"],
                var_name="share_type",
                value_name="ownership_units",
            )
            share_form_history["share_type"] = share_form_history["share_type"].map(
                {
                    "total_scripless": "Scripless shares",
                    "scrip_shares": "Scrip shares",
                }
            )
            share_form_pivot = historical_pivot(
                share_form_history,
                "share_type",
                "ownership_units",
                type_dates,
            )
            share_form_change = monthly_percentage_change_pivot(
                share_form_pivot,
                "share_type",
            )
            share_form_chart_column, share_form_pivot_column = st.columns(
                [1.25, 1],
                gap="medium",
            )
            with share_form_chart_column:
                st.plotly_chart(
                    scrip_vs_scripless_chart(
                        type_summary,
                        "Scrip and scripless shares",
                    ),
                    width="stretch",
                    config=PLOT_CONFIG,
                    key=f"scrip_chart_{selected_entity}",
                )
            with share_form_pivot_column:
                table_heading("Number of shares by form")
                render_pivot(
                    share_form_pivot,
                    "share_type",
                    "Type",
                    "comma",
                    f"share_form_pivot_{selected_entity}",
                    compact=True,
                )
                table_heading("Monthly change (%)", separated=True)
                render_pivot(
                    share_form_change,
                    "share_type",
                    "Type",
                    "signed_pct",
                    f"share_form_change_{selected_entity}",
                    compact=True,
                    heatmap=True,
                )

            section(
                "Type · Residency",
                f"{context_name} domestic versus foreign movement",
                "Domestic and Foreign come from the explicit source groups.",
            )
            residency_shares = historical_pivot(
                residency_history,
                "domestic_foreign",
                "ownership_units",
                type_dates,
            )
            residency_pct = historical_pivot(
                residency_history,
                "domestic_foreign",
                "ownership_pct",
                type_dates,
            )
            residency_change = monthly_percentage_change_pivot(
                residency_shares,
                "domestic_foreign",
            )
            residency_chart_column, residency_pivot_column = st.columns(
                [1.25, 1],
                gap="medium",
            )
            with residency_chart_column:
                st.plotly_chart(
                    stacked_area_line_chart(
                        residency_history,
                        "domestic_foreign",
                        "Domestic and foreign ownership",
                        "Number of scripless shares",
                        type_dates,
                        color_map={"Domestic": "#00C781", "Foreign": "#3182F6"},
                    ),
                    width="stretch",
                    config=PLOT_CONFIG,
                    key=f"residency_chart_{selected_entity}",
                )
            with residency_pivot_column:
                table_heading("Number of shares by residency")
                render_pivot(
                    residency_shares,
                    "domestic_foreign",
                    "Type",
                    "comma",
                    f"residency_shares_{selected_entity}",
                    compact=True,
                )
                table_heading("Monthly change (%)", separated=True)
                render_pivot(
                    residency_change,
                    "domestic_foreign",
                    "Type",
                    "signed_pct",
                    f"residency_change_{selected_entity}",
                    compact=True,
                    heatmap=True,
                )
                table_heading("Percentage of total shares", separated=True)
                render_pivot(
                    residency_pct,
                    "domestic_foreign",
                    "Type",
                    "%.2f%%",
                    f"residency_pct_{selected_entity}",
                    compact=True,
                )
            st.caption("Domestic and Foreign reconcile to Total Scripless; percentages use Number of Shares as the denominator.")


with entity_movement_tab:
    entity_column = "ticker" if analyze_by == "Stock" else "investor_name"
    monthly, breakdown, counterparty_label = entity_monthly_movement(
        ownership_data,
        entity_column,
        selected_entity,
    )
    if monthly.empty:
        st.info(f"No monthly ownership movement is available for {selected_entity}.")
    else:
        movement_metric = "Reported shares" if analyze_by == "Stock" else "Stake points"
        section(
            "Monthly Change",
            f"{context_name} period-over-period movement",
            (
                "Share movement across reported holders."
                if analyze_by == "Stock"
                else "Stake-point movement across reported stocks."
            ),
        )
        st.plotly_chart(
            monthly_movement_chart(
                monthly,
                f"{context_name} · Monthly movement",
                movement_metric,
            ),
            width="stretch",
            config=PLOT_CONFIG,
            key=f"monthly_total_{analyze_by}_{selected_entity}",
        )
        st.plotly_chart(
            movement_breakdown_chart(
                breakdown,
                counterparty_label,
                movement_metric,
            ),
            width="stretch",
            config=PLOT_CONFIG,
            key=f"monthly_breakdown_{analyze_by}_{selected_entity}",
        )


with monthly_changes_tab:
    comparison_dates = sorted(
        pd.Timestamp(value) for value in ownership_data["date"].dropna().unique()
    )
    if len(comparison_dates) < 2:
        st.info("At least two ownership reporting months are required for market-wide changes.")
    else:
        heading_column, month_column = st.columns([4, 1.2], gap="large")
        with heading_column:
            section(
                "Market-wide activity",
                "Monthly Changes",
                "Ranked from consecutive reported KSEI ownership snapshots.",
            )
        with month_column:
            selected_change_month = st.selectbox(
                "Reporting Month",
                comparison_dates[1:],
                index=len(comparison_dates) - 2,
                format_func=lambda value: pd.Timestamp(value).strftime("%B %Y"),
                key="market_change_month",
            )

        selected_change_month = pd.Timestamp(selected_change_month)
        previous_change_month = max(
            value for value in comparison_dates if value < selected_change_month
        )
        with st.spinner("Preparing market-wide ownership changes…"):
            change_detail, stock_changes, owner_changes = monthly_change_analysis(
                ownership_data
            )
        period_detail = change_detail[
            change_detail["date"].eq(selected_change_month)
            & change_detail["is_changed"]
        ].copy()
        period_stocks = (
            stock_changes[stock_changes["date"].eq(selected_change_month)]
            .sort_values(["absolute_change", "stock"], ascending=[False, True])
            .reset_index(drop=True)
        )
        period_owners = (
            owner_changes[owner_changes["date"].eq(selected_change_month)]
            .sort_values(["absolute_change", "owner"], ascending=[False, True])
            .reset_index(drop=True)
        )

        render_monthly_change_kpis(
            period_stocks,
            period_detail,
            selected_change_month,
            previous_change_month,
        )
        st.caption(
            "Changes describe reported 1% Ownership snapshots, not confirmed exchange transactions. "
            "Newly or no-longer-reported holders may have crossed the reporting threshold; their legal holding is not assumed to be zero."
        )

        table_heading("Stocks with largest ownership changes", separated=True)
        stock_controls = st.columns([3, 1], gap="large")
        with stock_controls[0]:
            st.caption(
                "Ranked by total absolute holder-level share movement. Select a row to inspect the owners responsible."
            )
        with stock_controls[1]:
            show_all_stocks = st.toggle(
                f"Show all {len(period_stocks):,} stocks",
                key=f"show_all_stocks_{selected_change_month:%Y%m}",
            )

        ranked_stocks = period_stocks.copy()
        ranked_stocks.insert(0, "Rank", range(1, len(ranked_stocks) + 1))
        visible_stocks = ranked_stocks if show_all_stocks else ranked_stocks.head(10)
        stock_display = visible_stocks[
            [
                "Rank",
                "stock",
                "previous_shares",
                "current_shares",
                "net_change",
                "percent_change",
                "absolute_change",
                "changing_holders",
            ]
        ].rename(
            columns={
                "stock": "Stock",
                "previous_shares": "Previous Month",
                "current_shares": "Current Month",
                "net_change": "Net Change",
                "percent_change": "% Change",
                "absolute_change": "Total Movement",
                "changing_holders": "Changing Holders",
            }
        )
        selected_stock_rows = render_activity_dataframe(
            stock_display,
            f"market_stock_changes_{selected_change_month:%Y%m}_{show_all_stocks}",
            {
                "Previous Month": "{:,.0f}",
                "Current Month": "{:,.0f}",
                "Net Change": lambda value: "0" if abs(value) < 0.5 else f"{value:+,.0f}",
                "% Change": lambda value: "0.00%" if abs(value) < 1e-9 else f"{value:+.2f}%",
                "Total Movement": "{:,.0f}",
                "Changing Holders": "{:,.0f}",
            },
            ["Net Change", "% Change"],
            ["Stock"],
            selectable=True,
            max_height=490,
        )
        if selected_stock_rows:
            stock_row = visible_stocks.iloc[selected_stock_rows[0]]
            stock_code = str(stock_row["stock"])
            stock_detail = period_detail[period_detail["stock"].eq(stock_code)].copy()
            stock_detail["_magnitude"] = stock_detail["change_shares"].abs()
            stock_detail = stock_detail.sort_values(
                ["_magnitude", "owner"], ascending=[False, True]
            )
            stock_detail_display = stock_detail[
                [
                    "owner",
                    "previous_shares",
                    "current_shares",
                    "change_shares",
                    "previous_percentage",
                    "current_percentage",
                    "change_percentage",
                    "observation_status",
                ]
            ].rename(
                columns={
                    "owner": "Owner",
                    "previous_shares": "Previous Shares",
                    "current_shares": "Current Shares",
                    "change_shares": "Change Shares",
                    "previous_percentage": "Previous %",
                    "current_percentage": "Current %",
                    "change_percentage": "Change % (pp)",
                    "observation_status": "Status",
                }
            )
            table_heading(
                f"{stock_code} · owners with reported changes",
                separated=True,
            )
            render_activity_dataframe(
                stock_detail_display,
                f"market_stock_detail_{selected_change_month:%Y%m}_{stock_code}",
                {
                    "Previous Shares": "{:,.0f}",
                    "Current Shares": "{:,.0f}",
                    "Change Shares": lambda value: "0" if abs(value) < 0.5 else f"{value:+,.0f}",
                    "Previous %": "{:.2f}%",
                    "Current %": "{:.2f}%",
                    "Change % (pp)": lambda value: "0.00 pp" if abs(value) < 1e-9 else f"{value:+.2f} pp",
                },
                ["Change Shares", "Change % (pp)"],
                ["Owner"],
                max_height=420,
            )
        else:
            st.caption("Select a stock row to see which owners increased or decreased reported ownership.")

        table_heading("Owners with largest monthly changes", separated=True)
        owner_controls = st.columns([3, 1], gap="large")
        with owner_controls[0]:
            st.caption(
                "Ranked by total absolute reported share movement across stocks. Select a row for the stock breakdown."
            )
        with owner_controls[1]:
            show_all_owners = st.toggle(
                f"Show all {len(period_owners):,} owners",
                key=f"show_all_owners_{selected_change_month:%Y%m}",
            )

        ranked_owners = period_owners.copy()
        ranked_owners.insert(0, "Rank", range(1, len(ranked_owners) + 1))
        visible_owners = ranked_owners if show_all_owners else ranked_owners.head(10)
        owner_display = visible_owners[
            [
                "Rank",
                "owner",
                "stocks_increased",
                "stocks_decreased",
                "absolute_change",
                "net_change",
                "stocks_changed",
            ]
        ].rename(
            columns={
                "owner": "Owner",
                "stocks_increased": "Stocks Increased",
                "stocks_decreased": "Stocks Decreased",
                "absolute_change": "Total Absolute Change",
                "net_change": "Net Change",
                "stocks_changed": "Stocks Changed",
            }
        )
        selected_owner_rows = render_activity_dataframe(
            owner_display,
            f"market_owner_changes_{selected_change_month:%Y%m}_{show_all_owners}",
            {
                "Stocks Increased": "{:,.0f}",
                "Stocks Decreased": "{:,.0f}",
                "Total Absolute Change": "{:,.0f}",
                "Net Change": lambda value: "0" if abs(value) < 0.5 else f"{value:+,.0f}",
                "Stocks Changed": "{:,.0f}",
            },
            ["Net Change"],
            ["Owner"],
            selectable=True,
            max_height=490,
        )
        if selected_owner_rows:
            owner_row = visible_owners.iloc[selected_owner_rows[0]]
            owner_name = str(owner_row["owner"])
            owner_detail = period_detail[period_detail["owner"].eq(owner_name)].copy()
            owner_detail["_magnitude"] = owner_detail["change_shares"].abs()
            owner_detail = owner_detail.sort_values(
                ["_magnitude", "stock"], ascending=[False, True]
            )
            owner_detail_display = owner_detail[
                [
                    "stock",
                    "previous_shares",
                    "current_shares",
                    "change_shares",
                    "previous_percentage",
                    "current_percentage",
                    "change_percentage",
                    "observation_status",
                ]
            ].rename(
                columns={
                    "stock": "Stock",
                    "previous_shares": "Previous Shares",
                    "current_shares": "Current Shares",
                    "change_shares": "Change Shares",
                    "previous_percentage": "Previous %",
                    "current_percentage": "Current %",
                    "change_percentage": "Change % (pp)",
                    "observation_status": "Status",
                }
            )
            table_heading(
                f"{owner_name} · stocks with reported changes",
                separated=True,
            )
            render_activity_dataframe(
                owner_detail_display,
                f"market_owner_detail_{selected_change_month:%Y%m}_{owner_name}",
                {
                    "Previous Shares": "{:,.0f}",
                    "Current Shares": "{:,.0f}",
                    "Change Shares": lambda value: "0" if abs(value) < 0.5 else f"{value:+,.0f}",
                    "Previous %": "{:.2f}%",
                    "Current %": "{:.2f}%",
                    "Change % (pp)": lambda value: "0.00 pp" if abs(value) < 1e-9 else f"{value:+.2f} pp",
                },
                ["Change Shares", "Change % (pp)"],
                ["Stock"],
                max_height=420,
            )
        else:
            st.caption("Select an owner row to see the stocks with reported ownership changes.")
