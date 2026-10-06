from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from charts import (
    daily_owner_trend_chart,
    market_activity_chart,
    movement_breakdown_chart,
    monthly_movement_chart,
    ownership_movement_lines,
    scrip_vs_scripless_chart,
    stacked_area_line_chart,
)
from daily_ownership import (
    SIGNAL_ACCUMULATING,
    SIGNAL_ENTERED,
    SIGNAL_EXITED,
    SIGNAL_INTERNAL_TRANSFER,
    SIGNAL_SELLING,
    SIGNAL_UNCHANGED,
    build_account_hierarchy_pivots,
    build_owner_account_hierarchy_pivots,
    load_daily_ownership_folder,
    normalize_identity,
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
    filter_market_overview,
    owner_change_summary,
    stock_change_summary,
)
from utils import compact_number


APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = APP_DIR / "BEI_Data"
OWNERSHIP_DIR = DATA_ROOT / "1% Ownership"
CLASSIFICATION_DIR = DATA_ROOT / "Classification"
TYPE_DIR = DATA_ROOT / "Type"
DAILY_OWNERSHIP_DIR = DATA_ROOT / "5% Ownership"
CONFIG_PATH = APP_DIR / "schema_mapping.json"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}
LOGGER = logging.getLogger("ksei_dashboard")


st.set_page_config(
    page_title="KSEI Ownership Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"<style>{(APP_DIR / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False, max_entries=4)
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


@st.cache_resource(show_spinner=False, max_entries=4)
def load_classification_dataset(
    folder: str,
    signature: tuple[tuple[str, int, int], ...],
    config_text: str,
) -> tuple[pd.DataFrame, dict]:
    return load_classification_folder(folder, json.loads(config_text))


@st.cache_resource(show_spinner=False, max_entries=4)
def load_type_dataset(
    folder: str,
    signature: tuple[tuple[str, int, int], ...],
    config_text: str,
) -> tuple[pd.DataFrame, dict]:
    return load_type_folder(folder, json.loads(config_text))


@st.cache_resource(show_spinner=False, max_entries=2)
def load_daily_ownership_dataset(
    folder: str,
    signature: tuple[tuple[str, int, int], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load normalized daily >5% data only when its source-file signature changes."""
    return load_daily_ownership_folder(folder)


@st.cache_resource(show_spinner=False, max_entries=4)
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


def empty_classification_data() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "security_name",
            "classification",
            "ownership_units",
            "total_scripless",
            "source_file",
        ]
    )


def empty_type_data() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "number_of_shares",
            "total_scripless",
            "scrip_shares",
            "domestic_foreign",
            "investor_category",
            "holding_band",
            "ownership_units",
            "ownership_pct",
            "source_file",
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
        classification_data = empty_classification_data()
        classification_metadata["issues"] = [str(error)]

    try:
        type_data, type_metadata = load_type_dataset(
            str(TYPE_DIR),
            folder_signature(TYPE_DIR),
            config_text,
        )
    except Exception as error:
        LOGGER.exception("Could not load Type data")
        type_data = empty_type_data()
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
            return "background-color:rgba(168,176,164,.10);color:#7B847F;"
        ratio = min(abs(numeric) / heatmap_scale, 1.0) if heatmap_scale else 0.0
        opacity = 0.07 + 0.16 * ratio**0.5
        rgb = "47,168,79" if numeric > 0 else "214,92,92"
        color = "#2FA84F" if numeric > 0 else "#D65C5C"
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
            "history_unit": "M",
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
        "history_unit": "M",
    }


def daily_selection_metrics(
    data: pd.DataFrame,
    analyze_by: str,
    selected_entity: str,
) -> dict[str, object]:
    """Build presentation-only header metrics from daily >5% data."""
    if analyze_by == "Stock":
        scoped = data[data["ticker"].eq(selected_entity)].copy()
        counterparty_column = "owner_normalized"
    else:
        owner_key = normalize_identity(selected_entity)
        scoped = data[data["owner_normalized"].eq(owner_key)].copy()
        counterparty_column = "ticker"
    scoped = scoped.dropna(subset=["date"])
    if scoped.empty:
        return {
            "reported_shares": 0.0,
            "delta": None,
            "delta_pct": None,
            "counterparties": 0,
            "largest_stake": None,
            "months": 0,
            "latest_period": "No data",
            "history_unit": "D",
        }

    dates = sorted(pd.Timestamp(value) for value in scoped["date"].unique())
    latest = scoped[scoped["date"].eq(dates[-1])]
    daily_totals = scoped.groupby("date", observed=True)["shares"].sum()
    current_total = float(daily_totals.loc[dates[-1]])
    previous_total = float(daily_totals.loc[dates[-2]]) if len(dates) > 1 else None
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
        "latest_period": dates[-1].strftime("%d-%b-%y").upper(),
        "history_unit": "D",
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
    history_unit = str(metrics.get("history_unit", "M"))
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
        f'<span><b>HISTORY</b><strong>{int(metrics["months"])}{escape(history_unit)}</strong></span>'
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
    daily_metadata: dict,
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
        f'<span>DAILY 5% <b>{int(daily_metadata.get("parsed_files", 0))}</b></span>'
        f'<span>UPDATED <b>{escape(str(latest_period))}</b></span>'
        '<span class="coverage-connected"><i></i>DATA CONNECTED</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_page_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    latest_period: object,
) -> None:
    """Render the shared compact page header for the presentation layer."""
    st.markdown(
        '<div class="page-header">'
        '<div class="page-header-copy">'
        f'<span>{escape(eyebrow)}</span>'
        f'<h1>{escape(title)}</h1>'
        f'<p>{escape(subtitle)}</p>'
        '</div>'
        '<div class="page-header-tools">'
        '<span class="live-dot"></span>'
        f'<div><small>LATEST DATA</small><strong>{escape(str(latest_period))}</strong></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_market_kpis(daily_movements: pd.DataFrame) -> None:
    """Summarize the latest daily >5% signal snapshot without changing it."""
    if daily_movements.empty or "date" not in daily_movements:
        return
    latest_date = pd.Timestamp(daily_movements["date"].max())
    latest = daily_movements[daily_movements["date"].eq(latest_date)].copy()
    change = pd.to_numeric(latest.get("delta_shares"), errors="coerce").fillna(0)
    cards = [
        ("ACCUMULATING", int(latest["signal"].eq(SIGNAL_ACCUMULATING).sum()), "positive", "↑"),
        ("SELLING", int(latest["signal"].eq(SIGNAL_SELLING).sum()), "negative", "↓"),
        ("INTERNAL TRANSFER", int(latest["signal"].eq(SIGNAL_INTERNAL_TRANSFER).sum()), "transfer", "⇄"),
        ("ENTERED >5%", int(latest["signal"].eq(SIGNAL_ENTERED).sum()), "positive", "+"),
        ("EXITED >5%", int(latest["signal"].eq(SIGNAL_EXITED).sum()), "negative", "−"),
        ("NET REPORTED Δ", terminal_number(change.sum()), "neutral", "Δ"),
    ]
    card_html = "".join(
        '<div class="market-kpi">'
        f'<span class="kpi-icon {css_class}">{icon}</span>'
        f'<div><small>{escape(label)}</small><strong class="{css_class}">{escape(str(value))}</strong></div>'
        '</div>'
        for label, value, css_class, icon in cards
    )
    st.markdown(
        f'<div class="market-kpi-grid">{card_html}</div>',
        unsafe_allow_html=True,
    )


def render_signal_summary(daily_movements: pd.DataFrame) -> None:
    if daily_movements.empty:
        st.info("No daily >5% signal data is available.")
        return
    latest_date = pd.Timestamp(daily_movements["date"].max())
    latest = daily_movements[daily_movements["date"].eq(latest_date)].copy()
    rows = [
        ("Accumulating", SIGNAL_ACCUMULATING, "positive"),
        ("Selling", SIGNAL_SELLING, "negative"),
        ("Internal transfer", SIGNAL_INTERNAL_TRANSFER, "transfer"),
        ("Entered >5%", SIGNAL_ENTERED, "positive"),
        ("Exited >5%", SIGNAL_EXITED, "negative"),
    ]
    total = max(len(latest), 1)
    row_html = "".join(
        '<div class="signal-summary-row">'
        f'<span><i class="{css_class}"></i>{escape(label)}</span>'
        f'<strong>{int(latest["signal"].eq(signal).sum()):,}</strong>'
        f'<small>{latest["signal"].eq(signal).sum() / total:.0%}</small>'
        '</div>'
        for label, signal, css_class in rows
    )
    st.markdown(
        '<div class="signal-summary-card">'
        '<div class="card-title"><span>SIGNAL MIX</span>'
        f'<small>{latest_date:%d %b %Y}</small></div>'
        f'{row_html}'
        '<p>Owner signals use combined beneficial ownership. Account transfers remain separate.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def movement_cell_style(value: object) -> str:
    if pd.isna(value):
        return "color:#7B847F;"
    numeric = float(value)
    if numeric > 1e-9:
        return "color:#2FA84F;background-color:rgba(47,168,79,.08);"
    if numeric < -1e-9:
        return "color:#D65C5C;background-color:rgba(214,92,92,.08);"
    return "color:#7B847F;"


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
                **{"color": "#4D75D7", "font-weight": "650"},
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


def render_account_pivot(
    pivot: pd.DataFrame,
    key: str,
    *,
    movement: bool = False,
    max_height: int = 430,
) -> None:
    """Render a compact Date x Account terminal table."""
    if pivot.empty:
        st.info("No account-level observations are available for this selection.")
        return

    view = pivot.copy()
    view.index = pd.DatetimeIndex(view.index).strftime("%d-%b-%y").str.upper()
    view.index.name = "Date"
    formatter = (
        signed_shares
        if movement
        else lambda value: "—" if pd.isna(value) else f"{float(value):,.0f}"
    )
    styled = view.style.format(formatter, na_rep="—")
    if movement:
        styled = styled.map(movement_cell_style)
    styled = styled.set_table_styles(
        [
            {
                "selector": "th.col_heading.level0",
                "props": "color:#17201D;font-weight:750;border-bottom:1px solid #E3E7E3;",
            },
            {
                "selector": "th.col_heading.level1",
                "props": "color:#7B847F;font-weight:650;",
            },
            {
                "selector": "th.row_heading",
                "props": "color:#17201D;font-weight:650;",
            },
        ],
        overwrite=False,
    )
    height = min(max_height, 39 + 35 * len(view))
    st.dataframe(
        styled,
        width="stretch",
        height=height,
        hide_index=False,
        row_height=34,
        key=key,
    )


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


DAILY_SIGNAL_LABELS = {
    SIGNAL_ACCUMULATING: "▲ ACCUMULATING",
    SIGNAL_SELLING: "▼ SELLING",
    SIGNAL_INTERNAL_TRANSFER: "⇄ INTERNAL TRANSFER",
    SIGNAL_UNCHANGED: "— UNCHANGED",
    SIGNAL_ENTERED: "NEW >5%",
    SIGNAL_EXITED: "EXIT <5%",
    "NEWLY REPORTED": "NEWLY REPORTED",
    "NO LONGER REPORTED": "NO LONGER REPORTED",
    "DATA ISSUE": "DATA ISSUE",
}


def daily_signal_label(signal: object) -> str:
    return DAILY_SIGNAL_LABELS.get(str(signal), str(signal))


def daily_signal_class(signal: object) -> str:
    value = str(signal)
    if value in {SIGNAL_ACCUMULATING, SIGNAL_ENTERED}:
        return "daily-positive"
    if value in {SIGNAL_SELLING, SIGNAL_EXITED}:
        return "daily-negative"
    if value == SIGNAL_INTERNAL_TRANSFER:
        return "daily-transfer"
    return "daily-flat"


def signed_shares(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    numeric = float(value)
    return "0" if abs(numeric) < .5 else f"{numeric:+,.0f}"


def signed_points(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    numeric = float(value)
    return "0.00 pp" if abs(numeric) < 1e-9 else f"{numeric:+.2f} pp"


def owner_daily_history(
    owners: pd.DataFrame,
    movements: pd.DataFrame,
    ticker: str,
    owner_normalized: str,
) -> pd.DataFrame:
    history = owners[
        owners["ticker"].eq(ticker)
        & owners["owner_normalized"].eq(owner_normalized)
    ].copy()
    if history.empty:
        return history
    daily_delta = (
        movements[
            movements["ticker"].eq(ticker)
            & movements["owner_normalized"].eq(owner_normalized)
        ][["date", "delta_shares", "signal"]]
        .drop_duplicates("date", keep="last")
        .rename(columns={"delta_shares": "daily_delta_shares"})
    )
    history = history.merge(daily_delta, on="date", how="left")
    history["daily_delta_shares"] = pd.to_numeric(
        history["daily_delta_shares"], errors="coerce"
    )
    return history.sort_values("date").reset_index(drop=True)


def render_daily_summary(period: pd.Timestamp, snapshot: pd.DataFrame) -> None:
    counts = snapshot["signal"].value_counts() if not snapshot.empty else pd.Series(dtype=int)
    changed_tickers = snapshot.loc[
        ~snapshot["signal"].eq(SIGNAL_UNCHANGED), "ticker"
    ].nunique()
    metrics = [
        ("REPORT DATE", period.strftime("%d %b %Y").upper(), ""),
        ("OWNERS", f"{snapshot['owner_normalized'].nunique():,}", ""),
        ("ACCUMULATING", f"{int(counts.get(SIGNAL_ACCUMULATING, 0)):,}", "daily-positive"),
        ("SELLING", f"{int(counts.get(SIGNAL_SELLING, 0)):,}", "daily-negative"),
        ("INTERNAL TRANSFERS", f"{int(counts.get(SIGNAL_INTERNAL_TRANSFER, 0)):,}", "daily-transfer"),
        ("NEW >5%", f"{int(counts.get(SIGNAL_ENTERED, 0)):,}", "daily-positive"),
        ("EXIT <5%", f"{int(counts.get(SIGNAL_EXITED, 0)):,}", "daily-negative"),
        ("TICKERS CHANGED", f"{changed_tickers:,}", ""),
    ]
    cells = "".join(
        f'<span><b>{escape(label)}</b><strong class="{css_class}">{escape(value)}</strong></span>'
        for label, value, css_class in metrics
    )
    st.markdown(f'<div class="daily-summary-strip">{cells}</div>', unsafe_allow_html=True)


def render_daily_owner_monitor(
    snapshot: pd.DataFrame,
    analyze_by: str = "Stock",
    limit: int = 7,
) -> None:
    if snapshot.empty:
        st.info("No daily owner snapshot is available.")
        return
    view = snapshot.sort_values("current_shares", ascending=False).head(limit)
    label_column = "owner" if analyze_by == "Stock" else "ticker"
    monitor_title = (
        "LATEST BENEFICIAL OWNERS"
        if analyze_by == "Stock"
        else "LATEST STOCK POSITIONS"
    )
    rows = []
    for position, (_, row) in enumerate(view.iterrows(), start=1):
        label = str(row[label_column])
        company = str(row.get("issuer", "")).strip()
        if analyze_by == "Owner" and company:
            label = f"{label} · {company}"
        rows.append(
            '<div class="daily-monitor-row">'
            f'<span class="daily-monitor-rank">{position}</span>'
            f'<span class="daily-monitor-owner" title="{escape(label, quote=True)}">{escape(label)}</span>'
            '<span class="daily-monitor-value">'
            f'<strong>{escape(terminal_number(row["current_shares"]))}</strong>'
            f'<small>{float(row["current_pct"]):.2f}%</small></span>'
            f'<em class="{daily_signal_class(row["signal"])}">{escape(daily_signal_label(row["signal"]))}</em>'
            '</div>'
        )
    st.markdown(
        f'<div class="daily-monitor"><div class="daily-monitor-header">{escape(monitor_title)}'
        '<span title="Beneficial Owner is Nama Pemegang Saham and determines the economic ownership signal.">ⓘ</span>'
        f'</div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_owner_metric_strip(history: pd.DataFrame, owner_label: str) -> None:
    if history.empty:
        return
    shares = pd.to_numeric(history["shares"], errors="coerce")
    current = shares.iloc[-1]
    current_pct = pd.to_numeric(history["ownership_pct"], errors="coerce").iloc[-1]

    def period_change(offset: int) -> str:
        if len(shares) <= offset or pd.isna(current) or pd.isna(shares.iloc[-offset - 1]):
            return "—"
        return signed_shares(float(current) - float(shares.iloc[-offset - 1]))

    signals = history.get("signal", pd.Series(dtype=str))
    values = [
        ("CURRENT SHARES", f"{current:,.0f}"),
        ("CURRENT OWNERSHIP", f"{current_pct:.2f}%"),
        ("1D CHANGE", period_change(1)),
        ("5D CHANGE", period_change(5)),
        ("20D CHANGE", period_change(20)),
        ("HIGH", f"{shares.max():,.0f}"),
        ("LOW", f"{shares.min():,.0f}"),
        ("UP / DOWN DAYS", f"{signals.eq(SIGNAL_ACCUMULATING).sum()} / {signals.eq(SIGNAL_SELLING).sum()}"),
    ]
    cells = "".join(
        f'<span><b>{escape(label)}</b><strong>{escape(value)}</strong></span>'
        for label, value in values
    )
    st.markdown(
        f'<div class="owner-metric-title">{escape(owner_label)}</div>'
        f'<div class="owner-metric-strip">{cells}</div>',
        unsafe_allow_html=True,
    )


def render_account_interpretation(
    owner_movement: pd.Series,
    account_changes: pd.DataFrame,
) -> None:
    changed = account_changes[account_changes["delta_shares"].abs().gt(.5)].copy()
    changed = changed.sort_values("delta_shares")
    accounts_html = "".join(
        '<span class="account-shift">'
        f'<b>{escape(str(row["account_holder"]))}</b>'
        f'<em class="{"daily-positive" if float(row["delta_shares"]) > 0 else "daily-negative"}">'
        f'{escape(signed_shares(row["delta_shares"]))}</em></span>'
        for _, row in changed.iterrows()
    )
    st.markdown(
        '<div class="account-interpretation">'
        f'<span><b>OWNER NET CHANGE</b><strong>{escape(signed_shares(owner_movement["delta_shares"]))}</strong></span>'
        f'<span><b>CLASSIFICATION</b><strong class="{daily_signal_class(owner_movement["signal"])}">{escape(daily_signal_label(owner_movement["signal"]))}</strong></span>'
        f'<div class="account-shifts">{accounts_html or "<small>NO ACCOUNT-LEVEL CHANGE</small>"}</div>'
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

daily_metadata: dict = {"source_files": 0, "parsed_files": 0, "quality_warnings": 0}
try:
    with st.spinner("Loading daily >5% ownership history…"):
        (
            daily_owner_data,
            daily_account_data,
            daily_movements,
            daily_account_movements,
            daily_quality,
            daily_metadata,
        ) = load_daily_ownership_dataset(
            str(DAILY_OWNERSHIP_DIR),
            folder_signature(DAILY_OWNERSHIP_DIR),
        )
except Exception as error:
    LOGGER.exception("Could not load daily >5%% Ownership data")
    daily_owner_data = pd.DataFrame()
    daily_account_data = pd.DataFrame()
    daily_movements = pd.DataFrame()
    daily_account_movements = pd.DataFrame()
    daily_quality = pd.DataFrame()
    daily_metadata["issues"] = [str(error)]


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
if not daily_owner_data.empty:
    stock_names.update(
        daily_owner_data[["ticker", "issuer"]]
        .dropna(subset=["ticker"])
        .drop_duplicates("ticker")
        .set_index("ticker")["issuer"]
        .fillna("")
        .astype(str)
        .to_dict()
    )
stock_options = sorted(
    set(stock_names)
    | (set(classification_data["ticker"].dropna().astype(str)) if "ticker" in classification_data else set())
    | (set(type_data["ticker"].dropna().astype(str)) if "ticker" in type_data else set())
    | (set(daily_owner_data["ticker"].dropna().astype(str)) if "ticker" in daily_owner_data else set())
)
owner_options = sorted(
    (set(ownership_data["investor_name"].dropna().astype(str)) if "investor_name" in ownership_data else set())
    | (set(daily_owner_data["owner"].dropna().astype(str)) if "owner" in daily_owner_data else set())
)

holder_link_target = st.query_params.get("holder")
if holder_link_target:
    holder_link_target = str(holder_link_target)
    if holder_link_target in owner_options:
        st.session_state["top_navigation"] = "1% Ownership"
        st.session_state["analysis_mode"] = "Owner"
        st.session_state["selected_owner"] = holder_link_target
    st.query_params.clear()

top_navigation = [
    "Overview",
    "1% Ownership",
    "5% Ownership",
    "Classification",
    "Type",
    "Entity Movement",
]
page_query = str(st.query_params.get("page", "")).strip().lower()
page_deep_links = {
    "overview": "Overview",
    "1pct": "1% Ownership",
    "5pct": "5% Ownership",
    "classification": "Classification",
    "type": "Type",
    "entity": "Entity Movement",
}
if (
    page_query in page_deep_links
    and st.session_state.get("_page_query_applied") != page_query
):
    st.session_state["top_navigation"] = page_deep_links[page_query]
    st.session_state["_page_query_applied"] = page_query
if st.session_state.get("top_navigation") not in top_navigation:
    st.session_state["top_navigation"] = "Overview"
all_source_dates = [
    pd.Timestamp(value)
    for frame in (ownership_data, classification_data, type_data, daily_owner_data)
    if not frame.empty and "date" in frame
    for value in frame["date"].dropna().unique()
]
global_latest = max(all_source_dates).strftime("%d %b %Y").upper() if all_source_dates else "—"

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="brand-mark">K</div>'
        '<div><strong>KSEI</strong><span>OWNERSHIP INTELLIGENCE</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    active_page = st.radio(
        "Workspace navigation",
        top_navigation,
        label_visibility="collapsed",
        key="top_navigation",
    )
    st.markdown(
        '<div class="sidebar-data-card">'
        '<span>DATA COVERAGE</span>'
        '<div class="sidebar-source-grid">'
        f'<span>Ownership</span><strong>{int(ownership_metadata.get("source_files", 0))}</strong>'
        f'<span>Classification</span><strong>{int(classification_metadata.get("source_files", 0))}</strong>'
        f'<span>Type</span><strong>{int(type_metadata.get("source_files", 0))}</strong>'
        f'<span>Daily 5%</span><strong>{int(daily_metadata.get("parsed_files", 0))}</strong>'
        f'<span>Latest</span><strong>{escape(global_latest)}</strong>'
        '</div>'
        '<div class="sidebar-status"><span></span>CONNECTED</div>'
        '</div>',
        unsafe_allow_html=True,
    )

global_search_options = [f"S|{ticker}" for ticker in stock_options] + [
    f"O|{owner}" for owner in owner_options
]
monthly_tickers = set(ownership_data["ticker"].dropna().astype(str))
daily_tickers = set(daily_owner_data["ticker"].dropna().astype(str))
monthly_owners = set(ownership_data["investor_name"].dropna().astype(str))
daily_owners = set(daily_owner_data["owner"].dropna().astype(str))


def apply_global_entity_search() -> None:
    """Apply search state before navigation and entity widgets are instantiated."""
    search_value = st.session_state.get("global_entity_search")
    if not search_value:
        return
    search_kind, entity = str(search_value).split("|", 1)
    current_page = st.session_state.get("top_navigation", "Overview")
    if search_kind == "S":
        st.session_state["selected_stock"] = entity
        st.session_state["analysis_mode"] = "Stock"
        if current_page == "5% Ownership" and entity in daily_tickers:
            destination = "5% Ownership"
        elif current_page == "1% Ownership" and entity in monthly_tickers:
            destination = "1% Ownership"
        else:
            destination = "1% Ownership" if entity in monthly_tickers else "5% Ownership"
    else:
        st.session_state["selected_owner"] = entity
        st.session_state["analysis_mode"] = "Owner"
        if current_page == "5% Ownership" and entity in daily_owners:
            destination = "5% Ownership"
        elif current_page == "1% Ownership" and entity in monthly_owners:
            destination = "1% Ownership"
        else:
            destination = "1% Ownership" if entity in monthly_owners else "5% Ownership"
    st.session_state["top_navigation"] = destination


global_search_column = st.columns([4.8, 1.65], gap="small")[1]
with global_search_column:
    global_search = st.selectbox(
        "Global Search",
        global_search_options,
        index=None,
        placeholder="Search ticker or owner…",
        format_func=lambda value: (
            f"{value[2:]} · {stock_names.get(value[2:], '')}".rstrip(" ·")
            if value.startswith("S|")
            else f"Owner · {value[2:]}"
        ),
        key="global_entity_search",
        on_change=apply_global_entity_search,
        label_visibility="collapsed",
    )

overview_section = ""
if active_page == "Overview":
    market_dates = [
        pd.Timestamp(value)
        for frame in (ownership_data, daily_owner_data)
        if not frame.empty and "date" in frame
        for value in frame["date"].dropna().unique()
    ]
    market_latest = max(market_dates).strftime("%d-%b-%y").upper() if market_dates else "—"
    render_page_header(
        "MARKET OVERVIEW",
        "Ownership Intelligence",
        "Monitor reported ownership changes, threshold events, and account-level transfers across the Indonesian market.",
        market_latest,
    )
    render_market_kpis(daily_movements)
    overview_chart_column, overview_signal_column = st.columns([2.45, 1], gap="small")
    with overview_chart_column:
        st.plotly_chart(
            market_activity_chart(daily_movements),
            width="stretch",
            config=PLOT_CONFIG,
            key="overview_market_activity",
        )
    with overview_signal_column:
        render_signal_summary(daily_movements)
    overview_section = st.radio(
        "Overview dataset",
        ["1% Monthly Changes", "5% Daily Movement"],
        horizontal=True,
        label_visibility="collapsed",
        key="overview_navigation",
    )
else:
    analyze_by = "Stock"
    if active_page in {"1% Ownership", "5% Ownership", "Entity Movement"}:
        selector_mode_column, selector_entity_column = st.columns([1, 4], gap="medium")
        with selector_mode_column:
            analyze_by = st.selectbox(
                "Analyze By", ["Stock", "Owner"], key="analysis_mode"
            )
    else:
        selector_entity_column = st.container()

    with selector_entity_column:
        if analyze_by == "Owner":
            page_owners = {
                "1% Ownership": sorted(monthly_owners),
                "5% Ownership": sorted(daily_owners),
            }.get(active_page, owner_options)
            if not page_owners:
                st.error("No individual owners were found in the ownership data.")
                st.stop()
            if st.session_state.get("selected_owner") not in page_owners:
                st.session_state["selected_owner"] = page_owners[0]
            selected_entity = st.selectbox("Owner", page_owners, key="selected_owner")
        else:
            page_stocks = {
                "1% Ownership": sorted(ownership_data["ticker"].dropna().astype(str).unique()),
                "5% Ownership": sorted(daily_owner_data["ticker"].dropna().astype(str).unique()),
                "Classification": sorted(classification_data["ticker"].dropna().astype(str).unique()),
                "Type": sorted(type_data["ticker"].dropna().astype(str).unique()),
            }.get(active_page, stock_options)
            if not page_stocks:
                st.error(f"No stock codes are available for {active_page}.")
                st.stop()
            if st.session_state.get("selected_stock") not in page_stocks:
                st.session_state["selected_stock"] = "AADI" if "AADI" in page_stocks else page_stocks[0]
            selected_entity = st.selectbox(
                "Stock",
                page_stocks,
                key="selected_stock",
                format_func=lambda ticker: f"{ticker} · {stock_names.get(ticker, '')}".rstrip(" ·"),
            )

    active_metrics = (
        daily_selection_metrics(daily_owner_data, analyze_by, selected_entity)
        if active_page == "5% Ownership"
        else selection_metrics(ownership_data, analyze_by, selected_entity)
    )
    if active_page == "5% Ownership":
        if analyze_by == "Stock":
            available_dates = daily_owner_data.loc[
                daily_owner_data["ticker"].eq(selected_entity), ["date"]
            ]
        else:
            available_dates = daily_owner_data.loc[
                daily_owner_data["owner_normalized"].eq(normalize_identity(selected_entity)),
                ["date"],
            ]
    elif analyze_by == "Stock":
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
            ownership_data["investor_name"].eq(selected_entity), ["date"]
        ]
    if active_page == "5% Ownership":
        context_name = (
            f"{selected_entity} · {stock_names.get(selected_entity, '')}".rstrip(" ·")
            if analyze_by == "Stock"
            else selected_entity
        )
    coverage_label = (
        "No history available"
        if available_dates.empty
        else f"{available_dates['date'].min():%b-%y} to {available_dates['date'].max():%b-%y}"
    )
    identity_code = selected_entity if analyze_by == "Stock" else "OWNER"
    identity_name = stock_names.get(selected_entity, "") if analyze_by == "Stock" else selected_entity
    render_page_header(
        active_page.upper(),
        context_name,
        f"{analyze_by} view · {coverage_label}",
        active_metrics["latest_period"],
    )
    render_terminal_header(identity_code, identity_name, analyze_by, active_metrics)


if active_page == "1% Ownership":
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


if active_page == "5% Ownership":
    section(
        "5% Ownership · Daily KSEI",
        f"{context_name} · beneficial-owner and securities-account monitoring",
        "WHO changed ownership is measured from combined beneficial ownership; WHERE shares moved is shown separately by securities account.",
    )
    requested_daily_subtab = str(st.query_params.get("subtab", "")).strip().lower()
    daily_subtab_default = {
        "beneficial": "Summary",
        "summary": "Summary",
        "latest": "Latest Account Position",
        "account": "Account Movement",
        "daily": "Summary",
    }.get(requested_daily_subtab, "Summary")
    owner_subtab, position_subtab, account_movement_subtab = st.tabs(
        [
            "Summary",
            "Latest Account Position",
            "Account Movement",
        ],
        default=daily_subtab_default,
        key="daily_ownership_subtabs",
    )
    if daily_owner_data.empty or daily_movements.empty:
        with owner_subtab:
            st.info(
                "No daily >5% ownership data is available. Add KSEI files to BEI_Data/5% Ownership."
            )
    else:
        if analyze_by == "Stock":
            daily_ticker = str(selected_entity)
            scoped_owner_normalized = None
            ticker_movements = daily_movements[
                daily_movements["ticker"].eq(daily_ticker)
            ].copy()
        else:
            scoped_owner_normalized = normalize_identity(selected_entity)
            owner_tickers = sorted(
                daily_owner_data.loc[
                    daily_owner_data["owner_normalized"].eq(scoped_owner_normalized),
                    "ticker",
                ].dropna().astype(str).unique()
            )
            if not owner_tickers:
                st.info(f"No daily >5% ownership history is available for {selected_entity}.")
                daily_ticker = ""
            else:
                daily_ticker = owner_tickers[0]
            ticker_movements = daily_movements[
                daily_movements["owner_normalized"].eq(scoped_owner_normalized)
            ].copy()

        if ticker_movements.empty:
            with owner_subtab:
                st.info(f"No daily >5% ownership movement is available for {selected_entity}.")
            with position_subtab:
                st.info(f"No securities-account positions are available for {selected_entity}.")
            with account_movement_subtab:
                st.info(f"No securities-account movement is available for {selected_entity}.")
        else:
            movement_dates = sorted(
                pd.Timestamp(value) for value in ticker_movements["date"].dropna().unique()
            )

            with owner_subtab:
                control_date, control_sort, control_owner, control_metric = st.columns(
                    [1.1, 1.35, 2.4, 1.45], gap="small"
                )
                scope_key = f"{analyze_by}_{selected_entity}"
                with control_date:
                    selected_daily_date = pd.Timestamp(
                        st.selectbox(
                            "Report Date",
                            movement_dates,
                            index=len(movement_dates) - 1,
                            format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"),
                            key=f"daily_report_date_{scope_key}",
                        )
                    )
                snapshot = ticker_movements[
                    ticker_movements["date"].eq(selected_daily_date)
                ].copy()
                with control_sort:
                    sort_mode = st.selectbox(
                        "Sort Owners" if analyze_by == "Stock" else "Sort Stocks",
                        ["Largest absolute change", "Largest accumulation", "Largest selling", "Ownership percentage"],
                        key=f"daily_sort_{scope_key}",
                    )

                if analyze_by == "Stock":
                    focus_labels = (
                        snapshot[["owner_normalized", "owner"]]
                        .drop_duplicates("owner_normalized")
                        .set_index("owner_normalized")["owner"]
                        .to_dict()
                    )
                    focus_help = (
                        "Beneficial Owner is Nama Pemegang Saham. The combined investor "
                        "holding determines accumulation or selling."
                    )
                    focus_label = "Beneficial Owner ⓘ"
                else:
                    focus_rows = snapshot[["ticker", "issuer"]].drop_duplicates("ticker")
                    focus_labels = {
                        str(row["ticker"]): (
                            f"{row['ticker']} · {row['issuer']}".rstrip(" ·")
                        )
                        for _, row in focus_rows.iterrows()
                    }
                    focus_help = "Select one stock for the detailed trend within this owner's portfolio."
                    focus_label = "Stock"
                focus_keys = list(focus_labels)
                largest_row = snapshot.sort_values("current_shares", ascending=False).iloc[0]
                preferred_focus = (
                    str(largest_row["owner_normalized"])
                    if analyze_by == "Stock"
                    else str(largest_row["ticker"])
                )
                with control_owner:
                    selected_focus = st.selectbox(
                        focus_label,
                        focus_keys,
                        index=focus_keys.index(preferred_focus),
                        format_func=lambda value: focus_labels[value],
                        help=focus_help,
                        key=f"daily_focus_picker_{scope_key}_{selected_daily_date:%Y%m%d}",
                    )
                if analyze_by == "Stock":
                    focused_ticker = daily_ticker
                    selected_daily_owner = selected_focus
                else:
                    focused_ticker = selected_focus
                    selected_daily_owner = scoped_owner_normalized
                with control_metric:
                    trend_metric = st.selectbox(
                        "Trend Metric",
                        ["Shares", "Ownership %", "Daily Δ Shares"],
                        key=f"daily_metric_{scope_key}",
                    )

                if sort_mode == "Largest accumulation":
                    snapshot = snapshot.sort_values(["delta_shares", "current_pct"], ascending=[False, False])
                elif sort_mode == "Largest selling":
                    snapshot = snapshot.sort_values(["delta_shares", "current_pct"], ascending=[True, False])
                elif sort_mode == "Ownership percentage":
                    snapshot = snapshot.sort_values("current_pct", ascending=False)
                else:
                    snapshot = snapshot.assign(_magnitude=snapshot["delta_shares"].abs()).sort_values(
                        ["_magnitude", "current_pct"], ascending=[False, False]
                    )

                render_daily_summary(selected_daily_date, snapshot)
                selected_history = owner_daily_history(
                    daily_owner_data,
                    daily_movements,
                    focused_ticker,
                    selected_daily_owner,
                )
                focus_display = focus_labels[selected_focus]
                render_owner_metric_strip(selected_history, focus_display)
                trend_column, monitor_column = st.columns([2.35, 1], gap="small")
                with trend_column:
                    st.plotly_chart(
                        daily_owner_trend_chart(
                            selected_history,
                            trend_metric,
                            f"{focus_display} · daily position",
                        ),
                        width="stretch",
                        config=PLOT_CONFIG,
                        key=f"daily_owner_trend_{focused_ticker}_{selected_daily_owner}_{trend_metric}",
                    )
                with monitor_column:
                    render_daily_owner_monitor(snapshot, analyze_by)

                table_heading(
                    "Beneficial-owner movement" if analyze_by == "Stock" else "Stock ownership movement",
                    separated=True,
                )
                identity_columns = ["owner"] if analyze_by == "Stock" else ["ticker", "issuer"]
                movement_display = snapshot[
                    identity_columns
                    + [
                        "previous_shares", "current_shares", "delta_shares",
                        "previous_pct", "current_pct", "delta_pct_point", "signal",
                        "number_of_accounts", "local_foreign",
                    ]
                ].rename(
                    columns={
                        "owner": "Owner",
                        "ticker": "Ticker",
                        "issuer": "Company",
                        "previous_shares": "Previous Shares",
                        "current_shares": "Current Shares",
                        "delta_shares": "Δ Shares",
                        "previous_pct": "Previous %",
                        "current_pct": "Current %",
                        "delta_pct_point": "Δ pp",
                        "signal": "Signal",
                        "number_of_accounts": "Accounts",
                        "local_foreign": "Local / Foreign",
                    }
                )
                movement_display["Signal"] = movement_display["Signal"].map(daily_signal_label)
                render_activity_dataframe(
                    movement_display,
                    f"daily_summary_table_{scope_key}_{selected_daily_date:%Y%m%d}_{sort_mode}",
                    {
                        "Previous Shares": "{:,.0f}",
                        "Current Shares": "{:,.0f}",
                        "Δ Shares": signed_shares,
                        "Previous %": "{:.2f}%",
                        "Current %": "{:.2f}%",
                        "Δ pp": signed_points,
                        "Accounts": "{:,.0f}",
                    },
                    ["Δ Shares", "Δ pp"],
                    ["Owner"] if analyze_by == "Stock" else ["Ticker", "Company"],
                    max_height=390,
                )

                selected_owner_movement = ticker_movements[
                    ticker_movements["date"].eq(selected_daily_date)
                    & ticker_movements["ticker"].eq(focused_ticker)
                    & ticker_movements["owner_normalized"].eq(selected_daily_owner)
                ]
                selected_account_changes = daily_account_movements[
                    daily_account_movements["ticker"].eq(focused_ticker)
                    & daily_account_movements["date"].eq(selected_daily_date)
                    & daily_account_movements["owner_normalized"].eq(selected_daily_owner)
                ]
                if not selected_owner_movement.empty:
                    render_account_interpretation(
                        selected_owner_movement.iloc[0], selected_account_changes
                    )

            with position_subtab:
                if analyze_by == "Stock":
                    ticker_accounts = daily_account_data[
                        daily_account_data["ticker"].eq(daily_ticker)
                    ].copy()
                else:
                    ticker_accounts = daily_account_data[
                        daily_account_data["owner_normalized"].eq(scoped_owner_normalized)
                    ].copy()
                position_dates = sorted(
                    pd.Timestamp(value) for value in ticker_accounts["date"].dropna().unique()
                )
                position_control, position_owner_control = st.columns([1, 2.4], gap="small")
                with position_control:
                    position_date = pd.Timestamp(
                        st.selectbox(
                            "Position Date",
                            position_dates,
                            index=len(position_dates) - 1,
                            format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"),
                            key=f"account_position_date_{scope_key}",
                        )
                    )
                position_owner_rows = ticker_accounts[ticker_accounts["date"].eq(position_date)]
                if analyze_by == "Stock":
                    position_owner_labels = (
                        position_owner_rows[["owner_normalized", "owner"]]
                        .drop_duplicates("owner_normalized")
                        .set_index("owner_normalized")["owner"]
                        .to_dict()
                    )
                    position_owner_keys = list(position_owner_labels)
                    with position_owner_control:
                        position_owner = st.selectbox(
                            "Beneficial Owner",
                            position_owner_keys,
                            format_func=lambda value: position_owner_labels[value],
                            key=f"account_position_owner_{scope_key}_{position_date:%Y%m%d}",
                            help="Select whose latest distribution across securities accounts is shown.",
                        )
                    positions = position_owner_rows[
                        position_owner_rows["owner_normalized"].eq(position_owner)
                        & position_owner_rows["shares"].fillna(0).gt(.5)
                    ].copy()
                    group_columns = ["account_holder", "account_name"]
                    position_title = position_owner_labels[position_owner]
                    display_identity = []
                else:
                    positions = position_owner_rows[
                        position_owner_rows["shares"].fillna(0).gt(.5)
                    ].copy()
                    positions["issuer"] = positions["issuer"].fillna("")
                    group_columns = ["ticker", "issuer", "account_holder", "account_name"]
                    position_title = selected_entity
                    display_identity = ["ticker", "issuer"]
                positions = (
                    positions.groupby(group_columns, observed=True, as_index=False)["shares"]
                    .sum(min_count=1)
                    .sort_values("shares", ascending=False)
                )
                table_heading(
                    f"Latest account position · {position_title} · {position_date:%d %b %Y}"
                )
                st.caption(
                    "Account Holder is Nama Pemegang Rekening Efek (custodian/securities institution). "
                    "Account Name is Nama Rekening Efek. This is a current-position view, not movement."
                )
                position_display = positions[
                    display_identity + ["account_holder", "account_name", "shares"]
                ].rename(
                    columns={
                        "ticker": "Ticker",
                        "issuer": "Company",
                        "account_holder": "Nama Pemegang Rekening Efek",
                        "account_name": "Nama Rekening Efek",
                        "shares": "Shares",
                    }
                )
                render_activity_dataframe(
                    position_display,
                    f"latest_account_positions_{scope_key}_{position_date:%Y%m%d}",
                    {"Shares": "{:,.0f}"},
                    [],
                    (["Ticker", "Company"] if analyze_by == "Owner" else [])
                    + ["Nama Pemegang Rekening Efek", "Nama Rekening Efek"],
                    max_height=620,
                )

            with account_movement_subtab:
                if analyze_by == "Stock":
                    account_scope = daily_account_data[
                        daily_account_data["ticker"].eq(daily_ticker)
                    ].copy()
                else:
                    account_scope = daily_account_data[
                        daily_account_data["owner_normalized"].eq(scoped_owner_normalized)
                    ].copy()
                account_owner_labels = (
                    account_scope[["owner_normalized", "owner"]]
                    .drop_duplicates("owner_normalized")
                    .set_index("owner_normalized")["owner"]
                    .to_dict()
                )
                account_owner_keys = list(account_owner_labels)
                if not account_owner_keys:
                    st.info("No account-level history is available for this selection.")
                else:
                    account_dates = sorted(
                        pd.Timestamp(value) for value in account_scope["date"].dropna().unique()
                    )
                    movement_range_control, dimension_control, movement_owner_control = st.columns(
                        [1.55, 1.25, 2.7], gap="small"
                    )
                    with movement_range_control:
                        account_date_range = st.date_input(
                            "Date Range",
                            value=(account_dates[0].date(), account_dates[-1].date()),
                            min_value=account_dates[0].date(),
                            max_value=account_dates[-1].date(),
                            key=f"account_movement_range_{scope_key}",
                        )
                    dimension_options = {
                        "Institution": "institution",
                        "Account Name": "account",
                    }
                    with dimension_control:
                        dimension_label = st.selectbox(
                            "Account Dimension",
                            list(dimension_options),
                            key=f"account_dimension_{scope_key}",
                        )
                    if analyze_by == "Stock":
                        with movement_owner_control:
                            selected_account_owners = st.multiselect(
                                "Owner Filter",
                                account_owner_keys,
                                format_func=lambda value: account_owner_labels[value],
                                placeholder="All beneficial owners",
                                key=f"account_owner_filter_{scope_key}",
                                help="Leave empty to show every beneficial owner for the selected ticker.",
                            )
                        position_pivot, movement_pivot = build_account_hierarchy_pivots(
                            daily_account_data,
                            daily_ticker,
                            dimension_options[dimension_label],
                            selected_account_owners or None,
                        )
                        hierarchy_label = "Beneficial Owner"
                        owner_filter_count = len(selected_account_owners)
                    else:
                        selected_account_owners = [scoped_owner_normalized]
                        position_pivot, movement_pivot = build_owner_account_hierarchy_pivots(
                            daily_account_data,
                            scoped_owner_normalized,
                            dimension_options[dimension_label],
                        )
                        hierarchy_label = "Ticker"
                        owner_filter_count = 1
                    if isinstance(account_date_range, (tuple, list)) and len(account_date_range) == 2:
                        range_start, range_end = map(pd.Timestamp, account_date_range)
                    else:
                        range_start = range_end = pd.Timestamp(account_date_range)
                    position_view = position_pivot.loc[
                        (position_pivot.index >= range_start)
                        & (position_pivot.index <= range_end)
                    ]
                    movement_view = movement_pivot.loc[
                        (movement_pivot.index >= range_start)
                        & (movement_pivot.index <= range_end)
                    ]
                    active_columns = [
                        column
                        for column in position_view.columns
                        if position_view[column].abs().gt(.5).any()
                        or movement_view[column].abs().gt(.5).any()
                    ]
                    position_view = position_view.loc[:, active_columns]
                    movement_view = movement_view.loc[:, active_columns]

                    st.markdown(
                        '<div class="account-definition"><b>ACCOUNT MOVEMENT = WHERE SHARES MOVED</b>'
                        '<span>Built from account-level Jumlah Saham; an account increase or decrease does not by itself mean the owner accumulated or sold.</span></div>',
                        unsafe_allow_html=True,
                    )
                    table_heading(
                        f"Account Position · Date × {hierarchy_label} × {dimension_label}",
                        separated=True,
                    )
                    render_account_pivot(
                        position_view,
                        f"account_position_pivot_{scope_key}_{dimension_label}_{owner_filter_count}_{range_start:%Y%m%d}_{range_end:%Y%m%d}",
                    )
                    table_heading(
                        "Account Daily Change · Current Position − Previous Position",
                        separated=True,
                    )
                    st.caption(
                        "Positive values show shares moving into or increasing in an account; negative values show shares moving out or decreasing."
                    )
                    render_account_pivot(
                        movement_view,
                        f"account_change_pivot_{scope_key}_{dimension_label}_{owner_filter_count}_{range_start:%Y%m%d}_{range_end:%Y%m%d}",
                        movement=True,
                    )

            if not daily_quality.empty:
                with st.expander(
                    f"Data quality and audit log · {len(daily_quality):,} warning(s)",
                    expanded=False,
                ):
                    st.dataframe(daily_quality, width="stretch", hide_index=True)


if active_page == "Classification":
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


if active_page == "Type":
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
                        color_map={"Domestic": "#45B820", "Foreign": "#4D75D7"},
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


if active_page == "Entity Movement":
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


if active_page == "Overview" and overview_section == "1% Monthly Changes":
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
            change_detail, _, _ = monthly_change_analysis(
                ownership_data
            )
        period_detail = change_detail[
            change_detail["date"].eq(selected_change_month)
            & change_detail["is_changed"]
        ].copy()
        overview_filters = st.columns([1.5, 2.2, 2.2, 1.35], gap="small")
        with overview_filters[0]:
            st.text_input(
                "Market Scope",
                value="ALL IDX TICKERS",
                disabled=True,
                key=f"overview_scope_{selected_change_month:%Y%m}",
            )
        with overview_filters[1]:
            overview_tickers = st.multiselect(
                "Ticker Filter",
                sorted(period_detail["stock"].dropna().astype(str).unique()),
                placeholder="All tickers",
                key=f"overview_tickers_{selected_change_month:%Y%m}",
            )
        with overview_filters[2]:
            overview_owners = st.multiselect(
                "Owner Filter",
                sorted(period_detail["owner"].dropna().astype(str).unique()),
                placeholder="All owners",
                key=f"overview_owners_{selected_change_month:%Y%m}",
            )
        with overview_filters[3]:
            overview_minimum_change = st.number_input(
                "Minimum |Δ Shares|",
                min_value=0.0,
                step=1_000_000.0,
                value=0.0,
                format="%.0f",
                key=f"overview_min_change_{selected_change_month:%Y%m}",
            )
        period_detail = filter_market_overview(
            period_detail,
            tickers=overview_tickers,
            owners=overview_owners,
            minimum_absolute_change=overview_minimum_change,
        )
        period_stocks = stock_change_summary(period_detail)
        period_owners = owner_change_summary(period_detail)

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


if active_page == "Overview" and overview_section == "5% Daily Movement":
    if daily_movements.empty:
        st.info("No daily >5% movement data is available.")
    else:
        section(
            "Market-wide scanner",
            "Daily >5% Movement",
            "Beneficial-owner changes are separated from transfers between securities accounts.",
        )
        all_daily_dates = sorted(
            pd.Timestamp(value) for value in daily_movements["date"].dropna().unique()
        )
        filter_row_1 = st.columns([1.15, 1.35, 1.8, 1.5], gap="small")
        with filter_row_1[0]:
            scanner_date = pd.Timestamp(
                st.selectbox(
                    "Date",
                    all_daily_dates,
                    index=len(all_daily_dates) - 1,
                    format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"),
                    key="daily_scanner_date",
                )
            )
        scanner_base = daily_movements[daily_movements["date"].eq(scanner_date)].copy()
        with filter_row_1[1]:
            ticker_filter = st.multiselect(
                "Ticker",
                sorted(scanner_base["ticker"].dropna().astype(str).unique()),
                placeholder="All tickers",
                key="daily_scanner_tickers",
            )
        with filter_row_1[2]:
            owner_filter = st.multiselect(
                "Beneficial Owner",
                sorted(scanner_base["owner"].dropna().astype(str).unique()),
                placeholder="All owners",
                key="daily_scanner_owners",
            )
        with filter_row_1[3]:
            signal_filter = st.multiselect(
                "Signal",
                sorted(scanner_base["signal"].dropna().astype(str).unique()),
                placeholder="All signals",
                format_func=daily_signal_label,
                key="daily_scanner_signals",
            )

        date_account_movements = daily_account_movements[
            daily_account_movements["date"].eq(scanner_date)
        ]
        filter_row_2 = st.columns([1.8, 1.2, 1, 1], gap="small")
        with filter_row_2[0]:
            account_holder_filter = st.multiselect(
                "Nama Pemegang Rekening Efek",
                sorted(date_account_movements["account_holder"].dropna().astype(str).unique()),
                placeholder="All account institutions",
                help="Filters owner movements to cases involving the selected securities institution or custodian.",
                key="daily_scanner_account_holders",
            )
        with filter_row_2[1]:
            residency_filter = st.multiselect(
                "Local / Foreign",
                sorted(scanner_base["local_foreign"].dropna().astype(str).unique()),
                placeholder="All",
                key="daily_scanner_residency",
            )
        with filter_row_2[2]:
            minimum_shares = st.number_input(
                "Minimum |Δ Shares|",
                min_value=0.0,
                step=1_000_000.0,
                value=0.0,
                format="%.0f",
                key="daily_scanner_min_shares",
            )
        with filter_row_2[3]:
            minimum_pct = st.number_input(
                "Minimum |Δ pp|",
                min_value=0.0,
                step=0.01,
                value=0.0,
                format="%.2f",
                key="daily_scanner_min_pct",
            )

        scanner = scanner_base.copy()
        if ticker_filter:
            scanner = scanner[scanner["ticker"].isin(ticker_filter)]
        if owner_filter:
            scanner = scanner[scanner["owner"].isin(owner_filter)]
        if signal_filter:
            scanner = scanner[scanner["signal"].isin(signal_filter)]
        if residency_filter:
            scanner = scanner[scanner["local_foreign"].isin(residency_filter)]
        scanner = scanner[
            scanner["delta_shares"].abs().fillna(0).ge(minimum_shares)
            & scanner["delta_pct_point"].abs().fillna(0).ge(minimum_pct)
        ]
        if account_holder_filter:
            matching_accounts = date_account_movements[
                date_account_movements["account_holder"].isin(account_holder_filter)
            ][["ticker", "owner_normalized"]].drop_duplicates()
            scanner = scanner.merge(
                matching_accounts,
                on=["ticker", "owner_normalized"],
                how="inner",
            )

        scanner = scanner.assign(_magnitude=scanner["delta_shares"].abs()).sort_values(
            ["_magnitude", "ticker", "owner"], ascending=[False, True, True]
        ).reset_index(drop=True)
        render_daily_summary(scanner_date, scanner)
        st.caption(
            "No market-wide net-share total is shown because share units are not economically comparable across different companies."
        )

        table_heading("Major shareholder movements", separated=True)
        scanner_display = scanner[
            [
                "date",
                "ticker",
                "owner",
                "previous_shares",
                "current_shares",
                "delta_shares",
                "previous_pct",
                "current_pct",
                "delta_pct_point",
                "signal",
                "accounts_changed",
            ]
        ].rename(
            columns={
                "date": "Date",
                "ticker": "Ticker",
                "owner": "Owner",
                "previous_shares": "Previous Shares",
                "current_shares": "Current Shares",
                "delta_shares": "Δ Shares",
                "previous_pct": "Previous %",
                "current_pct": "Current %",
                "delta_pct_point": "Δ pp",
                "signal": "Signal",
                "accounts_changed": "Accounts Changed",
            }
        )
        scanner_display["Signal"] = scanner_display["Signal"].map(daily_signal_label)
        scanner_rows = render_activity_dataframe(
            scanner_display,
            f"daily_market_scanner_{scanner_date:%Y%m%d}_{len(scanner)}",
            {
                "Date": lambda value: pd.Timestamp(value).strftime("%d %b %Y"),
                "Previous Shares": "{:,.0f}",
                "Current Shares": "{:,.0f}",
                "Δ Shares": signed_shares,
                "Previous %": "{:.2f}%",
                "Current %": "{:.2f}%",
                "Δ pp": signed_points,
                "Accounts Changed": "{:,.0f}",
            },
            ["Δ Shares", "Δ pp"],
            ["Ticker", "Owner"],
            selectable=True,
            max_height=520,
        )

        if scanner.empty:
            st.info("No daily >5% movements match the active filters.")
        else:
            selected_scanner_index = scanner_rows[0] if scanner_rows else 0
            drill_row = scanner.iloc[selected_scanner_index]
            drill_ticker = str(drill_row["ticker"])
            drill_owner = str(drill_row["owner_normalized"])
            drill_accounts = date_account_movements[
                date_account_movements["ticker"].eq(drill_ticker)
                & date_account_movements["owner_normalized"].eq(drill_owner)
            ].copy()
            table_heading(
                f"{drill_ticker} · {drill_row['owner']} · movement drill-down",
                separated=True,
            )
            render_account_interpretation(drill_row, drill_accounts)
            drill_chart_column, drill_table_column = st.columns([1.15, 1], gap="small")
            with drill_chart_column:
                drill_history = owner_daily_history(
                    daily_owner_data, daily_movements, drill_ticker, drill_owner
                )
                st.plotly_chart(
                    daily_owner_trend_chart(
                        drill_history,
                        "Shares",
                        f"{drill_row['owner']} · combined ownership",
                    ),
                    width="stretch",
                    config=PLOT_CONFIG,
                    key=f"scanner_drill_chart_{scanner_date:%Y%m%d}_{drill_ticker}_{drill_owner}",
                )
            with drill_table_column:
                changed_accounts = drill_accounts[
                    drill_accounts["delta_shares"].abs().gt(.5)
                ].copy()
                changed_accounts = changed_accounts.assign(
                    _magnitude=changed_accounts["delta_shares"].abs()
                ).sort_values("_magnitude", ascending=False)
                drill_display = changed_accounts[
                    ["account_holder", "account_name", "previous_shares", "current_shares", "delta_shares", "direction"]
                ].rename(
                    columns={
                        "account_holder": "Account Holder",
                        "account_name": "Account Name",
                        "previous_shares": "Previous",
                        "current_shares": "Current",
                        "delta_shares": "Δ Shares",
                        "direction": "Direction",
                    }
                )
                render_activity_dataframe(
                    drill_display,
                    f"scanner_drill_accounts_{scanner_date:%Y%m%d}_{drill_ticker}_{drill_owner}",
                    {
                        "Previous": "{:,.0f}",
                        "Current": "{:,.0f}",
                        "Δ Shares": signed_shares,
                    },
                    ["Δ Shares"],
                    ["Account Holder", "Account Name"],
                    max_height=350,
                )
