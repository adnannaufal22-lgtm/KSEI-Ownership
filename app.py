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
    initial_sidebar_state="expanded",
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


def table_heading(heading: str) -> None:
    st.markdown(
        f'<div class="table-heading">{escape(heading)}</div>',
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
            return "background-color:#fef9c3;color:#111827;"
        ratio = min(abs(numeric) / heatmap_scale, 1.0) if heatmap_scale else 0.0
        opacity = 0.14 + 0.38 * ratio**0.5
        rgb = "34,197,94" if numeric > 0 else "239,68,68"
        return f"background-color:rgba({rgb},{opacity:.3f});color:#111827;"

    header_cells = "".join(
        f"<th>{escape(str(column))}</th>" for column in [row_label, *value_columns]
    )
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

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<span class="information-mark">i</span>'
        '<div><div class="sidebar-title">KSEI Ownership Dashboard</div>'
        '<div class="sidebar-subtitle">Historical ownership movement</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    analyze_by = st.selectbox(
        "Analyze By",
        ["Stock", "Owner"],
        key="analysis_mode",
    )
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

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.caption(
        f"Automatic history: {ownership_metadata.get('source_files', 0)} ownership · "
        f"{classification_metadata.get('source_files', 0)} classification · "
        f"{type_metadata.get('source_files', 0)} type files"
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

st.markdown(
    '<div class="dashboard-header">'
    '<div class="dashboard-heading-group">'
    '<span class="information-mark main-mark">i</span>'
    '<div><h1 class="dashboard-title">KSEI Ownership Dashboard</h1>'
    '<div class="dashboard-subtitle">Ownership movement by stock or owner</div></div>'
    "</div>"
    f'<div class="selection-context"><div class="context-label">{analyze_by}</div>'
    f'<div class="context-value">{context_name}</div>'
    f'<div class="context-period">{coverage_label}</div></div>'
    "</div>",
    unsafe_allow_html=True,
)


ownership_tab, classification_tab, type_tab, monthly_change_tab = st.tabs(
    ["1% Ownership", "Classification", "Type", "Monthly Change"]
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
            "Each line is a reported holder." if analyze_by == "Stock" else "Each line is a stock held by this owner.",
        )
        st.plotly_chart(
            ownership_movement_lines(
                history,
                f"{context_name} · Number of shares held",
                ownership_dates,
            ),
            width="stretch",
            config=PLOT_CONFIG,
            key=f"ownership_lines_{analyze_by}_{selected_entity}",
        )

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
        )

        table_heading("Monthly change (%)")
        render_pivot(
            ownership_change,
            "series_label",
            row_label,
            "signed_pct",
            f"ownership_change_pivot_{analyze_by}_{selected_entity}",
            holder_links=analyze_by == "Stock",
            heatmap=True,
        )

        table_heading("Ownership percentage")
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

            table_heading("Monthly change (%)")
            render_pivot(
                classification_change,
                "classification",
                "Classification",
                "signed_pct",
                f"classification_change_{selected_entity}",
                heatmap=True,
            )

            if classification_history["ownership_pct"].notna().any():
                table_heading("Percentage of scripless ownership")
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
                table_heading("Monthly change (%)")
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
                        color_map={"Domestic": "#0F766E", "Foreign": "#2563EB"},
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
                table_heading("Monthly change (%)")
                render_pivot(
                    residency_change,
                    "domestic_foreign",
                    "Type",
                    "signed_pct",
                    f"residency_change_{selected_entity}",
                    compact=True,
                    heatmap=True,
                )
                table_heading("Percentage of total shares")
                render_pivot(
                    residency_pct,
                    "domestic_foreign",
                    "Type",
                    "%.2f%%",
                    f"residency_pct_{selected_entity}",
                    compact=True,
                )
            st.caption("Domestic and Foreign reconcile to Total Scripless; percentages use Number of Shares as the denominator.")


with monthly_change_tab:
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
