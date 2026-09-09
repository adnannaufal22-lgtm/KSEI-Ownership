from __future__ import annotations

import json
import logging
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
    movement_pivot_table,
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


def render_pivot(
    pivot: pd.DataFrame,
    source_row_column: str,
    row_label: str,
    number_format: str,
    key: str,
    holder_links: bool = False,
) -> None:
    if pivot.empty:
        st.info("No pivot data is available for this selection.")
        return
    view = pivot.rename(columns={source_row_column: row_label}).copy()
    column_config: dict = {}
    if holder_links:
        view[row_label] = view[row_label].map(
            lambda holder: f"?holder={quote(str(holder), safe='')}#holder={holder}"
        )
        column_config[row_label] = st.column_config.LinkColumn(
            row_label,
            pinned=True,
            display_text=r".*#holder=(.*)",
            help="Open this owner across all reported stocks.",
        )
    else:
        column_config[row_label] = st.column_config.TextColumn(row_label, pinned=True)
    for column in view.columns:
        if column == row_label:
            continue
        if number_format == "comma":
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:,.0f}"
            )
            column_config[column] = st.column_config.TextColumn(column)
        else:
            column_config[column] = st.column_config.NumberColumn(
                column,
                format=number_format,
            )
    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        height=min(540, max(180, 42 + len(view) * 34)),
        column_config=column_config,
        key=key,
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

        section("Pivot 1", "Number of shares")
        shares_pivot = historical_pivot(
            history,
            "series_label",
            "ownership_units",
            ownership_dates,
        )
        render_pivot(
            shares_pivot,
            "series_label",
            row_label,
            "comma",
            f"ownership_shares_pivot_{analyze_by}_{selected_entity}",
            holder_links=analyze_by == "Stock",
        )

        section("Pivot 2", "Ownership percentage")
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

            section("Pivot 1", "Number of shares by classification")
            classification_shares = historical_pivot(
                classification_history,
                "classification",
                "ownership_units",
                classification_dates,
            )
            render_pivot(
                classification_shares,
                "classification",
                "Classification",
                "comma",
                f"classification_shares_{selected_entity}",
            )

            if classification_history["ownership_pct"].notna().any():
                section("Pivot 2", "Percentage of scripless ownership")
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
            section(
                "Type · Share form",
                f"{context_name} scrip versus scripless movement",
                "Scrip shares equal Number of Shares minus Total Scripless.",
            )
            if type_summary["scrip_shares"].isna().any():
                st.warning(
                    "Scrip shares are unavailable for one or more periods because Total Scripless exceeds Number of Shares in the source."
                )
            st.plotly_chart(
                scrip_vs_scripless_chart(
                    type_summary,
                    f"{context_name} · Scrip and scripless shares",
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
            type_dates = sorted(pd.Timestamp(value) for value in type_data["date"].dropna().unique())
            st.plotly_chart(
                stacked_area_line_chart(
                    residency_history,
                    "domestic_foreign",
                    f"{context_name} · Domestic and foreign ownership",
                    "Number of scripless shares",
                    type_dates,
                    color_map={"Domestic": "#0F766E", "Foreign": "#2563EB"},
                ),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"residency_chart_{selected_entity}",
            )

            section("Pivot 1", "Number of shares by residency")
            residency_shares = historical_pivot(
                residency_history,
                "domestic_foreign",
                "ownership_units",
                type_dates,
            )
            render_pivot(
                residency_shares,
                "domestic_foreign",
                "Type",
                "comma",
                f"residency_shares_{selected_entity}",
            )

            section("Pivot 2", "Percentage of total shares")
            residency_pct = historical_pivot(
                residency_history,
                "domestic_foreign",
                "ownership_pct",
                type_dates,
            )
            render_pivot(
                residency_pct,
                "domestic_foreign",
                "Type",
                "%.2f%%",
                f"residency_pct_{selected_entity}",
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

        section("Pivot", f"Monthly change by {counterparty_label.lower()}")
        change_pivot = movement_pivot_table(
            breakdown,
            movement_metric,
            "Monthly change",
        ).rename(columns={"counterparty": counterparty_label})
        render_pivot(
            change_pivot,
            counterparty_label,
            counterparty_label,
            "comma" if movement_metric == "Reported shares" else "%+.2f",
            f"monthly_change_pivot_{analyze_by}_{selected_entity}",
            holder_links=counterparty_label == "Holder",
        )
        st.caption("The first observed period is blank because it has no preceding comparison month.")
