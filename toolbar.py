from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from data_processing import (
    apply_global_filters,
    period_slice,
    resolve_comparison_period,
    sort_ticker_snapshot,
    ticker_snapshot,
)


@dataclass
class ToolbarState:
    selected_period: pd.Timestamp
    comparison_mode: str
    comparison_period: pd.Timestamp | None
    ticker_filter: list[str]
    holder_filter: list[str]
    category_filter: list[str]
    residency_filter: list[str]
    institution_filter: list[str]
    sector_filter: list[str]
    minimum_holding_pct: float
    display_mode: str
    sort_mode: str
    sort_ascending: bool
    filtered_history: pd.DataFrame
    current: pd.DataFrame
    previous: pd.DataFrame


def render_global_toolbar(
    data: pd.DataFrame,
    periods: list[pd.Timestamp],
    source_name: str,
) -> ToolbarState:
    """Render the compact toolbar and return the single global-filter result."""
    latest_period = periods[-1]
    comparison_options = ["Previous period", "Previous month", "Previous quarter", "Previous year", "Custom period"]
    category_options = sorted(data["investor_category"].dropna().unique())
    residency_options = sorted(data["domestic_foreign"].dropna().unique())
    institution_options = sorted(data["institutional_individual"].dropna().unique())
    sector_options = sorted(data["sector"].dropna().unique()) if data["sector"].notna().any() else []
    default_custom_date = periods[-2].date() if len(periods) > 1 else latest_period.date()

    filter_defaults = {
        "selected_period": latest_period,
        "comparison_mode": "Previous period",
        "ticker_filter": [],
        "holder_filter": [],
        "minimum_holding_pct": 1.0,
        "category_filter": [],
        "residency_filter": [],
        "institution_filter": [],
        "sector_filter": [],
        "custom_comparison": default_custom_date,
    }
    for state_key, default_value in filter_defaults.items():
        st.session_state.setdefault(state_key, default_value)
    st.session_state.setdefault("display_mode", "Exact")
    st.session_state.setdefault("sort_mode", "Ticker")
    st.session_state.setdefault("sort_ascending", True)
    st.session_state.setdefault("global_search_selection", None)

    if pd.Timestamp(st.session_state["selected_period"]) not in periods:
        st.session_state["selected_period"] = latest_period

    valid_option_sets = {
        "ticker_filter": set(data["ticker"].dropna().unique()),
        "holder_filter": set(data["investor_name"].dropna().unique()),
        "category_filter": set(category_options),
        "residency_filter": set(residency_options),
        "institution_filter": set(institution_options),
        "sector_filter": set(sector_options),
    }
    for state_key, valid_values in valid_option_sets.items():
        st.session_state[state_key] = [value for value in st.session_state[state_key] if value in valid_values]

    active_to_draft = {
        "selected_period": "draft_selected_period",
        "comparison_mode": "draft_comparison_mode",
        "category_filter": "draft_category_filter",
        "residency_filter": "draft_residency_filter",
        "institution_filter": "draft_institution_filter",
        "sector_filter": "draft_sector_filter",
        "minimum_holding_pct": "draft_minimum_holding_pct",
        "custom_comparison": "draft_custom_comparison",
    }
    for active_key, draft_key in active_to_draft.items():
        st.session_state.setdefault(draft_key, st.session_state[active_key])

    def sync_draft(active_key: str) -> None:
        draft_key = active_to_draft.get(active_key)
        if not draft_key:
            return
        value = st.session_state[active_key]
        st.session_state[draft_key] = value.copy() if isinstance(value, list) else value

    def reset_filters() -> None:
        for state_key, default_value in filter_defaults.items():
            st.session_state[state_key] = default_value.copy() if isinstance(default_value, list) else default_value
        for active_key in active_to_draft:
            sync_draft(active_key)
        st.session_state["sort_mode"] = "Ticker"
        st.session_state["sort_ascending"] = True
        st.session_state["focus_movement_tab"] = False
        st.session_state["global_search_selection"] = None
        st.session_state.pop("active_filter_chips", None)

    def latest_shortcut() -> None:
        st.session_state["selected_period"] = latest_period
        sync_draft("selected_period")
        st.session_state.pop("active_filter_chips", None)

    def select_sort(sort_mode: str) -> None:
        if st.session_state["sort_mode"] == sort_mode and sort_mode == "Ticker":
            st.session_state["sort_ascending"] = not st.session_state["sort_ascending"]
        else:
            st.session_state["sort_mode"] = sort_mode
            st.session_state["sort_ascending"] = sort_mode == "Ticker"

    def remove_filter_chips(chip_actions: dict[str, tuple[str, object, str]]) -> None:
        selected_chips = set(st.session_state.get("active_filter_chips", []))
        for chip_id, (state_key, value, action_type) in chip_actions.items():
            if chip_id in selected_chips:
                continue
            if action_type == "list":
                st.session_state[state_key] = [item for item in st.session_state[state_key] if item != value]
            else:
                st.session_state[state_key] = value
            sync_draft(state_key)

    selected_period = pd.Timestamp(st.session_state["selected_period"])
    comparison_mode = st.session_state["comparison_mode"]
    custom_date: date | None = st.session_state["custom_comparison"] if comparison_mode == "Custom period" else None
    category_filter = st.session_state["category_filter"]
    residency_filter = st.session_state["residency_filter"]
    institution_filter = st.session_state["institution_filter"]
    sector_filter = st.session_state["sector_filter"]
    minimum_holding_pct = float(st.session_state["minimum_holding_pct"])
    ticker_filter = st.session_state["ticker_filter"]
    holder_filter = st.session_state["holder_filter"]

    advanced_filtered_history = apply_global_filters(
        data,
        categories=category_filter,
        residency=residency_filter,
        institution=institution_filter,
        sectors=sector_filter,
        minimum_ownership_pct=minimum_holding_pct,
    )
    comparison_period = resolve_comparison_period(periods, selected_period, comparison_mode, custom_date)
    search_current = period_slice(advanced_filtered_history, selected_period)
    search_previous = (
        period_slice(advanced_filtered_history, comparison_period)
        if comparison_period is not None
        else advanced_filtered_history.iloc[0:0]
    )
    search_snapshot = sort_ticker_snapshot(
        ticker_snapshot(search_current, search_previous),
        st.session_state["sort_mode"],
        st.session_state["sort_ascending"],
    )

    stock_names = (
        data[["ticker", "security_name"]]
        .drop_duplicates("ticker")
        .set_index("ticker")["security_name"]
        .to_dict()
    )
    search_result_map: dict[str, tuple[str, str]] = {}
    search_options: list[str] = []
    for ticker in search_snapshot["ticker"].dropna():
        if ticker in ticker_filter:
            continue
        label = f"STOCK  ·  {ticker} — {stock_names.get(ticker, '')}".rstrip(" —")
        search_result_map[label] = ("Stock", ticker)
        search_options.append(label)
    for holder in sorted(search_current["investor_name"].dropna().unique()):
        if holder in holder_filter:
            continue
        label = f"HOLDER  ·  {holder}"
        search_result_map[label] = ("Holder", holder)
        search_options.append(label)

    def apply_search_result() -> None:
        selection = st.session_state.get("global_search_selection")
        if not selection or selection not in search_result_map:
            return
        result_type, value = search_result_map[selection]
        state_key = "ticker_filter" if result_type == "Stock" else "holder_filter"
        if value not in st.session_state[state_key]:
            st.session_state[state_key] = [*st.session_state[state_key], value]
        st.session_state["movement_entity_type"] = result_type
        st.session_state["movement_stock" if result_type == "Stock" else "movement_holder"] = value
        st.session_state["focus_movement_tab"] = True
        st.session_state["global_search_selection"] = None
        st.session_state.pop("active_filter_chips", None)

    st.markdown(
        f"""
        <div class="dashboard-header">
          <div>
            <div class="dashboard-eyebrow">Indonesia · Securities ownership intelligence</div>
            <h1 class="dashboard-title">KSEI Ownership Dashboard</h1>
            <div class="dashboard-subtitle">Investor Ownership, Flow &amp; Concentration Analysis</div>
          </div>
          <div class="as-of">
            <div class="as-of-label">Reporting context</div>
            <div class="as-of-value">Per {selected_period.strftime('%d %b %Y')}</div>
            <div class="as-of-source">Source: KSEI</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.divider()
        st.markdown("### Display settings")
        st.selectbox("Number format", ["Exact", "Compact", "Million", "Billion"], key="display_mode")

    advanced_filter_count = sum(
        [
            selected_period != latest_period,
            comparison_mode != "Previous period",
            bool(category_filter),
            bool(residency_filter),
            bool(institution_filter),
            bool(sector_filter),
            minimum_holding_pct > 0,
        ]
    )

    with st.container(border=True, key="global_toolbar"):
        search_col, filter_col, sort_label_col, sort_col, latest_col, reset_col = st.columns(
            [5.2, 1.0, 0.75, 3.35, 0.85, 0.8],
            gap="small",
            vertical_alignment="center",
        )
        with search_col:
            st.selectbox(
                "Search securities, issuers, or holders",
                search_options,
                index=None,
                key="global_search_selection",
                placeholder="Search ticker, issuer, or holder…",
                label_visibility="collapsed",
                on_change=apply_search_result,
                filter_mode="fuzzy",
            )
        with filter_col:
            with st.popover(f"Filter · {advanced_filter_count}" if advanced_filter_count else "Filter", width="stretch"):
                st.markdown('<div class="popover-title">Advanced filters</div>', unsafe_allow_html=True)
                with st.form("advanced_filter_form", border=False):
                    af1, af2 = st.columns(2)
                    with af1:
                        st.selectbox(
                            "Report date",
                            periods,
                            key="draft_selected_period",
                            format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"),
                        )
                        st.multiselect(
                            "Sector",
                            sector_options,
                            key="draft_sector_filter",
                            placeholder="All sectors",
                            disabled=not sector_options,
                        )
                        st.multiselect(
                            "Domestic / foreign",
                            residency_options,
                            key="draft_residency_filter",
                            placeholder="All",
                        )
                        st.number_input(
                            "Minimum holding (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                            key="draft_minimum_holding_pct",
                        )
                    with af2:
                        st.selectbox("Compare with", comparison_options, key="draft_comparison_mode")
                        st.multiselect(
                            "Investor type",
                            category_options,
                            key="draft_category_filter",
                            placeholder="All categories",
                        )
                        st.multiselect(
                            "Institutional / individual",
                            institution_options,
                            key="draft_institution_filter",
                            placeholder="All",
                        )
                        st.date_input(
                            "Custom comparison date",
                            key="draft_custom_comparison",
                            help="Used only when Compare with is set to Custom period.",
                        )
                    apply_col, clear_col = st.columns(2)
                    apply_advanced = apply_col.form_submit_button("Apply", type="primary", width="stretch")
                    clear_advanced = clear_col.form_submit_button("Clear", width="stretch")
                if apply_advanced:
                    for active_key, draft_key in active_to_draft.items():
                        value = st.session_state[draft_key]
                        st.session_state[active_key] = value.copy() if isinstance(value, list) else value
                    st.session_state.pop("active_filter_chips", None)
                    st.rerun()
                if clear_advanced:
                    advanced_defaults = {
                        "selected_period": latest_period,
                        "comparison_mode": "Previous period",
                        "category_filter": [],
                        "residency_filter": [],
                        "institution_filter": [],
                        "sector_filter": [],
                        "minimum_holding_pct": 0.0,
                        "custom_comparison": default_custom_date,
                    }
                    for active_key, value in advanced_defaults.items():
                        st.session_state[active_key] = value.copy() if isinstance(value, list) else value
                        sync_draft(active_key)
                    st.session_state.pop("active_filter_chips", None)
                    st.rerun()
        with sort_label_col:
            st.markdown('<div class="toolbar-sort-label">SORT BY</div>', unsafe_allow_html=True)
        with sort_col:
            sort_a, sort_b, sort_c = st.columns(3, gap="small")
            ticker_arrow = "↑" if st.session_state["sort_ascending"] else "↓"
            sort_a.button(
                f"Ticker {ticker_arrow}",
                key="sort_ticker",
                type="primary" if st.session_state["sort_mode"] == "Ticker" else "secondary",
                on_click=select_sort,
                args=("Ticker",),
                width="stretch",
            )
            sort_b.button(
                "Holding",
                key="sort_holding",
                type="primary" if st.session_state["sort_mode"] == "Holding" else "secondary",
                on_click=select_sort,
                args=("Holding",),
                width="stretch",
            )
            sort_c.button(
                "Movement",
                key="sort_movement",
                type="primary" if st.session_state["sort_mode"] == "Movement" else "secondary",
                on_click=select_sort,
                args=("Movement",),
                width="stretch",
            )
        with latest_col:
            st.button("Latest", on_click=latest_shortcut, width="stretch")
        with reset_col:
            st.button("Reset", on_click=reset_filters, width="stretch")

        chip_labels: dict[str, str] = {}
        chip_actions: dict[str, tuple[str, object, str]] = {}

        def add_chip(chip_id: str, label: str, state_key: str, value: object, action_type: str) -> None:
            chip_labels[chip_id] = label
            chip_actions[chip_id] = (state_key, value, action_type)

        for ticker in ticker_filter:
            add_chip(f"ticker::{ticker}", f"Stock: {ticker}  ×", "ticker_filter", ticker, "list")
        for holder in holder_filter:
            add_chip(f"holder::{holder}", f"Holder: {holder}  ×", "holder_filter", holder, "list")
        if selected_period != latest_period:
            add_chip("period", f"{selected_period:%d %b %Y}  ×", "selected_period", latest_period, "scalar")
        if comparison_mode != "Previous period":
            add_chip("comparison", f"Compare: {comparison_mode}  ×", "comparison_mode", "Previous period", "scalar")
        for state_key, label_prefix, selected_values in (
            ("sector_filter", "Sector", sector_filter),
            ("category_filter", "Investor", category_filter),
            ("residency_filter", "Residency", residency_filter),
            ("institution_filter", "Class", institution_filter),
        ):
            for value in selected_values:
                add_chip(f"{state_key}::{value}", f"{label_prefix}: {value}  ×", state_key, value, "list")
        if minimum_holding_pct > 0:
            add_chip("minimum", f"Holding ≥ {minimum_holding_pct:g}%  ×", "minimum_holding_pct", 0.0, "scalar")

        if chip_actions:
            chip_ids = list(chip_actions)
            if set(st.session_state.get("active_filter_chips", [])) != set(chip_ids):
                st.session_state["active_filter_chips"] = chip_ids
            st.pills(
                "Active filters",
                chip_ids,
                selection_mode="multi",
                key="active_filter_chips",
                format_func=lambda chip_id: chip_labels[chip_id],
                on_change=remove_filter_chips,
                args=(chip_actions,),
                label_visibility="collapsed",
                width="stretch",
            )

    filtered_history = apply_global_filters(
        data,
        tickers=ticker_filter,
        holders=holder_filter,
        categories=category_filter,
        residency=residency_filter,
        institution=institution_filter,
        sectors=sector_filter,
        minimum_ownership_pct=minimum_holding_pct,
    )
    current = period_slice(filtered_history, selected_period)
    previous = (
        period_slice(filtered_history, comparison_period)
        if comparison_period is not None
        else filtered_history.iloc[0:0]
    )

    return ToolbarState(
        selected_period=selected_period,
        comparison_mode=comparison_mode,
        comparison_period=comparison_period,
        ticker_filter=ticker_filter,
        holder_filter=holder_filter,
        category_filter=category_filter,
        residency_filter=residency_filter,
        institution_filter=institution_filter,
        sector_filter=sector_filter,
        minimum_holding_pct=minimum_holding_pct,
        display_mode=st.session_state["display_mode"],
        sort_mode=st.session_state["sort_mode"],
        sort_ascending=st.session_state["sort_ascending"],
        filtered_history=filtered_history,
        current=current,
        previous=previous,
    )
