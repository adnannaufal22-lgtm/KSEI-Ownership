from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DOMESTIC = "#0F766E"
FOREIGN = "#2563EB"
POSITIVE = "#15803D"
NEGATIVE = "#B91C1C"
NAVY = "#172033"
MUTED = "#64748B"
GRID = "#E7EBF0"

CATEGORY_COLORS = [
    "#334155", "#2563EB", "#0F766E", "#7C3AED", "#C2410C",
    "#0891B2", "#BE123C", "#4D7C0F", "#6B7280", "#A16207",
]


def _empty(message: str, height: int = 300) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": MUTED, "size": 13})
    figure.update_layout(height=height, xaxis={"visible": False}, yaxis={"visible": False})
    return _style(figure, height=height)


def _style(figure: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 6, "r": 8, "t": 38, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": NAVY, "size": 10.5},
        title={"x": 0.01, "xanchor": "left", "y": 0.985, "yanchor": "top", "font": {"size": 13}},
        hoverlabel={"bgcolor": "#172033", "font_color": "white", "bordercolor": "#172033", "font_size": 11},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 9.5},
            "title": {"text": ""},
            "itemsizing": "constant",
        },
        showlegend=legend,
        bargap=0.24,
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, tickfont={"color": MUTED, "size": 9.5}, automargin=True, title_font={"size": 10})
    figure.update_yaxes(gridcolor=GRID, zeroline=False, tickfont={"color": MUTED, "size": 9.5}, automargin=True, title_font={"size": 10})
    return figure


def trend_chart(data: pd.DataFrame, mode: str) -> go.Figure:
    if data.empty:
        return _empty("No history is available for the active filters.")

    if mode == "Reported holdings":
        grouped = data.groupby("date", as_index=False)["ownership_units"].sum()
        grouped["period_change"] = grouped["ownership_units"].diff()
        figure = px.line(grouped, x="date", y="ownership_units", markers=True)
        figure.update_traces(
            line={"color": "#334155", "width": 2.4}, marker={"size": 6},
            customdata=grouped[["period_change"]],
            hovertemplate="%{x|%d %b %Y}<br>Reported shares: %{y:,.0f}<br>Period change: %{customdata[0]:+,.0f}<extra></extra>",
        )
        figure.update_yaxes(title="Reported shares", tickformat=",.0f")
    else:
        dimension = {
            "Domestic vs Foreign": "domestic_foreign",
            "Institutional vs Individual": "institutional_individual",
            "Investor categories": "investor_category",
        }[mode]
        grouped = data.groupby(["date", dimension], as_index=False)["ownership_pct"].sum()
        if mode == "Investor categories":
            leaders = grouped.groupby(dimension)["ownership_pct"].sum().nlargest(8).index
            grouped[dimension] = np.where(grouped[dimension].isin(leaders), grouped[dimension], "Other")
            grouped = grouped.groupby(["date", dimension], as_index=False)["ownership_pct"].sum()
        totals = grouped.groupby("date")["ownership_pct"].transform("sum")
        grouped["share"] = np.where(totals.ne(0), grouped["ownership_pct"] / totals * 100, 0)
        color_map = {"Domestic": DOMESTIC, "Foreign": FOREIGN, "Institutional": "#334155", "Individual": "#C2410C", "Unclassified": "#94A3B8"}
        figure = px.area(
            grouped, x="date", y="share", color=dimension,
            color_discrete_map=color_map,
            color_discrete_sequence=CATEGORY_COLORS,
            custom_data=["ownership_pct"],
        )
        figure.update_traces(
            hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:.1f}% of reported stake<br>Stake points: %{customdata[0]:,.2f}<extra></extra>"
        )
        figure.update_yaxes(title="Share of reported stake", ticksuffix="%", range=[0, 100])
    figure.update_layout(title={"text": mode, "font": {"size": 14}})
    figure.update_xaxes(title=None, tickformat="%b %Y")
    styled = _style(figure, height=330, legend=mode != "Reported holdings")
    if mode == "Investor categories":
        styled.update_layout(
            legend={
                "orientation": "v",
                "x": 1,
                "xanchor": "right",
                "y": 0.98,
                "yanchor": "top",
                "font": {"size": 8.5},
                "bgcolor": "rgba(255,255,255,0.86)",
                "bordercolor": GRID,
                "borderwidth": 1,
            }
        )
    return styled


def composition_bar(composition: pd.DataFrame, limit: int = 12) -> go.Figure:
    if composition.empty:
        return _empty("No investor composition is available.")
    view = composition.head(limit).sort_values("share_of_reported_pct")
    figure = px.bar(
        view,
        x="share_of_reported_pct",
        y="investor_category",
        orientation="h",
        color="investor_category",
        color_discrete_sequence=CATEGORY_COLORS,
        custom_data=["ownership_units", "stake_points", "investor_records"],
    )
    figure.update_traces(
        texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Share of reported stake: %{x:.2f}%<br>Reported shares: %{customdata[0]:,.0f}<br>Stake points: %{customdata[1]:,.2f}<br>Distinct holders: %{customdata[2]:,.0f}<extra></extra>",
    )
    figure.update_layout(title={"text": "Investor composition", "font": {"size": 14}}, showlegend=False)
    max_share = float(view["share_of_reported_pct"].max()) if len(view) else 100
    figure.update_xaxes(title="Share of reported stake", ticksuffix="%", range=[0, max(max_share * 1.16, 10)])
    figure.update_yaxes(title=None)
    return _style(figure, height=330, legend=False)


def residency_donut(composition: pd.DataFrame) -> go.Figure:
    view = composition[composition["domestic_foreign"].isin(["Domestic", "Foreign"])]
    if view.empty:
        return _empty("Domestic / foreign classification is unavailable.", 275)
    figure = px.pie(
        view,
        values="stake_points",
        names="domestic_foreign",
        hole=0.64,
        color="domestic_foreign",
        color_discrete_map={"Domestic": DOMESTIC, "Foreign": FOREIGN},
    )
    figure.update_traces(
        texttemplate="%{label}<br>%{percent:.1%}", textposition="outside",
        hovertemplate="%{label}<br>Stake points: %{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>",
        marker={"line": {"color": "white", "width": 2}},
    )
    figure.update_layout(title={"text": "Current ownership split", "font": {"size": 14}}, showlegend=False)
    return _style(figure, height=280, legend=False)


def foreign_share_history(data: pd.DataFrame) -> go.Figure:
    grouped = data.groupby(["date", "domestic_foreign"], as_index=False)["ownership_pct"].sum()
    pivot = grouped.pivot(index="date", columns="domestic_foreign", values="ownership_pct").fillna(0)
    if "Foreign" not in pivot:
        return _empty("Foreign history is unavailable.", 275)
    pivot["total"] = pivot.sum(axis=1)
    pivot["foreign_share"] = np.where(pivot["total"].ne(0), pivot["Foreign"] / pivot["total"] * 100, 0)
    view = pivot.reset_index()
    figure = px.line(view, x="date", y="foreign_share", markers=True)
    figure.update_traces(
        line={"color": FOREIGN, "width": 2.4}, marker={"size": 6},
        hovertemplate="%{x|%d %b %Y}<br>Foreign share: %{y:.2f}%<extra></extra>",
    )
    figure.update_layout(title={"text": "Foreign share history", "font": {"size": 14}}, showlegend=False)
    figure.update_yaxes(title="Share of reported stake", ticksuffix="%")
    figure.update_xaxes(title=None, tickformat="%b %Y")
    return _style(figure, height=280, legend=False)


def flow_bar(flow: pd.DataFrame, metric_label: str, limit: int = 14) -> go.Figure:
    if flow.empty:
        return _empty("No comparable flow is available.")
    ranked = pd.concat([flow.nlargest(limit // 2, "change"), flow.nsmallest(limit // 2, "change")]).drop_duplicates("investor_category")
    ranked = ranked.sort_values("change")
    colors = np.where(ranked["change"] >= 0, POSITIVE, NEGATIVE)
    figure = go.Figure(go.Bar(
        x=ranked["change"], y=ranked["investor_category"], orientation="h",
        marker_color=colors,
        customdata=ranked[["previous", "current", "pct_change", "share_of_total"]],
        hovertemplate="%{y}<br>Previous: %{customdata[0]:,.2f}<br>Current: %{customdata[1]:,.2f}<br>Change: %{x:+,.2f}<br>% change: %{customdata[2]:+.1f}%<br>Share: %{customdata[3]:.1f}%<extra></extra>",
    ))
    figure.add_vline(x=0, line_color="#94A3B8", line_width=1)
    figure.update_layout(title={"text": "Investor-category change", "font": {"size": 14}}, showlegend=False)
    figure.update_xaxes(title=metric_label, tickformat=",.0f" if "shares" in metric_label.lower() else ",.1f")
    figure.update_yaxes(title=None)
    return _style(figure, height=365, legend=False)


def ticker_change_bar(snapshot: pd.DataFrame, direction: str, rank_by: str, limit: int = 10) -> go.Figure:
    if snapshot.empty:
        return _empty("No security comparison is available.", 295)
    metric = "period_change" if rank_by == "Absolute change" else "period_change_pct"
    available = snapshot.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    view = available.nlargest(limit, metric) if direction == "increase" else available.nsmallest(limit, metric)
    view = view.sort_values(metric)
    figure = go.Figure(go.Bar(
        x=view[metric], y=view["ticker"], orientation="h",
        marker_color=POSITIVE if direction == "increase" else NEGATIVE,
        customdata=view[["security_name", "period_change", "period_change_pct"]],
        hovertemplate="%{y} · %{customdata[0]}<br>Absolute change: %{customdata[1]:+,.0f}<br>% change: %{customdata[2]:+.1f}%<extra></extra>",
    ))
    title = "Biggest ownership increases" if direction == "increase" else "Biggest ownership decreases"
    figure.update_layout(title={"text": title, "font": {"size": 14}}, showlegend=False)
    figure.update_xaxes(title=rank_by, ticksuffix="%" if rank_by == "Percentage change" else None, tickformat=None if rank_by == "Percentage change" else ",.0f")
    figure.update_yaxes(title=None)
    return _style(figure, height=305, legend=False)


def monthly_movement_chart(monthly: pd.DataFrame, title: str, metric: str) -> go.Figure:
    if monthly.empty:
        return _empty("No monthly movement history is available.", 330)

    is_shares = metric == "Reported shares"
    value_column = "ownership_units" if is_shares else "stake_points"
    change_column = "change_units" if is_shares else "stake_change"
    value_label = "Reported shares" if is_shares else "Stake points"
    number_format = ",.0f" if is_shares else ",.2f"
    change = monthly[change_column].fillna(0)

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=monthly["date"],
            y=monthly[value_column],
            name=value_label,
            mode="lines+markers",
            line={"color": "#334155", "width": 2.5},
            marker={"size": 7},
            customdata=monthly[[change_column, "change_pct", "stake_points", "counterparties"]],
            hovertemplate=(
                "%{x|%d %b %Y}<br>" + value_label + ": %{y:" + number_format + "}"
                + "<br>Monthly change: %{customdata[0]:+,.0f}"
                + "<br>Change: %{customdata[1]:+.1f}%"
                + "<br>Stake points: %{customdata[2]:,.2f}"
                + "<br>Counterparties: %{customdata[3]:,.0f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=monthly["date"],
            y=change,
            name="Monthly change",
            marker_color=np.where(change >= 0, POSITIVE, NEGATIVE),
            opacity=0.34,
            hovertemplate="%{x|%d %b %Y}<br>Monthly change: %{y:+," + (".0f" if is_shares else ".2f") + "}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(title={"text": title, "font": {"size": 14}}, barmode="relative")
    figure.update_xaxes(title=None, tickformat="%b %Y")
    figure.update_yaxes(title_text=value_label, tickformat=number_format, secondary_y=False)
    figure.update_yaxes(title_text="Monthly change", tickformat=number_format, showgrid=False, secondary_y=True)
    return _style(figure, height=345, legend=True)


def movement_breakdown_chart(breakdown: pd.DataFrame, counterparty_label: str, metric: str, limit: int = 8) -> go.Figure:
    if breakdown.empty:
        return _empty("No holder or stock breakdown is available.", 330)
    is_shares = metric == "Reported shares"
    value_column = "ownership_units" if is_shares else "stake_points"
    leaders = breakdown.groupby("counterparty")[value_column].sum().nlargest(limit).index
    view = breakdown[breakdown["counterparty"].isin(leaders)].copy()
    figure = px.line(
        view,
        x="date",
        y=value_column,
        color="counterparty",
        markers=True,
        color_discrete_sequence=CATEGORY_COLORS,
        custom_data=["change_units", "change_pct", "stake_points"],
    )
    figure.update_traces(
        hovertemplate=(
            "%{fullData.name}<br>%{x|%d %b %Y}<br>"
            + ("Reported shares: %{y:,.0f}" if is_shares else "Stake points: %{y:,.2f}")
            + "<br>Monthly share change: %{customdata[0]:+,.0f}"
            + "<br>Change: %{customdata[1]:+.1f}%"
            + "<br>Stake points: %{customdata[2]:,.2f}<extra></extra>"
        )
    )
    figure.update_layout(title={"text": f"Top {counterparty_label.lower()} movements", "font": {"size": 14}})
    figure.update_xaxes(title=None, tickformat="%b %Y")
    figure.update_yaxes(title=metric, tickformat=",.0f" if is_shares else ",.2f")
    styled = _style(figure, height=345, legend=True)
    styled.update_layout(
        legend={
            "orientation": "v",
            "x": 1,
            "xanchor": "right",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 8.5},
            "bgcolor": "rgba(255,255,255,0.86)",
            "bordercolor": GRID,
            "borderwidth": 1,
        }
    )
    return styled
