from __future__ import annotations

import numpy as np
import pandas as pd


DETAIL_COLUMNS = [
    "date",
    "previous_date",
    "stock",
    "security_name",
    "owner",
    "previous_shares",
    "current_shares",
    "change_shares",
    "previous_percentage",
    "current_percentage",
    "change_percentage",
    "share_change_percentage",
    "observation_status",
    "is_changed",
]


def build_monthly_change_detail(data: pd.DataFrame) -> pd.DataFrame:
    """Build consecutive KSEI snapshot differences for stock/owner drill-downs.

    A holder missing from one side of a comparison is treated as zero *reported*
    ownership for the delta and explicitly labelled as entering or leaving the
    report. This does not imply an exchange transaction or a zero legal holding.
    """
    required = {
        "date",
        "ticker",
        "security_name",
        "investor_name",
        "ownership_units",
        "ownership_pct",
    }
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    scoped = data[list(required)].copy()
    scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce").dt.normalize()
    scoped["ownership_units"] = pd.to_numeric(scoped["ownership_units"], errors="coerce")
    scoped["ownership_pct"] = pd.to_numeric(scoped["ownership_pct"], errors="coerce")
    scoped = scoped.dropna(subset=["date", "ticker", "investor_name", "ownership_units"])
    if scoped.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    stock_names = (
        scoped[["ticker", "security_name"]]
        .dropna(subset=["ticker", "security_name"])
        .drop_duplicates("ticker")
        .set_index("ticker")["security_name"]
        .astype(str)
        .to_dict()
    )
    snapshots = (
        scoped.groupby(
            ["date", "ticker", "investor_name"],
            as_index=False,
            observed=True,
        )
        .agg(
            ownership_units=("ownership_units", "sum"),
            ownership_pct=("ownership_pct", "sum"),
        )
        .rename(columns={"ticker": "stock", "investor_name": "owner"})
    )
    snapshots["security_name"] = snapshots["stock"].map(stock_names).astype("string")
    dates = sorted(pd.Timestamp(value) for value in snapshots["date"].dropna().unique())
    comparisons: list[pd.DataFrame] = []

    for previous_date, current_date in zip(dates, dates[1:]):
        previous = snapshots[snapshots["date"].eq(previous_date)].drop(columns="date")
        current = snapshots[snapshots["date"].eq(current_date)].drop(columns="date")
        pair = previous.merge(
            current,
            on=["stock", "owner"],
            how="outer",
            suffixes=("_previous", "_current"),
            indicator=True,
        )
        pair["date"] = current_date
        pair["previous_date"] = previous_date
        pair["security_name"] = pair["security_name_current"].combine_first(
            pair["security_name_previous"]
        )
        pair["previous_shares"] = pair["ownership_units_previous"].fillna(0.0)
        pair["current_shares"] = pair["ownership_units_current"].fillna(0.0)
        pair["change_shares"] = pair["current_shares"] - pair["previous_shares"]
        pair["previous_percentage"] = pair["ownership_pct_previous"].fillna(0.0)
        pair["current_percentage"] = pair["ownership_pct_current"].fillna(0.0)
        pair["change_percentage"] = (
            pair["current_percentage"] - pair["previous_percentage"]
        )
        previous_denominator = pair["previous_shares"].where(pair["previous_shares"].abs().gt(0.5))
        pair["share_change_percentage"] = (
            pair["change_shares"] / previous_denominator.abs() * 100
        ).replace([np.inf, -np.inf], np.nan)
        pair["observation_status"] = pair["_merge"].map(
            {
                "left_only": "No longer reported",
                "right_only": "Newly reported",
                "both": "Continuing",
            }
        ).astype("string")
        pair["is_changed"] = (
            pair["change_shares"].abs().gt(0.5)
            | pair["change_percentage"].abs().gt(1e-9)
        )
        comparisons.append(pair[DETAIL_COLUMNS])

    if not comparisons:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    return (
        pd.concat(comparisons, ignore_index=True)
        .sort_values(["date", "change_shares"], ascending=[True, False])
        .reset_index(drop=True)
    )


def stock_change_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize holder-level differences into stock-level monthly activity."""
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "previous_date",
                "stock",
                "security_name",
                "previous_shares",
                "current_shares",
                "net_change",
                "absolute_change",
                "percent_change",
                "changing_holders",
            ]
        )
    view = detail.copy()
    view["absolute_change"] = view["change_shares"].abs()
    view["changed_owner"] = view["owner"].where(view["is_changed"])
    summary = (
        view.groupby(["date", "previous_date", "stock"], as_index=False, observed=True)
        .agg(
            security_name=("security_name", "first"),
            previous_shares=("previous_shares", "sum"),
            current_shares=("current_shares", "sum"),
            net_change=("change_shares", "sum"),
            absolute_change=("absolute_change", "sum"),
            changing_holders=("changed_owner", "nunique"),
        )
    )
    summary = summary[summary["absolute_change"].gt(0.5)].copy()
    denominator = summary["previous_shares"].where(summary["previous_shares"].abs().gt(0.5))
    summary["percent_change"] = (
        summary["net_change"] / denominator.abs() * 100
    ).replace([np.inf, -np.inf], np.nan)
    return summary.sort_values(
        ["date", "absolute_change", "stock"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def owner_change_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize stock-level differences into owner monthly activity."""
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "previous_date",
                "owner",
                "stocks_increased",
                "stocks_decreased",
                "net_change",
                "absolute_change",
                "stocks_changed",
            ]
        )
    view = detail[detail["is_changed"]].copy()
    view["absolute_change"] = view["change_shares"].abs()
    view["increased_stock"] = view["stock"].where(view["change_shares"].gt(0.5))
    view["decreased_stock"] = view["stock"].where(view["change_shares"].lt(-0.5))
    summary = (
        view.groupby(["date", "previous_date", "owner"], as_index=False, observed=True)
        .agg(
            stocks_increased=("increased_stock", "nunique"),
            stocks_decreased=("decreased_stock", "nunique"),
            net_change=("change_shares", "sum"),
            absolute_change=("absolute_change", "sum"),
            stocks_changed=("stock", "nunique"),
        )
    )
    return summary.sort_values(
        ["date", "absolute_change", "owner"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
