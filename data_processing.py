from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
import re
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "date",
    "ticker",
    "security_name",
    "investor_name",
    "investor_category",
    "domestic_foreign",
    "institutional_individual",
    "nationality",
    "domicile",
    "ownership_scripless",
    "ownership_scrip",
    "ownership_units",
    "ownership_pct",
    "investor_count",
    "sector",
]


def normalize_header(value: object) -> str:
    return "_".join(str(value).strip().upper().replace("/", " ").replace("-", " ").split())


def infer_column_mapping(columns: Iterable[object], config: dict) -> dict[str, str]:
    available = {normalize_header(column): str(column) for column in columns}
    mapping: dict[str, str] = {}
    for standard_name, aliases in config.get("columns", {}).items():
        for alias in aliases:
            source = available.get(normalize_header(alias))
            if source:
                mapping[standard_name] = source
                break
    return mapping


def infer_column_sources(columns: Iterable[object], config: dict) -> dict[str, list[str]]:
    """Return every matching source column so monthly schema changes can be coalesced."""
    available = {normalize_header(column): str(column) for column in columns}
    sources: dict[str, list[str]] = {}
    for standard_name, aliases in config.get("columns", {}).items():
        matches = []
        for alias in aliases:
            source = available.get(normalize_header(alias))
            if source and source not in matches:
                matches.append(source)
        if matches:
            sources[standard_name] = matches
    return sources


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


LEGAL_ENTITY_MARKERS = {"PT", "CV", "UD", "PD"}


def normalize_holder_identity(value: object) -> str:
    """Normalize harmless legal-prefix, punctuation, spacing, and case differences."""
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    tokens = text.split()
    while tokens and tokens[0] in LEGAL_ENTITY_MARKERS:
        tokens.pop(0)
    while tokens and tokens[-1] in LEGAL_ENTITY_MARKERS:
        tokens.pop()
    return " ".join(tokens) or text


def build_holder_alias_map(data: pd.DataFrame, similarity_threshold: float = 0.96) -> tuple[dict[str, str], dict]:
    """Resolve high-confidence holder aliases using names plus exact share continuity."""
    names = sorted(data["investor_name"].dropna().astype(str).unique())
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    names_by_identity: defaultdict[str, list[str]] = defaultdict(list)
    for name in names:
        names_by_identity[normalize_holder_identity(name)].append(name)
    for identity_names in names_by_identity.values():
        for alias in identity_names[1:]:
            union(identity_names[0], alias)

    periods = sorted(pd.Timestamp(value) for value in data["date"].dropna().unique())
    monthly = data.groupby(["ticker", "date", "investor_name"], as_index=False)["ownership_units"].sum()
    continuity_edges: set[tuple[str, str]] = set()
    continuity_examples: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for ticker, ticker_data in monthly.groupby("ticker"):
        by_period = {pd.Timestamp(period): frame for period, frame in ticker_data.groupby("date")}
        for previous_period, current_period in zip(periods, periods[1:]):
            previous = by_period.get(previous_period)
            current = by_period.get(current_period)
            if previous is None or current is None:
                continue
            previous_names = set(previous["investor_name"])
            current_names = set(current["investor_name"])
            removed = previous[~previous["investor_name"].isin(current_names)]
            added = current[~current["investor_name"].isin(previous_names)]
            removed_by_units: defaultdict[float, list[str]] = defaultdict(list)
            added_by_units: defaultdict[float, list[str]] = defaultdict(list)
            for row in removed.itertuples():
                removed_by_units[float(row.ownership_units)].append(str(row.investor_name))
            for row in added.itertuples():
                added_by_units[float(row.ownership_units)].append(str(row.investor_name))
            for units in set(removed_by_units) & set(added_by_units):
                if len(removed_by_units[units]) != 1 or len(added_by_units[units]) != 1:
                    continue
                before = removed_by_units[units][0]
                after = added_by_units[units][0]
                before_identity = normalize_holder_identity(before)
                after_identity = normalize_holder_identity(after)
                if before_identity == after_identity:
                    continue
                similarity = SequenceMatcher(None, before_identity, after_identity).ratio()
                if similarity < similarity_threshold:
                    continue
                edge = tuple(sorted((before, after)))
                continuity_edges.add(edge)
                if len(continuity_examples[edge]) < 3:
                    continuity_examples[edge].append(
                        {
                            "ticker": str(ticker),
                            "from": previous_period.strftime("%Y-%m-%d"),
                            "to": current_period.strftime("%Y-%m-%d"),
                            "shares": int(units),
                            "similarity": round(similarity, 3),
                        }
                    )
                union(before, after)

    grouped_names: defaultdict[str, list[str]] = defaultdict(list)
    for name in names:
        grouped_names[find(name)].append(name)

    name_stats = (
        data.groupby("investor_name")
        .agg(last_seen=("date", "max"), observations=("date", "size"), stocks=("ticker", "nunique"))
        .to_dict("index")
    )

    def representative_score(name: str) -> tuple:
        stats = name_stats[name]
        cleaned = re.sub(r"\s+", " ", name).strip()
        leading_legal_name = bool(re.match(r"^(PT|CV|UD|PD)(?:\.|\s)", cleaned, flags=re.IGNORECASE))
        return (
            pd.Timestamp(stats["last_seen"]),
            int(stats["observations"]),
            int(stats["stocks"]),
            leading_legal_name,
            cleaned == cleaned.upper(),
            len(cleaned),
        )

    alias_map: dict[str, str] = {}
    alias_groups = []
    for group_names in grouped_names.values():
        canonical = max(group_names, key=representative_score)
        for name in group_names:
            alias_map[name] = canonical
        if len(group_names) > 1:
            group_edges = [edge for edge in continuity_edges if edge[0] in group_names and edge[1] in group_names]
            alias_groups.append(
                {
                    "canonical": canonical,
                    "aliases": sorted(group_names),
                    "method": "name + exact share continuity" if group_edges else "legal-name formatting",
                    "continuity_examples": [example for edge in group_edges for example in continuity_examples[edge]],
                }
            )

    audit = {
        "source_names": len(names),
        "canonical_names": len(set(alias_map.values())),
        "aliases_merged": len(names) - len(set(alias_map.values())),
        "alias_groups": sorted(alias_groups, key=lambda item: (-len(item["aliases"]), item["canonical"])),
        "continuity_pairs": len(continuity_edges),
    }
    return alias_map, audit


def standardize_dataframe(raw: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    source_columns = infer_column_sources(raw.columns, config)
    mapping = {standard_name: " / ".join(sources) for standard_name, sources in source_columns.items()}
    required = {"date", "ticker", "investor_name", "ownership_units"}
    missing_required = sorted(required - set(mapping))
    if missing_required:
        raise ValueError("Required fields could not be mapped: " + ", ".join(missing_required))

    standardized = pd.DataFrame(index=raw.index)
    for standard_name in STANDARD_COLUMNS:
        sources = source_columns.get(standard_name, [])
        if not sources:
            standardized[standard_name] = pd.NA
            continue
        values = raw[sources[0]]
        for source in sources[1:]:
            values = values.combine_first(raw[source])
        standardized[standard_name] = values

    text_columns = [
        "ticker",
        "security_name",
        "investor_name",
        "investor_category",
        "domestic_foreign",
        "nationality",
        "domicile",
        "sector",
    ]
    for column in text_columns:
        standardized[column] = _clean_text(standardized[column])

    standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce").dt.normalize()
    for column in ["ownership_scripless", "ownership_scrip", "ownership_units", "ownership_pct", "investor_count"]:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    type_map = {str(key).upper(): value for key, value in config.get("investor_type_map", {}).items()}
    original_category = standardized["investor_category"]
    standardized["investor_category"] = original_category.map(
        lambda value: type_map.get(str(value).upper(), value) if pd.notna(value) else "Unclassified"
    )
    residency_map = {str(key).upper(): value for key, value in config.get("residency_map", {}).items()}
    standardized["domestic_foreign"] = standardized["domestic_foreign"].map(
        lambda value: residency_map.get(str(value).upper(), value) if pd.notna(value) else "Unclassified"
    )
    standardized["institutional_individual"] = np.where(
        standardized["investor_category"].eq("Individual"),
        "Individual",
        np.where(standardized["investor_category"].eq("Unclassified"), "Unclassified", "Institutional"),
    )

    pct_median = standardized["ownership_pct"].dropna().median()
    if pd.notna(pct_median) and pct_median <= 1:
        standardized["ownership_pct"] = standardized["ownership_pct"] * 100

    if "SOURCE_FILE" in raw.columns:
        standardized["source_file"] = raw["SOURCE_FILE"].astype("string")
    else:
        standardized["source_file"] = pd.NA
    if "SOURCE_EXCEL_ROW" in raw.columns:
        standardized["source_row"] = pd.to_numeric(raw["SOURCE_EXCEL_ROW"], errors="coerce").astype("Int64")
    else:
        standardized["source_row"] = standardized.index + 2
    standardized = standardized.dropna(subset=["date", "ticker", "investor_name", "ownership_units"])
    standardized["investor_name_raw"] = standardized["investor_name"].astype("string")
    holder_alias_map, holder_alias_audit = build_holder_alias_map(standardized)
    standardized["investor_name"] = standardized["investor_name_raw"].map(holder_alias_map).astype("string")
    standardized["investor_identity_key"] = standardized["investor_name"].map(normalize_holder_identity).astype("string")
    subset = [
        "date",
        "ticker",
        "investor_name",
        "investor_category",
        "domestic_foreign",
        "ownership_units",
        "ownership_pct",
    ]
    standardized = standardized.drop_duplicates(subset=subset, keep="first")
    standardized = standardized.sort_values(["date", "ticker", "ownership_units"], ascending=[True, True, False])
    standardized = standardized.reset_index(drop=True)
    standardized.attrs["holder_alias_audit"] = holder_alias_audit
    return standardized, mapping


def build_quality_report(raw: pd.DataFrame, data: pd.DataFrame, mapping: dict[str, str], metadata: dict) -> dict:
    important = ["date", "ticker", "investor_name", "ownership_units", "ownership_pct"]
    missing = {column: int(data[column].isna().sum()) for column in important}
    duplicate_count = metadata["source_rows"] - len(data)
    component_mismatches = 0
    if data["ownership_scripless"].notna().any() and data["ownership_scrip"].notna().any():
        component_delta = (
            data["ownership_units"]
            - data["ownership_scripless"].fillna(0)
            - data["ownership_scrip"].fillna(0)
        )
        component_mismatches = int(component_delta.abs().gt(0.5).sum())
    holder_alias_audit = data.attrs.get("holder_alias_audit", {})
    holder_alias_examples = sorted(
        holder_alias_audit.get("alias_groups", []),
        key=lambda item: (
            item.get("canonical") != "PT TIRTA ORISA YASA",
            item.get("method") != "name + exact share continuity",
            -len(item.get("aliases", [])),
            item.get("canonical", ""),
        ),
    )[:20]
    return {
        **metadata,
        "earliest_date": data["date"].min(),
        "latest_date": data["date"].max(),
        "observations": int(len(data)),
        "securities": int(data["ticker"].nunique()),
        "investors": int(data["investor_name"].nunique()),
        "periods": int(data["date"].nunique()),
        "duplicates_removed": int(max(duplicate_count, 0)),
        "missing_important": missing,
        "component_mismatches": component_mismatches,
        "minimum_observed_stake": float(data["ownership_pct"].min()) if data["ownership_pct"].notna().any() else np.nan,
        "holder_aliases_merged": int(holder_alias_audit.get("aliases_merged", 0)),
        "holder_alias_groups": int(len(holder_alias_audit.get("alias_groups", []))),
        "holder_continuity_pairs": int(holder_alias_audit.get("continuity_pairs", 0)),
        "holder_alias_examples": holder_alias_examples,
        "mapping": mapping,
    }


def apply_dimension_filters(
    data: pd.DataFrame,
    tickers: list[str] | None = None,
    holders: list[str] | None = None,
    categories: list[str] | None = None,
    residency: list[str] | None = None,
    institution: list[str] | None = None,
    sectors: list[str] | None = None,
    minimum_ownership_pct: float | None = None,
) -> pd.DataFrame:
    filtered = data
    for column, selected in (
        ("ticker", tickers),
        ("investor_name", holders),
        ("investor_category", categories),
        ("domestic_foreign", residency),
        ("institutional_individual", institution),
        ("sector", sectors),
    ):
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    if minimum_ownership_pct is not None:
        filtered = filtered[filtered["ownership_pct"].ge(float(minimum_ownership_pct))]
    return filtered


def apply_global_filters(
    data: pd.DataFrame,
    tickers: list[str] | None = None,
    holders: list[str] | None = None,
    categories: list[str] | None = None,
    residency: list[str] | None = None,
    institution: list[str] | None = None,
    sectors: list[str] | None = None,
    minimum_ownership_pct: float | None = None,
) -> pd.DataFrame:
    """Apply the dashboard's single, shared set of global filters."""
    return apply_dimension_filters(
        data,
        tickers=tickers,
        holders=holders,
        categories=categories,
        residency=residency,
        institution=institution,
        sectors=sectors,
        minimum_ownership_pct=minimum_ownership_pct,
    )


def period_slice(data: pd.DataFrame, period: pd.Timestamp) -> pd.DataFrame:
    return data[data["date"].eq(pd.Timestamp(period))]


def _target_date(current: pd.Timestamp, comparison: str) -> pd.Timestamp | None:
    if comparison == "Previous month":
        year, month = current.year, current.month - 1
        if month == 0:
            year, month = year - 1, 12
        return pd.Timestamp(year, month, min(current.day, monthrange(year, month)[1]))
    if comparison == "Previous quarter":
        return current - pd.DateOffset(months=3)
    if comparison == "Previous year":
        return current - pd.DateOffset(years=1)
    return None


def resolve_comparison_period(
    periods: Iterable[pd.Timestamp],
    current: pd.Timestamp,
    comparison: str,
    custom: date | pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    available = sorted(pd.Timestamp(value) for value in periods if pd.Timestamp(value) < pd.Timestamp(current))
    if not available:
        return None
    if comparison == "Previous period":
        return available[-1]
    target = pd.Timestamp(custom) if comparison == "Custom period" and custom else _target_date(pd.Timestamp(current), comparison)
    if target is None:
        return available[-1]
    return min(available, key=lambda value: abs(value - target))


def composition_share(data: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = data.groupby(column, dropna=False).agg(
        ownership_units=("ownership_units", "sum"),
        stake_points=("ownership_pct", "sum"),
        investor_records=("investor_name", "nunique"),
    ).reset_index()
    total_points = grouped["stake_points"].sum()
    grouped["share_of_reported_pct"] = grouped["stake_points"] / total_points * 100 if total_points else 0
    return grouped.sort_values("share_of_reported_pct", ascending=False)


def snapshot_metrics(data: pd.DataFrame) -> dict[str, float]:
    by_residency = data.groupby("domestic_foreign")["ownership_pct"].sum()
    by_institution = data.groupby("institutional_individual")["ownership_pct"].sum()
    classified_residency = by_residency.get("Domestic", 0) + by_residency.get("Foreign", 0)
    classified_institution = by_institution.get("Institutional", 0) + by_institution.get("Individual", 0)
    return {
        "reported_units": float(data["ownership_units"].sum()),
        "foreign_share": float(by_residency.get("Foreign", 0) / classified_residency * 100) if classified_residency else np.nan,
        "domestic_share": float(by_residency.get("Domestic", 0) / classified_residency * 100) if classified_residency else np.nan,
        "institutional_share": float(by_institution.get("Institutional", 0) / classified_institution * 100) if classified_institution else np.nan,
        "individual_share": float(by_institution.get("Individual", 0) / classified_institution * 100) if classified_institution else np.nan,
        "investors": float(data["investor_name"].nunique()),
        "securities": float(data["ticker"].nunique()),
        "mean_coverage": float(data.groupby("ticker")["ownership_pct"].sum().mean()) if len(data) else np.nan,
    }


def category_flow(current: pd.DataFrame, previous: pd.DataFrame, metric: str = "ownership_units") -> pd.DataFrame:
    current_group = current.groupby("investor_category").agg(
        current_units=("ownership_units", "sum"), current_points=("ownership_pct", "sum")
    )
    previous_group = previous.groupby("investor_category").agg(
        previous_units=("ownership_units", "sum"), previous_points=("ownership_pct", "sum")
    )
    result = current_group.join(previous_group, how="outer").fillna(0).reset_index()
    if metric == "ownership_pct":
        result["current"] = result["current_points"]
        result["previous"] = result["previous_points"]
    else:
        result["current"] = result["current_units"]
        result["previous"] = result["previous_units"]
    result["change"] = result["current"] - result["previous"]
    result["pct_change"] = np.where(result["previous"].ne(0), result["change"] / result["previous"] * 100, np.nan)
    current_total = result["current"].sum()
    result["share_of_total"] = result["current"] / current_total * 100 if current_total else 0
    return result.sort_values("change", ascending=False)


def ticker_snapshot(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    def aggregate(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        grouped = frame.groupby(["ticker", "security_name"], dropna=False).agg(
            total_units=("ownership_units", "sum"),
            stake_points=("ownership_pct", "sum"),
            investors=("investor_name", "nunique"),
        )
        residency = frame.pivot_table(index=["ticker", "security_name"], columns="domestic_foreign", values="ownership_pct", aggfunc="sum", fill_value=0)
        institution = frame.pivot_table(index=["ticker", "security_name"], columns="institutional_individual", values="ownership_pct", aggfunc="sum", fill_value=0)
        grouped["foreign_points"] = residency.get("Foreign", 0)
        grouped["domestic_points"] = residency.get("Domestic", 0)
        grouped["institutional_points"] = institution.get("Institutional", 0)
        grouped["individual_points"] = institution.get("Individual", 0)
        grouped["foreign_share_pct"] = np.where(grouped["stake_points"].ne(0), grouped["foreign_points"] / grouped["stake_points"] * 100, np.nan)
        return grouped.add_prefix(prefix)

    current_agg = aggregate(current, "current_")
    previous_agg = aggregate(previous, "previous_") if len(previous) else pd.DataFrame(index=current_agg.index)
    result = current_agg.join(previous_agg[[column for column in previous_agg.columns if column == "previous_total_units"]], how="outer").fillna(0)
    result["period_change"] = result["current_total_units"] - result.get("previous_total_units", 0)
    result["period_change_pct"] = np.where(
        result.get("previous_total_units", 0) != 0,
        result["period_change"] / result.get("previous_total_units", 0) * 100,
        np.nan,
    )
    return result.reset_index().sort_values("current_total_units", ascending=False)


def sort_ticker_snapshot(snapshot: pd.DataFrame, sort_by: str, ascending: bool = True) -> pd.DataFrame:
    """Sort the security monitor using the toolbar's shared sort state."""
    if sort_by == "Ticker":
        return snapshot.sort_values(["ticker", "security_name"], ascending=[ascending, ascending]).reset_index(drop=True)
    if sort_by == "Holding":
        return snapshot.sort_values("current_total_units", ascending=False).reset_index(drop=True)
    if sort_by == "Movement":
        return (
            snapshot.assign(_movement_size=snapshot["period_change"].abs())
            .sort_values(["_movement_size", "ticker"], ascending=[False, True])
            .drop(columns="_movement_size")
            .reset_index(drop=True)
        )
    raise ValueError("sort_by must be 'Ticker', 'Holding', or 'Movement'")


def concentration_metrics(data: pd.DataFrame) -> dict[str, float]:
    by_holder = data.groupby("investor_name")["ownership_pct"].sum().sort_values(ascending=False)
    total = by_holder.sum()
    if not total:
        return {"top_1": np.nan, "top_3": np.nan, "top_5": np.nan, "hhi": np.nan}
    normalized = by_holder / total
    return {
        "top_1": float(normalized.head(1).sum() * 100),
        "top_3": float(normalized.head(3).sum() * 100),
        "top_5": float(normalized.head(5).sum() * 100),
        "hhi": float((normalized.pow(2).sum()) * 10000),
    }


def entity_monthly_movement(
    data: pd.DataFrame,
    entity_column: str,
    entity_value: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return monthly totals and counterparty-level movements for a stock or holder."""
    if entity_column not in {"ticker", "investor_name"}:
        raise ValueError("entity_column must be 'ticker' or 'investor_name'")

    scoped = data[data[entity_column].eq(entity_value)].copy()
    counterparty_column = "investor_name" if entity_column == "ticker" else "ticker"
    counterparty_label = "Holder" if entity_column == "ticker" else "Stock"

    monthly = (
        scoped.groupby("date", as_index=False)
        .agg(
            ownership_units=("ownership_units", "sum"),
            stake_points=("ownership_pct", "sum"),
            counterparties=(counterparty_column, "nunique"),
        )
        .sort_values("date")
    )
    monthly["change_units"] = monthly["ownership_units"].diff()
    monthly["change_pct"] = monthly["ownership_units"].pct_change(fill_method=None) * 100
    monthly["stake_change"] = monthly["stake_points"].diff()
    monthly["stake_change_pct"] = monthly["stake_points"].pct_change(fill_method=None) * 100

    breakdown = (
        scoped.groupby(["date", counterparty_column], as_index=False)
        .agg(ownership_units=("ownership_units", "sum"), stake_points=("ownership_pct", "sum"))
        .rename(columns={counterparty_column: "counterparty"})
        .sort_values(["counterparty", "date"])
    )
    if entity_column == "investor_name":
        stock_names = (
            scoped[["ticker", "security_name"]]
            .drop_duplicates("ticker")
            .set_index("ticker")["security_name"]
            .to_dict()
        )
        breakdown["counterparty"] = breakdown["counterparty"].map(
            lambda ticker: f"{ticker} · {stock_names.get(ticker, '')}".rstrip(" ·")
        )
    breakdown["change_units"] = breakdown.groupby("counterparty")["ownership_units"].diff()
    breakdown["change_pct"] = breakdown.groupby("counterparty")["ownership_units"].pct_change(fill_method=None) * 100
    breakdown["stake_change"] = breakdown.groupby("counterparty")["stake_points"].diff()
    breakdown["stake_change_pct"] = breakdown.groupby("counterparty")["stake_points"].pct_change(fill_method=None) * 100
    return monthly.reset_index(drop=True), breakdown.reset_index(drop=True), counterparty_label


def movement_pivot_table(breakdown: pd.DataFrame, metric: str, view: str) -> pd.DataFrame:
    """Pivot monthly stock/holder movements with counterparties on rows and periods on columns."""
    value_columns = {
        ("Reported shares", "Holding level"): "ownership_units",
        ("Reported shares", "Monthly change"): "change_units",
        ("Reported shares", "Monthly % change"): "change_pct",
        ("Stake points", "Holding level"): "stake_points",
        ("Stake points", "Monthly change"): "stake_change",
        ("Stake points", "Monthly % change"): "stake_change_pct",
    }
    value_column = value_columns[(metric, view)]
    all_dates = sorted(pd.Timestamp(value) for value in breakdown["date"].dropna().unique())
    pivot = breakdown.pivot_table(
        index="counterparty",
        columns="date",
        values=value_column,
        aggfunc="sum",
        dropna=False,
    )
    if not all_dates:
        return pd.DataFrame(columns=["counterparty"])

    # Change fields are intentionally blank for an entity's first observed
    # period. Reindexing ensures those reporting-month columns stay visible.
    pivot = pivot.reindex(columns=all_dates)
    latest_column = pivot.columns[-1]
    latest_values = pivot[latest_column].abs() if view != "Holding level" else pivot[latest_column]
    pivot = pivot.assign(_latest_sort=latest_values).sort_values("_latest_sort", ascending=False).drop(columns="_latest_sort")
    pivot.columns = [pd.Timestamp(column).strftime("%b-%y") for column in pivot.columns]
    return pivot.reset_index()


def selected_ownership_history(
    data: pd.DataFrame,
    analyze_by: str,
    entity: str,
) -> tuple[pd.DataFrame, str]:
    """Return owner-level history for one globally selected stock or owner."""
    if analyze_by not in {"Stock", "Owner"}:
        raise ValueError("analyze_by must be 'Stock' or 'Owner'")
    entity_column = "ticker" if analyze_by == "Stock" else "investor_name"
    scoped = data[data[entity_column].eq(entity)].copy()
    if scoped.empty:
        return scoped, "Holder" if analyze_by == "Stock" else "Stock"

    grouped = (
        scoped.groupby(["date", "ticker", "security_name", "investor_name"], as_index=False, dropna=False)
        .agg(
            ownership_units=("ownership_units", "sum"),
            ownership_pct=("ownership_pct", lambda values: values.sum(min_count=1)),
        )
        .sort_values(["date", "ownership_units"], ascending=[True, False])
        .reset_index(drop=True)
    )
    grouped["holder_label"] = grouped["investor_name"].astype(str)
    grouped["stock_label"] = (
        grouped["ticker"].astype(str)
        + " · "
        + grouped["security_name"].fillna("").astype(str)
    ).str.rstrip(" ·")
    if analyze_by == "Stock":
        grouped["series_label"] = grouped["holder_label"]
        row_label = "Holder"
    else:
        grouped["series_label"] = grouped["stock_label"]
        row_label = "Stock"
    return grouped, row_label


def historical_pivot(
    data: pd.DataFrame,
    row_column: str,
    value_column: str,
    all_dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Create a chronological historical pivot without filling missing observations."""
    if data.empty:
        return pd.DataFrame(columns=[row_column])
    dates = (
        sorted(pd.Timestamp(value) for value in all_dates)
        if all_dates is not None
        else sorted(pd.Timestamp(value) for value in data["date"].dropna().unique())
    )
    pivot = data.pivot_table(
        index=row_column,
        columns="date",
        values=value_column,
        aggfunc="sum",
        dropna=False,
        observed=True,
    )
    pivot = pivot.reindex(columns=dates)
    if dates:
        latest = pivot.iloc[:, -1]
        pivot = pivot.assign(_latest_sort=latest).sort_values(
            "_latest_sort",
            ascending=False,
            na_position="last",
        ).drop(columns="_latest_sort")
    pivot.columns = [pd.Timestamp(column).strftime("%b-%y") for column in pivot.columns]
    return pivot.reset_index()


def monthly_percentage_change_pivot(
    level_pivot: pd.DataFrame,
    row_column: str,
) -> pd.DataFrame:
    """Convert a chronological level pivot into month-over-month percentage changes."""
    if level_pivot.empty:
        return pd.DataFrame(columns=[row_column])
    value_columns = [column for column in level_pivot.columns if column != row_column]
    result = level_pivot[[row_column]].copy()
    numeric_levels = level_pivot[value_columns].apply(pd.to_numeric, errors="coerce")
    previous_levels = numeric_levels.shift(1, axis=1)
    percentage_change = (numeric_levels - previous_levels) / previous_levels * 100
    percentage_change = percentage_change.mask(
        previous_levels.eq(0) & numeric_levels.eq(0),
        0.0,
    )
    percentage_change = percentage_change.replace([np.inf, -np.inf], np.nan)
    result[value_columns] = percentage_change
    return result


def classification_stock_history(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    scoped = data[data["ticker"].eq(ticker)].copy()
    if scoped.empty:
        return scoped
    active_classifications = (
        scoped.groupby("classification", observed=True)["ownership_units"]
        .apply(lambda values: values.fillna(0).abs().sum())
        .loc[lambda values: values.gt(0)]
        .index
    )
    return (
        scoped[scoped["classification"].isin(active_classifications)]
        .sort_values(["date", "classification"])
        .reset_index(drop=True)
    )


def type_stock_summary(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    scoped = data[data["ticker"].eq(ticker)]
    if scoped.empty:
        return pd.DataFrame(
            columns=["date", "number_of_shares", "total_scripless", "scrip_shares"]
        )
    return (
        scoped.groupby("date", as_index=False)
        .agg(
            number_of_shares=("number_of_shares", "first"),
            total_scripless=("total_scripless", "first"),
            scrip_shares=("scrip_shares", "first"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def type_residency_history(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    scoped = data[data["ticker"].eq(ticker)]
    if scoped.empty:
        return pd.DataFrame(
            columns=["date", "domestic_foreign", "ownership_units", "ownership_pct"]
        )
    grouped = (
        scoped.groupby(["date", "domestic_foreign"], as_index=False)
        .agg(
            ownership_units=("ownership_units", "sum"),
            number_of_shares=("number_of_shares", "first"),
        )
        .sort_values(["date", "domestic_foreign"])
        .reset_index(drop=True)
    )
    denominator = grouped["number_of_shares"].where(grouped["number_of_shares"].gt(0))
    grouped["ownership_pct"] = grouped["ownership_units"] / denominator * 100
    return grouped
