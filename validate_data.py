from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_loader import (
    load_classification_folder,
    load_excel_folder,
    load_type_folder,
)
from data_processing import standardize_dataframe
from daily_ownership import load_daily_ownership_folder
from monthly_changes import (
    build_monthly_change_detail,
    owner_change_summary,
    stock_change_summary,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "BEI_Data"
OWNERSHIP_DIR = DATA_ROOT / "1% Ownership"
CLASSIFICATION_DIR = DATA_ROOT / "Classification"
TYPE_DIR = DATA_ROOT / "Type"
DAILY_OWNERSHIP_DIR = DATA_ROOT / "5% Ownership"
CONFIG_PATH = PROJECT_DIR / "schema_mapping.json"


def require_clean_load(dataset_name: str, metadata: dict) -> None:
    skipped = [
        issue for issue in metadata.get("issues", [])
        if str(issue).startswith("Skipped ")
    ]
    if skipped:
        details = "\n  - ".join(skipped)
        raise ValueError(f"{dataset_name} contains unusable monthly files:\n  - {details}")
    if metadata.get("source_files", 0) < 1:
        raise ValueError(f"No valid {dataset_name} workbooks were found.")


def periods_label(data: pd.DataFrame) -> str:
    periods = sorted(pd.Timestamp(value) for value in data["date"].dropna().unique())
    if not periods:
        return "no valid reporting periods"
    return f"{periods[0]:%b %Y} to {periods[-1]:%b %Y} ({len(periods)} months)"


def validate_daily_ownership() -> None:
    owners, accounts, movements, account_movements, quality, metadata = (
        load_daily_ownership_folder(DAILY_OWNERSHIP_DIR)
    )
    if metadata.get("source_files", 0) < 1:
        raise ValueError("No daily >5% Ownership workbooks were found.")
    if metadata.get("failed_files", 0):
        raise ValueError(
            "Daily >5% Ownership contains "
            f"{metadata['failed_files']:,} workbook(s) that could not be parsed."
        )
    if owners.empty:
        raise ValueError("No daily >5% beneficial-owner records were loaded.")

    error_count = 0
    if not quality.empty and "severity" in quality:
        error_count = int(quality["severity"].eq("ERROR").sum())
    if error_count:
        raise ValueError(
            f"Daily >5% Ownership contains {error_count:,} data-quality error(s)."
        )

    duplicate_checks = {
        "beneficial-owner positions": owners.duplicated(
            ["date", "ticker", "owner_normalized"]
        ).sum(),
        "account positions": accounts.duplicated(
            [
                "date",
                "ticker",
                "owner_normalized",
                "group_no",
                "account_holder_normalized",
                "account_name_normalized",
                "account_occurrence",
            ]
        ).sum(),
        "beneficial-owner movements": movements.duplicated(
            ["date", "ticker", "owner_normalized"]
        ).sum(),
        "account movements": account_movements.duplicated(
            [
                "date",
                "ticker",
                "owner_normalized",
                "group_no",
                "account_holder_normalized",
                "account_name_normalized",
                "account_occurrence",
            ]
        ).sum(),
    }
    duplicates = {name: int(count) for name, count in duplicate_checks.items() if count}
    if duplicates:
        details = ", ".join(f"{name}: {count:,}" for name, count in duplicates.items())
        raise ValueError(f"Daily >5% Ownership contains duplicate output keys ({details}).")

    earliest = metadata.get("earliest_date")
    latest = metadata.get("latest_date")
    period = (
        f"{pd.Timestamp(earliest):%d %b %Y} to {pd.Timestamp(latest):%d %b %Y}"
        if earliest is not None and latest is not None
        else "no valid reporting dates"
    )
    print(
        f"Daily >5% Ownership: {metadata['source_files']:,} files, "
        f"{metadata['dates']:,} dates, {period}; "
        f"{len(owners):,} owner positions and {len(accounts):,} account positions."
    )
    print(
        f"  Parsed files: {metadata['parsed_files']:,}; failed files: 0; "
        f"non-fatal deduplication warnings: {metadata['quality_warnings']:,}."
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    raw, ownership_meta = load_excel_folder(OWNERSHIP_DIR, config)
    ownership, _ = standardize_dataframe(raw, config)
    require_clean_load("1% Ownership", ownership_meta)
    if ownership.empty:
        raise ValueError("No individual ownership records were loaded.")
    aliases = ownership.attrs.get("holder_alias_audit", {})
    print(
        f"1% Ownership: {ownership_meta['source_files']} files, "
        f"{len(ownership):,} records, {periods_label(ownership)}."
    )
    print(
        f"  Securities: {ownership['ticker'].nunique():,}; "
        f"canonical holders: {ownership['investor_name'].nunique():,}; "
        f"aliases merged: {aliases.get('aliases_merged', 0):,}."
    )
    ownership_periods = sorted(
        pd.Timestamp(value) for value in ownership["date"].dropna().unique()
    )
    if len(ownership_periods) > 1:
        change_detail = build_monthly_change_detail(ownership)
        if change_detail.duplicated(["date", "stock", "owner"]).any():
            raise ValueError("Monthly Changes contains duplicate stock-owner comparisons.")
        if change_detail["previous_date"].ge(change_detail["date"]).any():
            raise ValueError("Monthly Changes comparison periods are not chronological.")
        stock_changes = stock_change_summary(change_detail)
        owner_changes = owner_change_summary(change_detail)
        latest_period = ownership_periods[-1]
        latest_detail = change_detail[
            change_detail["date"].eq(latest_period) & change_detail["is_changed"]
        ]
        print(
            f"Monthly Changes: {latest_period:%b %Y} versus "
            f"{ownership_periods[-2]:%b %Y}; "
            f"{latest_detail['stock'].nunique():,} stocks and "
            f"{latest_detail['owner'].nunique():,} owners with reported changes."
        )
        del change_detail, stock_changes, owner_changes, latest_detail
    del raw

    classification, classification_meta = load_classification_folder(
        CLASSIFICATION_DIR,
        config,
    )
    require_clean_load("Classification", classification_meta)
    if classification_meta.get("reconciliation_mismatches", 0):
        raise ValueError(
            "Classification totals do not reconcile to Total Scripless for "
            f"{classification_meta['reconciliation_mismatches']:,} stock-month rows."
        )
    print(
        f"Classification: {classification_meta['source_files']} files, "
        f"{len(classification):,} rows, {periods_label(classification)}; "
        "all stock-month totals reconcile; "
        f"{classification_meta.get('merged_classification_groups', 0):,} "
        "residency-split category pairs standardized."
    )

    type_data, type_meta = load_type_folder(TYPE_DIR, config)
    require_clean_load("Type", type_meta)
    if type_meta.get("reconciliation_mismatches", 0):
        raise ValueError(
            "Domestic and Foreign totals do not reconcile to Total Scripless for "
            f"{type_meta['reconciliation_mismatches']:,} stock-month rows."
        )
    if type_meta.get("negative_scrip_rows", 0):
        raise ValueError(
            "Total Scripless exceeds Number of Shares for "
            f"{type_meta['negative_scrip_rows']:,} stock-month rows."
        )
    print(
        f"Type: {type_meta['source_files']} files, {len(type_data):,} rows, "
        f"{periods_label(type_data)}; all stock-month totals reconcile and "
        "all scrip values are non-negative."
    )

    if "AADI" in set(type_data["ticker"].astype(str)):
        aadi = type_data[type_data["ticker"].eq("AADI")]
        aadi_summary = aadi.drop_duplicates(["date", "ticker"])
        formula_delta = (
            aadi_summary["scrip_shares"]
            - (aadi_summary["number_of_shares"] - aadi_summary["total_scripless"])
        ).abs()
        if formula_delta.gt(0.5).any():
            raise ValueError("AADI scrip history does not follow the required formula.")
        print(
            f"AADI check: {len(aadi_summary)} Type periods; "
            "Scrip Shares = Number of Shares - Total Scripless in every period."
        )

    validate_daily_ownership()


if __name__ == "__main__":
    main()
