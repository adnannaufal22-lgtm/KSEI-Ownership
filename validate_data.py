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


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "BEI_Data"
OWNERSHIP_DIR = DATA_ROOT / "1% Ownership"
CLASSIFICATION_DIR = DATA_ROOT / "Classification"
TYPE_DIR = DATA_ROOT / "Type"
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
        "all stock-month totals reconcile."
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


if __name__ == "__main__":
    main()
