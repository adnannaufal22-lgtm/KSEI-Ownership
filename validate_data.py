from __future__ import annotations

import json
from pathlib import Path

from data_loader import load_excel_folder
from data_processing import build_quality_report, standardize_dataframe


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "BEI_Data"
CONFIG_PATH = PROJECT_DIR / "schema_mapping.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw, metadata = load_excel_folder(DATA_DIR, config)
    data, mapping = standardize_dataframe(raw, config)
    quality = build_quality_report(raw, data, mapping, metadata)

    periods = sorted(data["date"].dropna().unique())
    if not periods:
        raise ValueError("No valid reporting dates were found.")
    if data.empty:
        raise ValueError("No ownership records were loaded.")

    print(f"Validated {metadata['source_files']} BEI files and {len(data):,} ownership records.")
    print(f"Coverage: {periods[0]:%b %Y} to {periods[-1]:%b %Y} ({len(periods)} months).")
    print(f"Securities: {quality['securities']:,}. Canonical holders: {quality['investors']:,}.")
    print(
        f"Holder aliases merged: {quality['holder_aliases_merged']:,} "
        f"across {quality['holder_alias_groups']:,} groups."
    )


if __name__ == "__main__":
    main()

