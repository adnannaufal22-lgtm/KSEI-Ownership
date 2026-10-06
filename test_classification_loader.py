import json
from pathlib import Path

import pandas as pd

from data_loader import load_classification_folder, standardize_classification_label


ROOT = Path(__file__).resolve().parent
CLASSIFICATION_DIR = ROOT / "BEI_Data" / "Classification"


def _config() -> dict:
    return json.loads((ROOT / "schema_mapping.json").read_text(encoding="utf-8"))


def test_classification_labels_remove_residency_prefix_and_standardize_case() -> None:
    assert standardize_classification_label("Local_Bank") == "BANK"
    assert standardize_classification_label("Foreign_Bank") == "BANK"
    assert standardize_classification_label("Bank") == "BANK"
    assert standardize_classification_label("Local_New Investor Category") == (
        "NEW INVESTOR CATEGORY"
    )


def test_september_residency_columns_merge_into_historical_classifications() -> None:
    data, metadata = load_classification_folder(CLASSIFICATION_DIR, _config())
    august = set(data.loc[data["date"].eq(pd.Timestamp("2026-08-31")), "classification"].astype(str))
    september = set(
        data.loc[data["date"].eq(pd.Timestamp("2026-09-30")), "classification"].astype(str)
    )

    assert len(august) == 39
    assert september == august
    assert metadata["merged_classification_groups"] == 39
    assert metadata["standardized_classifications"] == 39
    assert metadata["reconciliation_mismatches"] == 0


def test_september_local_and_foreign_values_are_summed() -> None:
    source = pd.read_excel(
        CLASSIFICATION_DIR / "2026-09_Classification.xlsx",
        header=3,
    )
    source_row = source.loc[source["SHARE_CODE"].eq("AADI")].iloc[0]
    expected = float(source_row["Local_Bank"]) + float(source_row["Foreign_Bank"])

    data, _ = load_classification_folder(CLASSIFICATION_DIR, _config())
    actual = data.loc[
        data["date"].eq(pd.Timestamp("2026-09-30"))
        & data["ticker"].astype(str).eq("AADI")
        & data["classification"].astype(str).eq("BANK"),
        "ownership_units",
    ].iloc[0]

    assert actual == expected
