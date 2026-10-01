from pathlib import Path

import pandas as pd

from daily_ownership import (
    SIGNAL_ACCUMULATING,
    SIGNAL_INTERNAL_TRANSFER,
    SIGNAL_NO_LONGER_REPORTED,
    SIGNAL_SELLING,
    SIGNAL_UNCHANGED,
    classify_owner_movement,
    load_daily_ownership_files,
    parse_ksei_share_value,
    resolve_account_share_values,
)


DATA_DIR = Path(__file__).resolve().parent / "BEI_Data" / "5% Ownership"


def test_actual_accumulation_signal() -> None:
    assert classify_owner_movement(100_000_000, 125_000_000, 5.2, 6.1, True) == SIGNAL_ACCUMULATING


def test_actual_selling_signal() -> None:
    assert classify_owner_movement(125_000_000, 100_000_000, 6.1, 5.2, True) == SIGNAL_SELLING


def test_unchanged_signal() -> None:
    assert classify_owner_movement(100_000_000, 100_000_000, 6.1, 6.1, False) == SIGNAL_UNCHANGED


def test_internal_transfer_signal() -> None:
    assert (
        classify_owner_movement(4_598_463_100, 4_598_463_100, 27.73, 27.73, True)
        == SIGNAL_INTERNAL_TRANSFER
    )


def test_missing_next_observation_is_not_selling() -> None:
    assert (
        classify_owner_movement(100_000_000, pd.NA, 6.0, pd.NA, False)
        == SIGNAL_NO_LONGER_REPORTED
    )


def test_excel_number_normalization_and_group_reconciliation() -> None:
    assert parse_ksei_share_value(965) == 965_000
    assert parse_ksei_share_value("1,015,200") == 1_015_200
    assert parse_ksei_share_value(450.4) == 450_400
    assert parse_ksei_share_value(398.1) == 398_100

    raw_values = [450.6, 500, 134.261, "533,177,194", "1,695,360,887", "22,774,600"]
    resolved, residual = resolve_account_share_values(raw_values, 2_251_898_042)
    assert resolved[:3] == [450_600, 500, 134_261]
    assert residual == 0


def test_bmtr_real_internal_transfer_case() -> None:
    source = DATA_DIR / "peng-2026-09-29-00084-lima-persen.xlsx"
    owners, accounts, movements, account_movements, quality, metadata = load_daily_ownership_files([source])
    assert metadata["failed_files"] == 0
    bmtr = movements[
        movements["ticker"].eq("BMTR")
        & movements["owner_normalized"].eq("PT MNC ASIA HOLDING TBK")
    ].iloc[0]
    assert bmtr["previous_shares"] == 4_598_463_100
    assert bmtr["current_shares"] == 4_598_463_100
    assert bmtr["delta_shares"] == 0
    assert bmtr["signal"] == SIGNAL_INTERNAL_TRANSFER

    changed = account_movements[
        account_movements["ticker"].eq("BMTR")
        & account_movements["owner_normalized"].eq("PT MNC ASIA HOLDING TBK")
        & account_movements["delta_shares"].abs().gt(.5)
    ]
    mnc = changed[changed["account_holder"].eq("PT MNC SEKURITAS")].iloc[0]
    kiwoom = changed[
        changed["account_holder"].eq("PT KIWOOM SEKURITAS INDONESIA")
    ].iloc[0]
    assert mnc["previous_shares"] == 116_320_983
    assert mnc["current_shares"] == 34_993_483
    assert mnc["delta_shares"] == -81_327_500
    assert kiwoom["delta_shares"] == 81_327_500
    assert quality.empty
    assert not owners.empty and not accounts.empty


def test_multiple_daily_files_are_deduplicated() -> None:
    files = [
        DATA_DIR / "peng-2026-09-28-00083-lima-persen.xlsx",
        DATA_DIR / "peng-2026-09-29-00084-lima-persen.xlsx",
    ]
    owners, accounts, movements, account_movements, quality, metadata = load_daily_ownership_files(files)
    assert metadata["parsed_files"] == 2
    assert not owners.duplicated(["date", "ticker", "owner_normalized"]).any()
    assert not movements.duplicated(["date", "ticker", "owner_normalized"]).any()
    assert not accounts.duplicated(
        [
            "date",
            "ticker",
            "owner_normalized",
            "group_no",
            "account_holder_normalized",
            "account_name_normalized",
            "account_occurrence",
        ]
    ).any()
    assert not account_movements.duplicated(
        [
            "date",
            "ticker",
            "owner_normalized",
            "group_no",
            "account_holder_normalized",
            "account_name_normalized",
            "account_occurrence",
        ]
    ).any()
