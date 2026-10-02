from pathlib import Path

import pandas as pd

from daily_ownership import (
    SIGNAL_ACCUMULATING,
    SIGNAL_INTERNAL_TRANSFER,
    SIGNAL_NO_LONGER_REPORTED,
    SIGNAL_SELLING,
    SIGNAL_UNCHANGED,
    build_account_hierarchy_pivots,
    build_account_position_pivots,
    classify_owner_movement,
    load_daily_ownership_files,
    normalize_identity,
    parse_ksei_share_value,
    resolve_account_share_values,
)


DATA_DIR = Path(__file__).resolve().parent / "BEI_Data" / "5% Ownership"


def test_legal_form_position_uses_one_daily_identity() -> None:
    assert normalize_identity("PT DELTA ROYAL SEJAHTERA") == "PT DELTA ROYAL SEJAHTERA"
    assert normalize_identity("DELTA ROYAL SEJAHTERA, PT") == "PT DELTA ROYAL SEJAHTERA"
    assert normalize_identity("PT. DELTA ROYAL SEJAHTERA") == "PT DELTA ROYAL SEJAHTERA"


def test_delta_daily_history_does_not_split_when_pt_moves_to_suffix() -> None:
    files = [
        DATA_DIR / "peng-2026-09-21-00078-lima-persen.xlsx",
        DATA_DIR / "peng-2026-09-22-00079-lima-persen.xlsx",
    ]
    owners, accounts, movements, account_movements, quality, metadata = load_daily_ownership_files(files)
    delta = owners[
        owners["ticker"].eq("BULL")
        & owners["date"].eq(pd.Timestamp("2026-09-21"))
        & owners["owner_normalized"].eq("PT DELTA ROYAL SEJAHTERA")
    ]
    assert metadata["parsed_files"] == 2
    assert len(delta) == 1
    assert delta.iloc[0]["owner"] == "PT DELTA ROYAL SEJAHTERA"
    assert delta.iloc[0]["shares"] == 2_251_898_042


def test_personal_title_reordering_does_not_split_daily_history() -> None:
    files = [
        DATA_DIR / "peng-2026-08-31-00063-lima-persen.xlsx",
        DATA_DIR / "peng-2026-09-01-00064-lima-persen.xlsx",
    ]
    owners, accounts, movements, account_movements, quality, metadata = load_daily_ownership_files(files)
    identity_key = "GAFUR SULISTYO UMAR"
    gafur = owners[
        owners["ticker"].eq("OASA")
        & owners["date"].eq(pd.Timestamp("2026-08-31"))
        & owners["owner_normalized"].eq(identity_key)
    ]
    latest_movement = movements[
        movements["ticker"].eq("OASA")
        & movements["date"].eq(pd.Timestamp("2026-09-01"))
        & movements["owner_normalized"].eq(identity_key)
    ]
    assert metadata["parsed_files"] == 2
    assert len(gafur) == 1
    assert gafur.iloc[0]["owner"] == "IR GAFUR SULISTYO UMAR MBA"
    assert gafur.iloc[0]["shares"] == 2_066_136_693
    assert len(latest_movement) == 1
    assert latest_movement.iloc[0]["signal"] == SIGNAL_UNCHANGED


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


def test_account_position_pivot_uses_account_shares_and_daily_diff() -> None:
    accounts = pd.DataFrame(
        [
            {
                "date": "2026-09-28",
                "source_date": "2026-09-28",
                "ticker": "BMTR",
                "owner_normalized": "PT MNC ASIA HOLDING TBK",
                "account_holder": "PT MNC SEKURITAS",
                "account_name": "MNC ACCOUNT",
                "shares": 500_000_000,
            },
            {
                "date": "2026-09-28",
                "source_date": "2026-09-28",
                "ticker": "BMTR",
                "owner_normalized": "PT MNC ASIA HOLDING TBK",
                "account_holder": "PT KIWOOM SEKURITAS INDONESIA",
                "account_name": "KIWOOM ACCOUNT",
                "shares": 100_000_000,
            },
            {
                "date": "2026-09-29",
                "source_date": "2026-09-29",
                "ticker": "BMTR",
                "owner_normalized": "PT MNC ASIA HOLDING TBK",
                "account_holder": "PT MNC SEKURITAS",
                "account_name": "MNC ACCOUNT",
                "shares": 418_672_500,
            },
            {
                "date": "2026-09-29",
                "source_date": "2026-09-29",
                "ticker": "BMTR",
                "owner_normalized": "PT MNC ASIA HOLDING TBK",
                "account_holder": "PT KIWOOM SEKURITAS INDONESIA",
                "account_name": "KIWOOM ACCOUNT",
                "shares": 181_327_500,
            },
            # The next report repeats the prior date; it must not double the
            # 28-Sep position in the cross-file account pivot.
            {
                "date": "2026-09-28",
                "source_date": "2026-09-29",
                "ticker": "BMTR",
                "owner_normalized": "PT MNC ASIA HOLDING TBK",
                "account_holder": "PT MNC SEKURITAS",
                "account_name": "MNC ACCOUNT",
                "shares": 500_000_000,
            },
            {
                "date": "2026-09-28",
                "source_date": "2026-09-29",
                "ticker": "BMTR",
                "owner_normalized": "PT MNC ASIA HOLDING TBK",
                "account_holder": "PT KIWOOM SEKURITAS INDONESIA",
                "account_name": "KIWOOM ACCOUNT",
                "shares": 100_000_000,
            },
        ]
    )

    position, movement = build_account_position_pivots(
        accounts,
        "BMTR",
        "PT MNC ASIA HOLDING TBK",
        "institution",
    )

    assert position.loc[pd.Timestamp("2026-09-28"), "PT MNC SEKURITAS"] == 500_000_000
    assert position.loc[pd.Timestamp("2026-09-29"), "PT KIWOOM SEKURITAS INDONESIA"] == 181_327_500
    assert movement.loc[pd.Timestamp("2026-09-29"), "PT MNC SEKURITAS"] == -81_327_500
    assert movement.loc[pd.Timestamp("2026-09-29"), "PT KIWOOM SEKURITAS INDONESIA"] == 81_327_500
    assert movement.iloc[0].isna().all()


def test_account_hierarchy_pivot_shows_multiple_owners_and_two_levels() -> None:
    accounts = pd.DataFrame(
        [
            ("2026-09-28", "2026-09-28", "OASA", "OWNER A", "Owner A", "Broker X", "Acct A", 100),
            ("2026-09-29", "2026-09-29", "OASA", "OWNER A", "Owner A", "Broker X", "Acct A", 130),
            ("2026-09-28", "2026-09-28", "OASA", "OWNER B", "Owner B", "Broker Y", "Acct B", 70),
            ("2026-09-29", "2026-09-29", "OASA", "OWNER B", "Owner B", "Broker Y", "Acct B", 50),
        ],
        columns=[
            "date", "source_date", "ticker", "owner_normalized", "owner",
            "account_holder", "account_name", "shares",
        ],
    )

    position, movement = build_account_hierarchy_pivots(
        accounts, "OASA", "institution"
    )

    assert isinstance(position.columns, pd.MultiIndex)
    assert position.columns.names == ["Beneficial Owner", "Institution"]
    assert set(position.columns.get_level_values(0)) == {"Owner A", "Owner B"}
    assert position.loc[pd.Timestamp("2026-09-29"), ("Owner A", "Broker X")] == 130
    assert movement.loc[pd.Timestamp("2026-09-29"), ("Owner A", "Broker X")] == 30
    assert movement.loc[pd.Timestamp("2026-09-29"), ("Owner B", "Broker Y")] == -20


def test_account_hierarchy_account_name_mode_and_optional_owner_filter() -> None:
    accounts = pd.DataFrame(
        [
            ("2026-09-28", "2026-09-28", "OASA", "OWNER A", "Owner A", "Broker X", "Primary", 100),
            ("2026-09-29", "2026-09-29", "OASA", "OWNER A", "Owner A", "Broker X", "Primary", 125),
            ("2026-09-28", "2026-09-28", "OASA", "OWNER B", "Owner B", "Broker Y", "Secondary", 70),
            ("2026-09-29", "2026-09-29", "OASA", "OWNER B", "Owner B", "Broker Y", "Secondary", 75),
        ],
        columns=[
            "date", "source_date", "ticker", "owner_normalized", "owner",
            "account_holder", "account_name", "shares",
        ],
    )

    position, movement = build_account_hierarchy_pivots(
        accounts, "OASA", "account", owners=["OWNER A"]
    )

    assert position.columns.names == ["Beneficial Owner", "Account Name"]
    assert list(position.columns) == [("Owner A", "Primary")]
    assert movement.iloc[-1, 0] == 25
