import pandas as pd

from data_processing import build_holder_alias_map, normalize_holder_identity
from utils import (
    canonicalize_identity_name,
    canonicalize_legal_entity_name,
    normalize_identity_key,
)


def test_company_legal_form_is_canonicalized_at_the_front() -> None:
    variants = [
        "PT DELTA ROYAL SEJAHTERA",
        "PT. DELTA ROYAL SEJAHTERA",
        "P.T. DELTA ROYAL SEJAHTERA",
        "DELTA ROYAL SEJAHTERA, PT",
        "Delta Royal Sejahtera, PT.",
    ]
    assert {
        canonicalize_legal_entity_name(value) for value in variants
    } == {"PT DELTA ROYAL SEJAHTERA"}


def test_non_company_name_keeps_its_source_spelling() -> None:
    assert canonicalize_legal_entity_name("  Richard   Rachmadi Wiriahardja  ") == (
        "Richard Rachmadi Wiriahardja"
    )


def test_personal_title_and_credential_positions_are_canonicalized() -> None:
    variants = [
        "IR GAFUR SULISTYO UMAR MBA",
        "GAFUR SULISTYO UMAR,IR,MBA",
        "Gafur Sulistyo Umar, Ir. M.B.A.",
    ]
    assert {canonicalize_identity_name(value) for value in variants} == {
        "IR GAFUR SULISTYO UMAR MBA"
    }
    assert {normalize_identity_key(value) for value in variants} == {
        "GAFUR SULISTYO UMAR"
    }


def test_personal_title_rules_do_not_rewrite_corporate_names() -> None:
    assert canonicalize_identity_name("THE BANK OF NEW YORK MELLON DR") == (
        "THE BANK OF NEW YORK MELLON DR"
    )
    assert normalize_identity_key("H HOLDINGS INC") == "H HOLDINGS INC"


def test_monthly_alias_map_uses_canonical_company_display_name() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-31", "2026-09-30"]),
            "ticker": ["BULL", "BULL"],
            "investor_name": [
                "PT DELTA ROYAL SEJAHTERA",
                "DELTA ROYAL SEJAHTERA, PT",
            ],
            "ownership_units": [2_251_898_042, 2_251_898_042],
        }
    )
    alias_map, audit = build_holder_alias_map(data)
    assert alias_map == {
        "PT DELTA ROYAL SEJAHTERA": "PT DELTA ROYAL SEJAHTERA",
        "DELTA ROYAL SEJAHTERA, PT": "PT DELTA ROYAL SEJAHTERA",
    }
    assert audit["aliases_merged"] == 1
    assert normalize_holder_identity("DELTA ROYAL SEJAHTERA, PT") == (
        normalize_holder_identity("PT DELTA ROYAL SEJAHTERA")
    )


def test_monthly_alias_map_merges_personal_title_placement_variants() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-31", "2026-09-30"]),
            "ticker": ["OASA", "OASA"],
            "investor_name": [
                "GAFUR SULISTYO UMAR,IR,MBA",
                "IR GAFUR SULISTYO UMAR MBA",
            ],
            "ownership_units": [2_066_136_693, 2_066_136_693],
        }
    )
    alias_map, audit = build_holder_alias_map(data)
    assert set(alias_map.values()) == {"IR GAFUR SULISTYO UMAR MBA"}
    assert audit["aliases_merged"] == 1
