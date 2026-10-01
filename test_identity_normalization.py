import pandas as pd

from data_processing import build_holder_alias_map, normalize_holder_identity
from utils import canonicalize_legal_entity_name


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
