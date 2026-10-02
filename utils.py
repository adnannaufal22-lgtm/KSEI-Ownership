from __future__ import annotations

from io import BytesIO
import re

import numpy as np
import pandas as pd


LEGAL_ENTITY_MARKERS = {"PT", "CV", "UD", "PD"}
PERSON_PREFIX_TITLES = (
    "PROF",
    "DR",
    "DRS",
    "DRA",
    "IR",
    "KH",
    "HJ",
    "H",
)
PERSON_SUFFIX_CREDENTIALS = (
    "APTH",
    "SKH",
    "SPOG",
    "SPM",
    "DTH",
    "MKOM",
    "MSI",
    "MSC",
    "MBA",
    "BSC",
    "ACPA",
    "CPA",
    "CFA",
    "CMA",
    "SAK",
    "SIP",
    "SSOS",
    "MAK",
    "MKN",
    "BBA",
    "SE",
    "SH",
    "ST",
    "AK",
    "CA",
    "CN",
    "BA",
    "MS",
    "ME",
    "MM",
    "MH",
    "MT",
)
CORPORATE_IDENTITY_HINTS = {
    "BANK",
    "CAPITAL",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "FUND",
    "HOLDING",
    "HOLDINGS",
    "INC",
    "INSURANCE",
    "INVESTMENT",
    "INVESTMENTS",
    "LIMITED",
    "LTD",
    "MANAGEMENT",
    "SECURITIES",
    "TRUST",
}


def _identity_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def _boundary_legal_marker(tokens: list[str], *, leading: bool) -> tuple[str, int] | None:
    """Identify PT/CV/UD/PD even when the source spells it as P.T. or C.V."""
    if not tokens:
        return None
    ordered_widths = (1, 2)
    for width in ordered_widths:
        if len(tokens) < width:
            continue
        boundary = tokens[:width] if leading else tokens[-width:]
        candidate = "".join(boundary)
        if candidate in LEGAL_ENTITY_MARKERS:
            return candidate, width
    return None


def _boundary_affix(
    tokens: list[str],
    allowed: set[str],
    *,
    leading: bool,
) -> tuple[str, int] | None:
    """Read compact or dotted identity affixes such as MBA, M.B.A., or S.K.H."""
    for width in range(min(4, len(tokens)), 0, -1):
        boundary = tokens[:width] if leading else tokens[-width:]
        candidate = "".join(boundary)
        if candidate in allowed:
            return candidate, width
    return None


def _personal_identity_parts(
    value: object,
) -> tuple[list[str], list[str], list[str]] | None:
    """Return canonical prefix titles, base-name tokens, and suffix credentials."""
    cleaned = _identity_text(value)
    tokens = re.findall(r"[A-Z0-9]+", cleaned.upper())
    if len(tokens) < 2 or _boundary_legal_marker(tokens, leading=True) or _boundary_legal_marker(tokens, leading=False):
        return None

    prefix_set = set(PERSON_PREFIX_TITLES)
    suffix_set = set(PERSON_SUFFIX_CREDENTIALS)
    trailing_set = prefix_set | suffix_set
    prefixes: list[str] = []
    suffixes: list[str] = []

    while tokens:
        match = _boundary_affix(tokens, prefix_set, leading=True)
        if match is None:
            break
        title, width = match
        prefixes.append(title)
        del tokens[:width]
    while tokens:
        match = _boundary_affix(tokens, trailing_set, leading=False)
        if match is None:
            break
        affix, width = match
        if affix in prefix_set:
            prefixes.append(affix)
        else:
            suffixes.append(affix)
        del tokens[-width:]

    if not prefixes and not suffixes:
        return None
    if len(tokens) < 2 or CORPORATE_IDENTITY_HINTS.intersection(tokens):
        return None

    canonical_prefixes = [title for title in PERSON_PREFIX_TITLES if title in set(prefixes)]
    canonical_suffixes = [title for title in PERSON_SUFFIX_CREDENTIALS if title in set(suffixes)]
    return canonical_prefixes, tokens, canonical_suffixes


def canonicalize_legal_entity_name(value: object) -> str:
    """Return a stable display name with an Indonesian legal form at the front.

    Examples such as ``PT DELTA ROYAL SEJAHTERA``, ``PT. DELTA ROYAL
    SEJAHTERA`` and ``DELTA ROYAL SEJAHTERA, PT`` all become
    ``PT DELTA ROYAL SEJAHTERA``. Non-company names retain their source
    spelling apart from whitespace cleanup.
    """
    cleaned = _identity_text(value)
    if not cleaned:
        return ""

    tokens = re.findall(r"[A-Z0-9]+", cleaned.upper())
    markers: list[str] = []
    while tokens:
        match = _boundary_legal_marker(tokens, leading=True)
        if match is None:
            break
        marker, width = match
        markers.append(marker)
        del tokens[:width]
    while tokens:
        match = _boundary_legal_marker(tokens, leading=False)
        if match is None:
            break
        marker, width = match
        markers.append(marker)
        del tokens[-width:]

    if not markers:
        return cleaned
    marker = markers[0]
    return " ".join([marker, *tokens]).strip()


def canonicalize_identity_name(value: object) -> str:
    """Canonicalize company legal forms and personal title placement."""
    company_name = canonicalize_legal_entity_name(value)
    company_tokens = re.findall(r"[A-Z0-9]+", company_name.upper())
    if company_tokens and company_tokens[0] in LEGAL_ENTITY_MARKERS:
        return company_name

    parts = _personal_identity_parts(value)
    if parts is None:
        return company_name
    prefixes, base_tokens, suffixes = parts
    return " ".join([*prefixes, *base_tokens, *suffixes])


def normalize_identity_key(value: object) -> str:
    """Return a stable identity key, excluding harmless personal qualifications."""
    canonical = canonicalize_identity_name(value)
    tokens = re.findall(r"[A-Z0-9]+", canonical.upper())
    parts = _personal_identity_parts(canonical)
    if parts is not None:
        _, base_tokens, _ = parts
        tokens = base_tokens
    return " ".join(tokens)


def personal_qualification_count(value: object) -> int:
    """Count recognized personal titles/credentials for canonical-label scoring."""
    parts = _personal_identity_parts(value)
    if parts is None:
        return 0
    prefixes, _, suffixes = parts
    return len(prefixes) + len(suffixes)


def compact_number(value: float | int | None, decimals: int = 2) -> str:
    """Format a number compactly without changing its stored precision."""
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    units = ((1e12, "tn"), (1e9, "bn"), (1e6, "mn"), (1e3, "k"))
    for divisor, suffix in units:
        if absolute >= divisor:
            return f"{sign}{absolute / divisor:,.{decimals}f} {suffix}"
    return f"{value:,.0f}"


def exact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f}"


def display_number(value: float | int | None, mode: str) -> str:
    """Format a value using the user's display preference."""
    if value is None or pd.isna(value):
        return "—"
    if mode == "Exact":
        return exact_number(value)
    if mode == "Million":
        return f"{float(value) / 1e6:,.2f} mn"
    if mode == "Billion":
        return f"{float(value) / 1e9:,.2f} bn"
    return compact_number(value)


def format_pct(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}%"


def metric_delta(current: float, previous: float | None, display_mode: str = "Compact") -> str | None:
    if previous is None or pd.isna(previous):
        return None
    absolute = current - previous
    relative = absolute / abs(previous) * 100 if previous else np.nan
    absolute_text = display_number(abs(absolute), display_mode)
    sign = "+" if absolute > 0 else "−" if absolute < 0 else ""
    if pd.isna(relative):
        return f"{sign}{absolute_text}"
    return f"{sign}{absolute_text} ({relative:+.1f}%)"


def dataframe_to_excel_bytes(dataframe: pd.DataFrame, sheet_name: str = "Filtered Data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.book[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else "" for cell in column_cells[:200]]
            width = min(max(max((len(value) for value in values), default=0) + 2, 10), 34)
            worksheet.column_dimensions[column_cells[0].column_letter].width = width
    return output.getvalue()
