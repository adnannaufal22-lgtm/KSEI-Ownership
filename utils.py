from __future__ import annotations

from io import BytesIO
import re

import numpy as np
import pandas as pd


LEGAL_ENTITY_MARKERS = {"PT", "CV", "UD", "PD"}


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
