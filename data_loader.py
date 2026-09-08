from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Any

import pandas as pd


def _source_for_pandas(source: bytes | str | Path) -> BytesIO | str | Path:
    return BytesIO(source) if isinstance(source, bytes) else source


def _normalize_header(value: Any) -> str:
    return "_".join(str(value).strip().upper().replace("/", " ").replace("-", " ").split())


def choose_data_sheet(excel: pd.ExcelFile, config: dict) -> str:
    names = excel.sheet_names
    normalized = {_normalize_header(name): name for name in names}
    for candidate in config.get("sheet_candidates", []):
        match = normalized.get(_normalize_header(candidate))
        if match:
            return match

    aliases = {
        _normalize_header(alias)
        for alias_list in config.get("columns", {}).values()
        for alias in alias_list
    }
    best_sheet = names[0]
    best_score = -1
    for name in names:
        preview = pd.read_excel(excel, sheet_name=name, nrows=3)
        score = sum(_normalize_header(column) in aliases for column in preview.columns)
        if score > best_score:
            best_sheet, best_score = name, score
    return best_sheet


def _find_header_row(excel: pd.ExcelFile, sheet_name: str, config: dict) -> int:
    """Find the ownership table header below BEI's disclaimer rows."""
    preview = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=30)
    aliases = {
        _normalize_header(alias)
        for standard_name in ("date", "ticker", "investor_name", "ownership_units")
        for alias in config.get("columns", {}).get(standard_name, [])
    }
    for row_number, row in preview.iterrows():
        normalized = {_normalize_header(value) for value in row.dropna()}
        if len(normalized & aliases) >= 4:
            return int(row_number)
    return 0


def _read_data_sheet(excel: pd.ExcelFile, sheet_name: str, config: dict) -> tuple[pd.DataFrame, int]:
    header_row = _find_header_row(excel, sheet_name, config)
    raw = pd.read_excel(excel, sheet_name=sheet_name, header=header_row)
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return raw, header_row


def load_excel(source: bytes | str | Path, config: dict) -> tuple[pd.DataFrame, dict]:
    pandas_source = _source_for_pandas(source)
    excel = pd.ExcelFile(pandas_source, engine="openpyxl")
    sheet_name = choose_data_sheet(excel, config)
    raw, header_row = _read_data_sheet(excel, sheet_name, config)
    metadata = {
        "sheet_names": excel.sheet_names,
        "data_sheet": sheet_name,
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "source_files": 1,
        "source_file_names": [],
        "header_rows": {sheet_name: header_row + 1},
    }
    return raw.copy(deep=True), metadata


def discover_ownership_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = [
        path
        for pattern in ("*.xlsx", "*.xlsm")
        for path in folder.glob(pattern)
        if not path.name.startswith("~$")
    ]
    return sorted(set(files), key=lambda path: path.name.casefold())


def folder_signature(folder: str | Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in discover_ownership_files(folder)
    )


def _filename_month(path: Path) -> str | None:
    match = re.search(r"(?<!\d)(20\d{2})[-_ ](0[1-9]|1[0-2])(?!\d)", path.stem)
    return f"{match.group(1)}-{match.group(2)}" if match else None


def load_excel_folder(folder: str | Path, config: dict) -> tuple[pd.DataFrame, dict]:
    """Load and combine one BEI ownership workbook per reporting month."""
    files = discover_ownership_files(folder)
    if not files:
        raise FileNotFoundError(f"No .xlsx or .xlsm ownership files were found in {folder}")

    filename_months: dict[str, list[str]] = {}
    for path in files:
        month = _filename_month(path)
        if month:
            filename_months.setdefault(month, []).append(path.name)
    duplicate_months = {month: names for month, names in filename_months.items() if len(names) > 1}
    if duplicate_months:
        detail = "; ".join(f"{month}: {', '.join(names)}" for month, names in sorted(duplicate_months.items()))
        raise ValueError(f"Multiple BEI workbooks were found for the same month ({detail}). Keep one file per month.")

    frames: list[pd.DataFrame] = []
    header_rows: dict[str, int] = {}
    sheet_names: list[str] = []
    all_columns: set[str] = set()
    actual_months: dict[str, str] = {}
    for path in files:
        excel = pd.ExcelFile(path, engine="openpyxl")
        sheet_name = choose_data_sheet(excel, config)
        raw, header_row = _read_data_sheet(excel, sheet_name, config)
        raw["SOURCE_FILE"] = path.name
        raw["SOURCE_EXCEL_ROW"] = raw.index + header_row + 2
        frames.append(raw)
        all_columns.update(map(str, raw.columns))
        header_rows[path.name] = header_row + 1
        sheet_names.append(f"{path.name}: {sheet_name}")

        date_column = next(
            (column for column in raw.columns if _normalize_header(column) in {"DATE", "REPORT_DATE", "REPORTING_DATE", "PERIOD", "AS_OF_DATE"}),
            None,
        )
        if date_column is not None:
            periods = pd.to_datetime(raw[date_column], errors="coerce").dropna().dt.to_period("M").astype(str).unique()
            if len(periods) != 1:
                raise ValueError(f"{path.name} must contain exactly one reporting month; found {len(periods)}.")
            actual_months[path.name] = str(periods[0])
            expected_month = _filename_month(path)
            if expected_month and expected_month != actual_months[path.name]:
                raise ValueError(
                    f"{path.name} is labeled {expected_month}, but its data belongs to {actual_months[path.name]}."
                )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    metadata = {
        "sheet_names": sheet_names,
        "data_sheet": f"BEI_Data ({len(files)} monthly files)",
        "source_rows": int(len(combined)),
        "source_columns": int(len(all_columns)),
        "source_files": len(files),
        "source_file_names": [path.name for path in files],
        "header_rows": header_rows,
        "source_months": actual_months,
    }
    return combined.copy(deep=True), metadata
