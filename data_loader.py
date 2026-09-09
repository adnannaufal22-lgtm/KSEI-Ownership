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


def _filename_reporting_date(path: Path) -> pd.Timestamp | None:
    month = _filename_month(path)
    return pd.Period(month, freq="M").to_timestamp("M") if month else None


def _select_monthly_files(files: list[Path]) -> tuple[list[Path], list[str]]:
    """Use the newest file when more than one file is labelled for a month."""
    by_month: dict[str, list[Path]] = {}
    without_month: list[Path] = []
    for path in files:
        month = _filename_month(path)
        if month:
            by_month.setdefault(month, []).append(path)
        else:
            without_month.append(path)

    selected = list(without_month)
    issues: list[str] = []
    for month, month_files in sorted(by_month.items()):
        winner = max(month_files, key=lambda path: (path.stat().st_mtime_ns, path.name.casefold()))
        selected.append(winner)
        for skipped in month_files:
            if skipped != winner:
                issues.append(
                    f"Skipped {skipped.name}: another file for {month} was modified more recently."
                )
    return sorted(selected, key=lambda path: path.name.casefold()), issues


def _coalesce_columns(data: pd.DataFrame, aliases: list[str]) -> tuple[pd.Series | None, list[str]]:
    available = {_normalize_header(column): str(column) for column in data.columns}
    matches: list[str] = []
    for alias in aliases:
        match = available.get(_normalize_header(alias))
        if match and match not in matches:
            matches.append(match)
    if not matches:
        return None, []
    values = data[matches[0]]
    for column in matches[1:]:
        values = values.combine_first(data[column])
    return values, matches


def _find_header_for_fields(
    excel: pd.ExcelFile,
    sheet_name: str,
    field_aliases: list[list[str]],
) -> int | None:
    preview = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=35)
    normalized_groups = [
        {_normalize_header(alias) for alias in aliases}
        for aliases in field_aliases
    ]
    for row_number, row in preview.iterrows():
        normalized = {_normalize_header(value) for value in row.dropna()}
        if all(normalized & aliases for aliases in normalized_groups):
            return int(row_number)
    return None


def _locate_table(
    excel: pd.ExcelFile,
    field_aliases: list[list[str]],
) -> tuple[str, int]:
    for sheet_name in excel.sheet_names:
        header_row = _find_header_for_fields(excel, sheet_name, field_aliases)
        if header_row is not None:
            return sheet_name, header_row
    raise ValueError("No worksheet contains the required table headers.")


def _normalize_reporting_dates(values: pd.Series, path: Path) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce").dt.normalize()
    fallback = _filename_reporting_date(path)
    if fallback is not None:
        dates = dates.fillna(fallback)
    return dates


def _short_type_category(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    match = re.match(r"^([A-Z]{2}):\s*([^()]+)", text, flags=re.IGNORECASE)
    if not match:
        return text
    code, label = match.groups()
    return f"{label.strip()} ({code.upper()})"


def load_excel_folder(folder: str | Path, config: dict) -> tuple[pd.DataFrame, dict]:
    """Load and combine one BEI ownership workbook per reporting month."""
    discovered_files = discover_ownership_files(folder)
    files, issues = _select_monthly_files(discovered_files)
    if not files:
        raise FileNotFoundError(f"No .xlsx or .xlsm ownership files were found in {folder}")

    frames: list[pd.DataFrame] = []
    header_rows: dict[str, int] = {}
    sheet_names: list[str] = []
    all_columns: set[str] = set()
    actual_months: dict[str, str] = {}
    for path in files:
        try:
            excel = pd.ExcelFile(path, engine="openpyxl")
            sheet_name = choose_data_sheet(excel, config)
            raw, header_row = _read_data_sheet(excel, sheet_name, config)
            date_column = next(
                (column for column in raw.columns if _normalize_header(column) in {"DATE", "REPORT_DATE", "REPORTING_DATE", "PERIOD", "AS_OF_DATE"}),
                None,
            )
            if date_column is None:
                raise ValueError("reporting date column was not found")
            periods = pd.to_datetime(raw[date_column], errors="coerce").dropna().dt.to_period("M").astype(str).unique()
            if len(periods) != 1:
                raise ValueError(f"expected one reporting month, found {len(periods)}")
            actual_months[path.name] = str(periods[0])
            expected_month = _filename_month(path)
            if expected_month and expected_month != actual_months[path.name]:
                raise ValueError(
                    f"filename month {expected_month} does not match data month {actual_months[path.name]}"
                )

            raw["SOURCE_FILE"] = path.name
            raw["SOURCE_EXCEL_ROW"] = raw.index + header_row + 2
            frames.append(raw)
            all_columns.update(map(str, raw.columns))
            header_rows[path.name] = header_row + 1
            sheet_names.append(f"{path.name}: {sheet_name}")
        except Exception as error:
            issues.append(f"Skipped {path.name}: {error}.")

    if not frames:
        raise ValueError("None of the ownership files contained a usable ownership table.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    metadata = {
        "sheet_names": sheet_names,
        "data_sheet": f"1% Ownership ({len(actual_months)} monthly files)",
        "source_rows": int(len(combined)),
        "source_columns": int(len(all_columns)),
        "source_files": len(actual_months),
        "source_file_names": list(actual_months),
        "header_rows": header_rows,
        "source_months": actual_months,
        "issues": issues,
    }
    return combined.copy(deep=True), metadata


def load_classification_folder(folder: str | Path, config: dict) -> tuple[pd.DataFrame, dict]:
    """Load BEI's wide investor-classification files into a historical long table."""
    discovered_files = discover_ownership_files(folder)
    files, issues = _select_monthly_files(discovered_files)
    frames: list[pd.DataFrame] = []
    loaded_files: list[str] = []

    date_aliases = config.get("columns", {}).get("date", ["DATE"])
    ticker_aliases = config.get("columns", {}).get("ticker", ["SHARE_CODE", "STOCK_CODE"])
    security_aliases = config.get("columns", {}).get("security_name", ["ISSUER_NAME"])
    total_scripless_aliases = config.get("columns", {}).get(
        "total_scripless",
        ["TOTAL_SCRIPLESS", "TOTAL SCRIPLESS"],
    )

    for path in files:
        try:
            excel = pd.ExcelFile(path, engine="openpyxl")
            sheet_name, header_row = _locate_table(excel, [date_aliases, ticker_aliases])
            raw = pd.read_excel(excel, sheet_name=sheet_name, header=header_row)
            raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")

            date_values, date_columns = _coalesce_columns(raw, date_aliases)
            ticker_values, ticker_columns = _coalesce_columns(raw, ticker_aliases)
            security_values, security_columns = _coalesce_columns(raw, security_aliases)
            scripless_values, scripless_columns = _coalesce_columns(raw, total_scripless_aliases)
            if date_values is None or ticker_values is None:
                raise ValueError("date or stock-code column was not found")

            excluded = set(date_columns + ticker_columns + security_columns + scripless_columns)
            classification_columns = [
                column
                for column in raw.columns
                if column not in excluded
                and not str(column).upper().startswith("UNNAMED")
            ]
            if not classification_columns:
                raise ValueError("no investor-classification columns were found")

            base = pd.DataFrame(
                {
                    "date": _normalize_reporting_dates(date_values, path),
                    "ticker": ticker_values.astype("string").str.strip().str.upper(),
                    "security_name": (
                        security_values.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
                        if security_values is not None
                        else pd.Series(pd.NA, index=raw.index, dtype="string")
                    ),
                    "total_scripless": (
                        pd.to_numeric(scripless_values, errors="coerce")
                        if scripless_values is not None
                        else pd.Series(pd.NA, index=raw.index, dtype="Float64")
                    ),
                },
                index=raw.index,
            )

            for column in classification_columns:
                values = pd.to_numeric(raw[column], errors="coerce")
                if not values.notna().any():
                    continue
                part = base.copy()
                part["classification"] = re.sub(r"\s+", " ", str(column)).strip()
                part["ownership_units"] = values
                part["source_file"] = path.name
                frames.append(part)
            loaded_files.append(path.name)
        except Exception as error:
            issues.append(f"Skipped {path.name}: {error}.")

    columns = [
        "date",
        "ticker",
        "security_name",
        "classification",
        "ownership_units",
        "total_scripless",
        "ownership_pct",
        "source_file",
    ]
    if not frames:
        return pd.DataFrame(columns=columns), {
            "source_files": 0,
            "source_file_names": [],
            "issues": issues or ["No Classification workbooks were found."],
            "reconciliation_mismatches": 0,
        }

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date", "ticker", "classification", "ownership_units"])
    combined = (
        combined.sort_values(["source_file", "date", "ticker", "classification"])
        .drop_duplicates(["date", "ticker", "classification"], keep="last")
        .sort_values(["date", "ticker", "classification"])
        .reset_index(drop=True)
    )
    denominator = combined["total_scripless"].where(combined["total_scripless"].gt(0))
    combined["ownership_pct"] = combined["ownership_units"] / denominator * 100

    for column in ("ticker", "security_name", "classification", "source_file"):
        combined[column] = combined[column].astype("category")

    reconciliation = (
        combined.groupby(["date", "ticker"], as_index=False)
        .agg(classification_total=("ownership_units", "sum"), total_scripless=("total_scripless", "first"))
    )
    mismatch_count = int(
        (reconciliation["classification_total"] - reconciliation["total_scripless"]).abs().gt(0.5).sum()
    )
    metadata = {
        "source_files": len(loaded_files),
        "source_file_names": loaded_files,
        "issues": issues,
        "reconciliation_mismatches": mismatch_count,
    }
    return combined[columns], metadata


def load_type_folder(folder: str | Path, config: dict) -> tuple[pd.DataFrame, dict]:
    """Load BEI's three-row Domestic/Foreign type files into a historical long table."""
    discovered_files = discover_ownership_files(folder)
    files, issues = _select_monthly_files(discovered_files)
    frames: list[pd.DataFrame] = []
    loaded_files: list[str] = []

    date_aliases = config.get("columns", {}).get("date", ["DATE"])
    ticker_aliases = config.get("columns", {}).get("ticker", ["STOCK_CODE", "SHARE_CODE"])
    shares_aliases = config.get("columns", {}).get("number_of_shares", ["NUMBER_OF_SHARES"])
    scripless_aliases = config.get("columns", {}).get("total_scripless", ["TOTAL_SCRIPLESS"])

    for path in files:
        try:
            excel = pd.ExcelFile(path, engine="openpyxl")
            sheet_name, header_row = _locate_table(
                excel,
                [date_aliases, ticker_aliases, shares_aliases],
            )
            raw = pd.read_excel(excel, sheet_name=sheet_name, header=None)
            if header_row + 2 >= len(raw):
                raise ValueError("the three-row type header is incomplete")

            top = raw.iloc[header_row]
            category_row = raw.iloc[header_row + 1]
            band_row = raw.iloc[header_row + 2]
            top_normalized = [_normalize_header(value) for value in top]

            def find_position(aliases: list[str]) -> int | None:
                normalized_aliases = {_normalize_header(alias) for alias in aliases}
                return next(
                    (position for position, value in enumerate(top_normalized) if value in normalized_aliases),
                    None,
                )

            date_position = find_position(date_aliases)
            ticker_position = find_position(ticker_aliases)
            shares_position = find_position(shares_aliases)
            scripless_position = find_position(scripless_aliases)
            if None in (date_position, ticker_position, shares_position, scripless_position):
                raise ValueError("date, stock code, number of shares, or total scripless was not found")

            ownership_columns: list[tuple[int, str, str, str]] = []
            current_residency: str | None = None
            current_category: str | None = None
            for position, top_value in enumerate(top_normalized):
                if top_value == "DOMESTIC":
                    current_residency = "Domestic"
                elif top_value == "FOREIGN":
                    current_residency = "Foreign"
                elif position in {date_position, ticker_position, shares_position, scripless_position}:
                    continue

                category_value = category_row.iloc[position]
                if pd.notna(category_value) and str(category_value).strip():
                    current_category = _short_type_category(category_value)
                band_value = band_row.iloc[position]
                if (
                    current_residency
                    and current_category
                    and pd.notna(band_value)
                    and str(band_value).strip()
                ):
                    ownership_columns.append(
                        (position, current_residency, current_category, str(band_value).strip())
                    )
            if not ownership_columns:
                raise ValueError("no Domestic/Foreign investor-type columns were found")

            values = raw.iloc[header_row + 3 :].copy()
            base = pd.DataFrame(
                {
                    "date": _normalize_reporting_dates(values.iloc[:, date_position], path),
                    "ticker": values.iloc[:, ticker_position].astype("string").str.strip().str.upper(),
                    "number_of_shares": pd.to_numeric(values.iloc[:, shares_position], errors="coerce"),
                    "total_scripless": pd.to_numeric(values.iloc[:, scripless_position], errors="coerce"),
                },
                index=values.index,
            )
            calculated_scrip = base["number_of_shares"] - base["total_scripless"]
            negative_scrip = calculated_scrip.lt(-0.5)
            if negative_scrip.any():
                issues.append(
                    f"{path.name}: {int(negative_scrip.sum())} rows have total scripless above number of shares; scrip shares are unavailable for those rows."
                )
            base["scrip_shares"] = calculated_scrip.where(~negative_scrip).clip(lower=0)

            for position, residency, category, band in ownership_columns:
                part = base.copy()
                part["domestic_foreign"] = residency
                part["investor_category"] = category
                part["holding_band"] = band
                part["ownership_units"] = pd.to_numeric(values.iloc[:, position], errors="coerce")
                part["source_file"] = path.name
                frames.append(part)
            loaded_files.append(path.name)
        except Exception as error:
            issues.append(f"Skipped {path.name}: {error}.")

    columns = [
        "date",
        "ticker",
        "number_of_shares",
        "total_scripless",
        "scrip_shares",
        "domestic_foreign",
        "investor_category",
        "holding_band",
        "ownership_units",
        "ownership_pct",
        "source_file",
    ]
    if not frames:
        return pd.DataFrame(columns=columns), {
            "source_files": 0,
            "source_file_names": [],
            "issues": issues or ["No Type workbooks were found."],
            "reconciliation_mismatches": 0,
            "negative_scrip_rows": 0,
        }

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date", "ticker", "domestic_foreign", "ownership_units"])
    combined = (
        combined.sort_values(
            ["source_file", "date", "ticker", "domestic_foreign", "investor_category", "holding_band"]
        )
        .drop_duplicates(
            ["date", "ticker", "domestic_foreign", "investor_category", "holding_band"],
            keep="last",
        )
        .sort_values(["date", "ticker", "domestic_foreign", "investor_category", "holding_band"])
        .reset_index(drop=True)
    )
    denominator = combined["number_of_shares"].where(combined["number_of_shares"].gt(0))
    combined["ownership_pct"] = combined["ownership_units"] / denominator * 100

    for column in (
        "ticker",
        "domestic_foreign",
        "investor_category",
        "holding_band",
        "source_file",
    ):
        combined[column] = combined[column].astype("category")

    reconciliation = (
        combined.groupby(["date", "ticker"], as_index=False)
        .agg(type_total=("ownership_units", "sum"), total_scripless=("total_scripless", "first"))
    )
    mismatch_count = int(
        (reconciliation["type_total"] - reconciliation["total_scripless"]).abs().gt(0.5).sum()
    )
    summary_rows = combined.drop_duplicates(["date", "ticker"])
    negative_scrip_rows = int(summary_rows["scrip_shares"].isna().sum())
    metadata = {
        "source_files": len(loaded_files),
        "source_file_names": loaded_files,
        "issues": issues,
        "reconciliation_mismatches": mismatch_count,
        "negative_scrip_rows": negative_scrip_rows,
    }
    return combined[columns], metadata
