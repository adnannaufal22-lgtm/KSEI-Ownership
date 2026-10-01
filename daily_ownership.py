from __future__ import annotations

import math
import numbers
import gzip
import hashlib
import pickle
import re
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SIGNAL_ACCUMULATING = "ACCUMULATING"
SIGNAL_SELLING = "SELLING"
SIGNAL_UNCHANGED = "UNCHANGED"
SIGNAL_INTERNAL_TRANSFER = "INTERNAL TRANSFER"
SIGNAL_ENTERED = "ENTERED >5%"
SIGNAL_EXITED = "EXITED >5%"
SIGNAL_NEWLY_REPORTED = "NEWLY REPORTED"
SIGNAL_NO_LONGER_REPORTED = "NO LONGER REPORTED"
SIGNAL_DATA_ISSUE = "DATA ISSUE"
CACHE_VERSION = 2

_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MEI": 5,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AGU": 8,
    "AGS": 8,
    "AUG": 8,
    "SEP": 9,
    "OKT": 10,
    "OCT": 10,
    "NOV": 11,
    "DES": 12,
    "DEC": 12,
}

OWNER_COLUMNS = [
    "date",
    "ticker",
    "issuer",
    "owner",
    "owner_raw",
    "owner_normalized",
    "nationality",
    "domicile",
    "local_foreign",
    "shares",
    "ownership_pct",
    "raw_share_value",
    "normalized_share_value",
    "source_file",
    "source_date",
    "group_no",
    "account_share_sum",
    "reconciliation_difference",
    "reconciliation_ok",
]

ACCOUNT_COLUMNS = [
    "date",
    "ticker",
    "issuer",
    "owner",
    "owner_raw",
    "owner_normalized",
    "account_holder",
    "account_holder_raw",
    "account_holder_normalized",
    "account_name",
    "account_name_raw",
    "account_name_normalized",
    "shares",
    "raw_share_value",
    "normalized_share_value",
    "source_file",
    "source_date",
    "group_no",
    "account_occurrence",
    "source_row",
]

MOVEMENT_COLUMNS = [
    "date",
    "previous_date",
    "ticker",
    "issuer",
    "owner",
    "owner_raw",
    "owner_normalized",
    "previous_shares",
    "current_shares",
    "delta_shares",
    "previous_pct",
    "current_pct",
    "delta_pct_point",
    "signal",
    "number_of_accounts",
    "accounts_changed",
    "account_reallocation_flag",
    "nationality",
    "domicile",
    "local_foreign",
    "source_file",
    "source_date",
    "group_no",
    "quality_status",
]

ACCOUNT_MOVEMENT_COLUMNS = [
    "date",
    "previous_date",
    "ticker",
    "issuer",
    "owner",
    "owner_raw",
    "owner_normalized",
    "account_holder",
    "account_holder_raw",
    "account_holder_normalized",
    "account_name",
    "account_name_raw",
    "account_name_normalized",
    "previous_shares",
    "current_shares",
    "delta_shares",
    "direction",
    "source_file",
    "source_date",
    "group_no",
    "account_occurrence",
    "source_row",
]

QUALITY_COLUMNS = [
    "severity",
    "check",
    "source_file",
    "date",
    "ticker",
    "owner",
    "message",
]

COMPACT_TEXT_COLUMNS = {
    "ticker",
    "issuer",
    "owner",
    "owner_raw",
    "owner_normalized",
    "nationality",
    "domicile",
    "local_foreign",
    "source_file",
    "group_no",
    "account_holder",
    "account_holder_raw",
    "account_holder_normalized",
    "account_name",
    "account_name_raw",
    "account_name_normalized",
    "signal",
    "direction",
    "quality_status",
    "severity",
    "check",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def normalize_identity(value: object) -> str:
    """Conservative identity normalization; raw names remain available for audit."""
    text = clean_text(value).upper()
    text = re.sub(r"\bPT\s*[\.,]?\s*", "PT ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Store repeated labels as categories to keep the full history memory-safe."""
    if data.empty:
        return data
    for column in COMPACT_TEXT_COLUMNS.intersection(data.columns):
        data[column] = data[column].astype("category")
    return data


def parse_ksei_share_value(value: object) -> int | float:
    """Parse mixed KSEI Excel shares without losing the source's thousand scaling.

    Numeric Excel cells below 1,000 are stored in thousands by the source. Text
    values containing grouping separators are already exact share counts.
    """
    if _is_missing(value):
        return np.nan
    if isinstance(value, bool):
        return np.nan
    if isinstance(value, numbers.Number):
        numeric = float(value)
        if not math.isfinite(numeric):
            return np.nan
        if numeric == 0:
            return 0
        if abs(numeric) < 1000:
            return int(round(numeric * 1000))
        return int(round(numeric))

    text = clean_text(value)
    if not text or text.upper() in {"N/A", "NA", "NONE", "NAN", "-", "—"}:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(" ", "")
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+", text):
        numeric = int(text.replace(",", ""))
    elif re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3}){2,}", text):
        numeric = int(text.replace(".", ""))
    elif re.fullmatch(r"[-+]?\d+", text):
        numeric = int(text)
    elif re.fullmatch(r"[-+]?\d+\.\d+", text):
        numeric = int(round(float(text)))
    else:
        cleaned = re.sub(r"[^0-9+\-\.]", "", text.replace(",", ""))
        try:
            numeric = int(round(float(cleaned)))
        except ValueError:
            return np.nan
    return -abs(numeric) if negative else numeric


def _account_share_candidates(value: object) -> list[int | float]:
    """Return exact and thousand-scaled candidates for ambiguous numeric cells."""
    if _is_missing(value) or isinstance(value, bool):
        return [np.nan]
    if isinstance(value, numbers.Number):
        numeric = float(value)
        if not math.isfinite(numeric):
            return [np.nan]
        exact = int(round(numeric))
        if numeric != 0 and abs(numeric) < 1000:
            scaled = int(round(numeric * 1000))
            return [exact, scaled] if exact != scaled else [exact]
        return [exact]
    return [parse_ksei_share_value(value)]


def resolve_account_share_values(
    raw_values: list[object],
    combined_investor_shares: object,
) -> tuple[list[int | float], float]:
    """Resolve ambiguous account cells against the authoritative combined total."""
    candidates = [_account_share_candidates(value) for value in raw_values]
    if _is_missing(combined_investor_shares):
        resolved = [options[-1] for options in candidates]
        return resolved, np.nan

    target = float(combined_investor_shares)
    ambiguous = [index for index, options in enumerate(candidates) if len(options) > 1]
    baseline = [options[0] for options in candidates]
    if len(ambiguous) <= 18:
        best_values = baseline
        best_score = (math.inf, math.inf)
        for choices in product((0, 1), repeat=len(ambiguous)):
            values = baseline.copy()
            scaled_count = 0
            for index, choice in zip(ambiguous, choices):
                values[index] = candidates[index][choice]
                scaled_count += choice
            total = sum(float(value) for value in values if not _is_missing(value))
            # Prefer reconciled totals, then the source-default thousand interpretation.
            score = (abs(target - total), len(ambiguous) - scaled_count)
            if score < best_score:
                best_values = values
                best_score = score
        return best_values, target - sum(
            float(value) for value in best_values if not _is_missing(value)
        )

    # Large groups are rare. Use a deterministic greedy reconciliation fallback.
    resolved = baseline.copy()
    current_total = sum(float(value) for value in resolved if not _is_missing(value))
    for index in sorted(
        ambiguous,
        key=lambda item: abs(float(candidates[item][1]) - float(candidates[item][0])),
        reverse=True,
    ):
        upgraded_total = current_total - float(resolved[index]) + float(candidates[index][1])
        if abs(target - upgraded_total) < abs(target - current_total):
            resolved[index] = candidates[index][1]
            current_total = upgraded_total
    return resolved, target - current_total


def parse_percentage(value: object) -> float:
    if _is_missing(value):
        return np.nan
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return float(value)
    text = clean_text(value).replace("%", "").replace(" ", "")
    if not text:
        return np.nan
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def _header_key(value: object) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"[^A-Z0-9%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _header_date(value: object) -> pd.Timestamp | None:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).normalize()
    text = clean_text(value).upper()
    match = re.search(r"(\d{1,2})[\-/ ]([A-Z]{3})[\-/ ](20\d{2})", text)
    if match and match.group(2) in _MONTHS:
        return pd.Timestamp(int(match.group(3)), _MONTHS[match.group(2)], int(match.group(1)))
    return None


def _first_value(values: Iterable[object]) -> object:
    for value in values:
        if not _is_missing(value) and clean_text(value):
            return value
    return np.nan


def _locate_headers(raw: pd.DataFrame) -> tuple[int, dict[str, int], dict[pd.Timestamp, dict[str, int]]]:
    header_row = None
    for row_index in range(min(20, len(raw))):
        keys = {_header_key(value) for value in raw.iloc[row_index].tolist()}
        if "KODE EFEK" in keys and "NAMA PEMEGANG SAHAM" in keys:
            header_row = row_index
            break
    if header_row is None or header_row + 1 >= len(raw):
        raise ValueError("Could not locate the KSEI two-row header.")

    top = raw.iloc[header_row].copy().ffill()
    sub = raw.iloc[header_row + 1].copy()
    fixed: dict[str, int] = {}
    aliases = {
        "NO": "no",
        "KODE EFEK": "ticker",
        "NAMA EMITEN": "issuer",
        "NAMA PEMEGANG REKENING EFEK": "account_holder",
        "NAMA PEMEGANG SAHAM": "owner",
        "NAMA REKENING EFEK": "account_name",
        "KEBANGSAAN": "nationality",
        "DOMISILI": "domicile",
        "STATUS LOKAL ASING": "local_foreign",
        "STATUS": "local_foreign",
        "PERUBAHAN": "change",
    }
    for column, value in raw.iloc[header_row].items():
        key = _header_key(value)
        if key in aliases and aliases[key] not in fixed:
            fixed[aliases[key]] = int(column)

    required = {"no", "ticker", "issuer", "account_holder", "owner", "account_name"}
    missing = sorted(required.difference(fixed))
    if missing:
        raise ValueError(f"Missing required KSEI columns: {', '.join(missing)}")

    date_columns: dict[pd.Timestamp, dict[str, int]] = {}
    for column in raw.columns:
        date = _header_date(top.iloc[int(column)])
        if date is None:
            continue
        subkey = _header_key(sub.iloc[int(column)])
        bucket = date_columns.setdefault(date, {})
        if subkey == "JUMLAH SAHAM":
            bucket["account_shares"] = int(column)
        elif "SAHAM GABUNGAN PER INVESTOR" in subkey:
            bucket["combined_shares"] = int(column)
        elif "PERSENTASE" in subkey:
            bucket["ownership_pct"] = int(column)

    valid = {
        date: columns
        for date, columns in date_columns.items()
        if {"account_shares", "combined_shares", "ownership_pct"}.issubset(columns)
    }
    if len(valid) < 2:
        raise ValueError("Expected at least two dated KSEI ownership column groups.")
    return header_row, fixed, valid


def classify_owner_movement(
    previous_shares: object,
    current_shares: object,
    previous_pct: object,
    current_pct: object,
    account_changed: bool,
) -> str:
    previous_available = not _is_missing(previous_shares)
    current_available = not _is_missing(current_shares)
    previous_pct_available = not _is_missing(previous_pct)
    current_pct_available = not _is_missing(current_pct)

    if previous_pct_available and current_pct_available:
        if float(previous_pct) < 5 <= float(current_pct):
            return SIGNAL_ENTERED
        if float(previous_pct) >= 5 > float(current_pct):
            return SIGNAL_EXITED
    if not previous_available and current_available:
        return SIGNAL_NEWLY_REPORTED
    if previous_available and not current_available:
        return SIGNAL_NO_LONGER_REPORTED
    if not previous_available or not current_available:
        return SIGNAL_DATA_ISSUE

    delta = float(current_shares) - float(previous_shares)
    tolerance = 0.5
    if delta > tolerance:
        return SIGNAL_ACCUMULATING
    if delta < -tolerance:
        return SIGNAL_SELLING
    if account_changed:
        return SIGNAL_INTERNAL_TRANSFER
    return SIGNAL_UNCHANGED


def _quality_row(
    severity: str,
    check: str,
    path: Path,
    date: object,
    ticker: object,
    owner: object,
    message: str,
) -> dict[str, object]:
    return {
        "severity": severity,
        "check": check,
        "source_file": path.name,
        "date": date,
        "ticker": clean_text(ticker),
        "owner": clean_text(owner),
        "message": message,
    }


def parse_daily_ownership_file(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object, engine="openpyxl")
    header_row, fixed, date_columns = _locate_headers(raw)
    dates = sorted(date_columns)[-2:]
    previous_date, current_date = dates[0], dates[1]
    source_date = current_date
    body = raw.iloc[header_row + 2 :].copy()
    body = body.dropna(how="all")
    if body.empty:
        raise ValueError("No ownership rows were found below the KSEI header.")

    body["_group_no"] = body.iloc[:, fixed["no"]].ffill()
    numeric_group = pd.to_numeric(body["_group_no"], errors="coerce")
    body = body[numeric_group.notna()].copy()
    body["_group_no"] = numeric_group[numeric_group.notna()].astype(int)

    owners: list[dict[str, object]] = []
    accounts: list[dict[str, object]] = []
    movements: list[dict[str, object]] = []
    account_movements: list[dict[str, object]] = []
    quality: list[dict[str, object]] = []

    for group_no, group in body.groupby("_group_no", sort=False, dropna=False):
        group_values = group.iloc[:, : raw.shape[1]].to_numpy(dtype=object)
        group_indices = group.index.to_numpy()
        ticker_raw = _first_value(group_values[:, fixed["ticker"]])
        issuer_raw = _first_value(group_values[:, fixed["issuer"]])
        owner_raw = _first_value(group_values[:, fixed["owner"]])
        nationality_raw = _first_value(group_values[:, fixed.get("nationality", fixed["owner"])])
        domicile_raw = _first_value(group_values[:, fixed.get("domicile", fixed["owner"])])
        status_raw = _first_value(group_values[:, fixed.get("local_foreign", fixed["owner"])])
        ticker = clean_text(ticker_raw).upper()
        issuer = clean_text(issuer_raw)
        owner_text = clean_text(owner_raw)
        owner = owner_text or "(MISSING BENEFICIAL OWNER)"
        owner_normalized = normalize_identity(owner_text) or f"MISSING {path.name} {group_no}"

        if not ticker:
            quality.append(_quality_row("ERROR", "missing_ticker", path, current_date, ticker, owner, "Investor group has no ticker."))
        if not owner_text:
            quality.append(_quality_row("ERROR", "missing_owner", path, current_date, ticker, owner, "Investor group has no Nama Pemegang Saham."))

        owner_values: dict[pd.Timestamp, tuple[object, float, float]] = {}
        for date in dates:
            columns = date_columns[date]
            raw_combined = _first_value(group_values[:, columns["combined_shares"]])
            combined = parse_ksei_share_value(raw_combined)
            pct = parse_percentage(_first_value(group_values[:, columns["ownership_pct"]]))
            owner_values[date] = (raw_combined, combined, pct)

        group_accounts: list[dict[str, object]] = []
        occurrence_counts: dict[tuple[str, str], int] = {}
        for position, row in enumerate(group_values):
            row_index = int(group_indices[position])
            holder_raw = row[fixed["account_holder"]]
            account_name_raw = row[fixed["account_name"]]
            holder = clean_text(holder_raw)
            account_name = clean_text(account_name_raw)
            if not holder and not account_name:
                continue
            key = (normalize_identity(holder), normalize_identity(account_name))
            occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
            group_accounts.append(
                {
                    "account_holder": holder or "(MISSING ACCOUNT HOLDER)",
                    "account_holder_raw": holder_raw,
                    "account_holder_normalized": key[0],
                    "account_name": account_name or "(MISSING ACCOUNT NAME)",
                    "account_name_raw": account_name_raw,
                    "account_name_normalized": key[1],
                    "account_occurrence": occurrence_counts[key],
                    "source_row": int(row_index) + 1,
                    "row": row,
                },
            )

        resolved_accounts: dict[pd.Timestamp, dict[int, int | float]] = {}
        for date in dates:
            column = date_columns[date]["account_shares"]
            raw_values = [record["row"][column] for record in group_accounts]
            combined = owner_values[date][1]
            resolved_values, residual = resolve_account_share_values(raw_values, combined)
            resolved_accounts[date] = {
                int(record["source_row"]): value
                for record, value in zip(group_accounts, resolved_values)
            }
            if not _is_missing(residual) and abs(float(residual)) > 1.0:
                quality.append(
                    _quality_row(
                        "WARNING",
                        "share_scale_resolution",
                        path,
                        date,
                        ticker,
                        owner,
                        f"Ambiguous numeric account cells could not reconcile within one share; residual {float(residual):+,.0f}.",
                    )
                )

        account_totals: dict[pd.Timestamp, float] = {date: 0.0 for date in dates}
        account_total_seen: dict[pd.Timestamp, bool] = {date: False for date in dates}
        changed_accounts = 0
        current_account_count = 0

        for account_record in group_accounts:
            parsed_by_date: dict[pd.Timestamp, float] = {}
            for date in dates:
                column = date_columns[date]["account_shares"]
                raw_value = account_record["row"][column]
                parsed = resolved_accounts[date][int(account_record["source_row"])]
                parsed_by_date[date] = parsed
                if not _is_missing(parsed):
                    account_totals[date] += float(parsed)
                    account_total_seen[date] = True
                accounts.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "issuer": issuer,
                        "owner": owner,
                        "owner_raw": owner_raw,
                        "owner_normalized": owner_normalized,
                        "account_holder": account_record["account_holder"],
                        "account_holder_raw": account_record["account_holder_raw"],
                        "account_holder_normalized": account_record["account_holder_normalized"],
                        "account_name": account_record["account_name"],
                        "account_name_raw": account_record["account_name_raw"],
                        "account_name_normalized": account_record["account_name_normalized"],
                        "shares": parsed,
                        "raw_share_value": raw_value,
                        "normalized_share_value": parsed,
                        "source_file": path.name,
                        "source_date": source_date,
                        "group_no": group_no,
                        "account_occurrence": account_record["account_occurrence"],
                        "source_row": account_record["source_row"],
                    }
                )

            previous_account = parsed_by_date[previous_date]
            current_account = parsed_by_date[current_date]
            comparable_previous = 0.0 if _is_missing(previous_account) and not _is_missing(current_account) else previous_account
            comparable_current = 0.0 if _is_missing(current_account) and not _is_missing(previous_account) else current_account
            if _is_missing(comparable_previous) or _is_missing(comparable_current):
                delta_account = np.nan
                direction = "DATA ISSUE"
            else:
                delta_account = float(comparable_current) - float(comparable_previous)
                direction = "INCREASE" if delta_account > 0.5 else "DECREASE" if delta_account < -0.5 else "UNCHANGED"
                if abs(delta_account) > 0.5:
                    changed_accounts += 1
            if not _is_missing(current_account) and float(current_account) > 0.5:
                current_account_count += 1
            account_movements.append(
                {
                    "date": current_date,
                    "previous_date": previous_date,
                    "ticker": ticker,
                    "issuer": issuer,
                    "owner": owner,
                    "owner_raw": owner_raw,
                    "owner_normalized": owner_normalized,
                    "account_holder": account_record["account_holder"],
                    "account_holder_raw": account_record["account_holder_raw"],
                    "account_holder_normalized": account_record["account_holder_normalized"],
                    "account_name": account_record["account_name"],
                    "account_name_raw": account_record["account_name_raw"],
                    "account_name_normalized": account_record["account_name_normalized"],
                    "previous_shares": previous_account,
                    "current_shares": current_account,
                    "delta_shares": delta_account,
                    "direction": direction,
                    "source_file": path.name,
                    "source_date": source_date,
                    "group_no": group_no,
                    "account_occurrence": account_record["account_occurrence"],
                    "source_row": account_record["source_row"],
                }
            )

        reconciliation_ok = True
        for date in dates:
            raw_combined, combined, pct = owner_values[date]
            account_sum = account_totals[date] if account_total_seen[date] else np.nan
            difference = combined - account_sum if not _is_missing(combined) and not _is_missing(account_sum) else np.nan
            row_ok = _is_missing(difference) or abs(float(difference)) <= 1.0
            reconciliation_ok = reconciliation_ok and row_ok
            if not row_ok:
                quality.append(
                    _quality_row(
                        "WARNING",
                        "account_reconciliation",
                        path,
                        date,
                        ticker,
                        owner,
                        f"Account shares differ from combined investor shares by {float(difference):+,.0f}.",
                    )
                )
            if not _is_missing(combined) and float(combined) < 0:
                quality.append(_quality_row("ERROR", "negative_holding", path, date, ticker, owner, "Combined investor holding is negative."))
            owner_values[date] = (raw_combined, combined, pct)
            owners.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "issuer": issuer,
                    "owner": owner,
                    "owner_raw": owner_raw,
                    "owner_normalized": owner_normalized,
                    "nationality": clean_text(nationality_raw),
                    "domicile": clean_text(domicile_raw),
                    "local_foreign": clean_text(status_raw),
                    "shares": combined,
                    "ownership_pct": pct,
                    "raw_share_value": raw_combined,
                    "normalized_share_value": combined,
                    "source_file": path.name,
                    "source_date": source_date,
                    "group_no": group_no,
                    "account_share_sum": account_sum,
                    "reconciliation_difference": difference,
                    "reconciliation_ok": row_ok,
                }
            )

        previous_combined = owner_values[previous_date][1]
        current_combined = owner_values[current_date][1]
        previous_pct = owner_values[previous_date][2]
        current_pct = owner_values[current_date][2]
        delta_shares = (
            float(current_combined) - float(previous_combined)
            if not _is_missing(previous_combined) and not _is_missing(current_combined)
            else np.nan
        )
        delta_pct = (
            float(current_pct) - float(previous_pct)
            if not _is_missing(previous_pct) and not _is_missing(current_pct)
            else np.nan
        )
        signal = classify_owner_movement(
            previous_combined,
            current_combined,
            previous_pct,
            current_pct,
            changed_accounts > 0,
        )
        movements.append(
            {
                "date": current_date,
                "previous_date": previous_date,
                "ticker": ticker,
                "issuer": issuer,
                "owner": owner,
                "owner_raw": owner_raw,
                "owner_normalized": owner_normalized,
                "previous_shares": previous_combined,
                "current_shares": current_combined,
                "delta_shares": delta_shares,
                "previous_pct": previous_pct,
                "current_pct": current_pct,
                "delta_pct_point": delta_pct,
                "signal": signal,
                "number_of_accounts": current_account_count,
                "accounts_changed": changed_accounts,
                "account_reallocation_flag": signal == SIGNAL_INTERNAL_TRANSFER,
                "nationality": clean_text(nationality_raw),
                "domicile": clean_text(domicile_raw),
                "local_foreign": clean_text(status_raw),
                "source_file": path.name,
                "source_date": source_date,
                "group_no": group_no,
                "quality_status": "OK" if reconciliation_ok else "REVIEW",
            }
        )

    return (
        pd.DataFrame(owners, columns=OWNER_COLUMNS),
        pd.DataFrame(accounts, columns=ACCOUNT_COLUMNS),
        pd.DataFrame(movements, columns=MOVEMENT_COLUMNS),
        pd.DataFrame(account_movements, columns=ACCOUNT_MOVEMENT_COLUMNS),
        pd.DataFrame(quality, columns=QUALITY_COLUMNS),
    )


def _deduplicate(
    data: pd.DataFrame,
    keys: list[str],
    value_columns: list[str],
    quality: list[dict[str, object]],
    check_name: str,
) -> pd.DataFrame:
    if data.empty:
        return data
    duplicated = data[data.duplicated(keys, keep=False)].copy()
    if not duplicated.empty:
        # Most duplicates are identical overlapping observations from adjacent
        # daily workbooks. Remove those exact repeats first so the comparatively
        # expensive grouped audit only visits genuinely conflicting keys.
        variants = duplicated.drop_duplicates(keys + value_columns)
        conflicts = variants[variants.duplicated(keys, keep=False)]
        for _, group in conflicts.groupby(keys, dropna=False, sort=False):
            row = group.iloc[-1]
            quality.append(
                {
                    "severity": "WARNING",
                    "check": check_name,
                    "source_file": " | ".join(sorted(group["source_file"].astype(str).unique())),
                    "date": row.get("date"),
                    "ticker": row.get("ticker", ""),
                    "owner": row.get("owner", ""),
                    "message": f"{len(group)} conflicting observations share the same normalized key; newest source retained.",
                }
            )
    return (
        data.sort_values(["source_date", "source_file"], kind="stable")
        .drop_duplicates(keys, keep="last")
        .sort_values([column for column in ("date", "ticker", "owner") if column in data], kind="stable")
        .reset_index(drop=True)
    )


def _aggregate_owner_observations(data: pd.DataFrame) -> pd.DataFrame:
    """Combine repeated beneficial-owner groups within the same source snapshot."""
    if data.empty:
        return data
    keys = ["date", "ticker", "owner_normalized", "source_file", "source_date"]
    duplicate_mask = data.duplicated(keys, keep=False)
    if not duplicate_mask.any():
        return data
    unique_rows = data.loc[~duplicate_mask]
    rows: list[dict[str, object]] = []
    for _, group in data.loc[duplicate_mask].groupby(keys, dropna=False, sort=False):
        row = group.iloc[0].to_dict()
        for column in ("shares", "ownership_pct", "account_share_sum", "reconciliation_difference"):
            row[column] = pd.to_numeric(group[column], errors="coerce").sum(min_count=1)
        row["reconciliation_ok"] = bool(group["reconciliation_ok"].fillna(False).all())
        row["group_no"] = " | ".join(str(value) for value in group["group_no"].dropna().unique())
        raw_names = [clean_text(value) for value in group["owner_raw"] if clean_text(value)]
        row["owner_raw"] = " | ".join(dict.fromkeys(raw_names))
        rows.append(row)
    aggregated = pd.DataFrame(rows, columns=OWNER_COLUMNS)
    return pd.concat([unique_rows, aggregated], ignore_index=True)[OWNER_COLUMNS]


def _aggregate_owner_movements(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    keys = ["date", "ticker", "owner_normalized", "source_file", "source_date"]
    duplicate_mask = data.duplicated(keys, keep=False)
    if not duplicate_mask.any():
        return data
    unique_rows = data.loc[~duplicate_mask]
    rows: list[dict[str, object]] = []
    for _, group in data.loc[duplicate_mask].groupby(keys, dropna=False, sort=False):
        row = group.iloc[0].to_dict()
        for column in ("previous_shares", "current_shares", "previous_pct", "current_pct"):
            row[column] = pd.to_numeric(group[column], errors="coerce").sum(min_count=1)
        row["delta_shares"] = (
            float(row["current_shares"]) - float(row["previous_shares"])
            if not _is_missing(row["current_shares"]) and not _is_missing(row["previous_shares"])
            else np.nan
        )
        row["delta_pct_point"] = (
            float(row["current_pct"]) - float(row["previous_pct"])
            if not _is_missing(row["current_pct"]) and not _is_missing(row["previous_pct"])
            else np.nan
        )
        row["number_of_accounts"] = int(pd.to_numeric(group["number_of_accounts"], errors="coerce").sum())
        row["accounts_changed"] = int(pd.to_numeric(group["accounts_changed"], errors="coerce").sum())
        row["signal"] = classify_owner_movement(
            row["previous_shares"],
            row["current_shares"],
            row["previous_pct"],
            row["current_pct"],
            row["accounts_changed"] > 0,
        )
        row["account_reallocation_flag"] = row["signal"] == SIGNAL_INTERNAL_TRANSFER
        row["quality_status"] = "REVIEW" if group["quality_status"].eq("REVIEW").any() else "OK"
        row["group_no"] = " | ".join(str(value) for value in group["group_no"].dropna().unique())
        raw_names = [clean_text(value) for value in group["owner_raw"] if clean_text(value)]
        row["owner_raw"] = " | ".join(dict.fromkeys(raw_names))
        rows.append(row)
    aggregated = pd.DataFrame(rows, columns=MOVEMENT_COLUMNS)
    return pd.concat([unique_rows, aggregated], ignore_index=True)[MOVEMENT_COLUMNS]


def load_daily_ownership_files(
    files: Iterable[str | Path],
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    owner_frames: list[pd.DataFrame] = []
    account_frames: list[pd.DataFrame] = []
    movement_frames: list[pd.DataFrame] = []
    account_movement_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []
    paths = sorted({Path(file) for file in files}, key=lambda path: path.name.casefold())
    parsed_files = 0
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    for path in paths:
        parsed = None
        file_cache = None
        if cache_path is not None:
            stat = path.stat()
            cache_key = hashlib.sha256(
                f"{CACHE_VERSION}|{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
            ).hexdigest()[:20]
            file_cache = cache_path / f"{path.stem}-{cache_key}.pkl.gz"
            if file_cache.exists():
                try:
                    with gzip.open(file_cache, "rb") as handle:
                        parsed = pickle.load(handle)
                except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
                    parsed = None
        try:
            if parsed is None:
                parsed = parse_daily_ownership_file(path)
                if file_cache is not None:
                    temporary = file_cache.with_suffix(file_cache.suffix + ".tmp")
                    with gzip.open(temporary, "wb", compresslevel=5) as handle:
                        pickle.dump(parsed, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    temporary.replace(file_cache)
            owners, accounts, movements, account_movements, quality = parsed
        except Exception as error:
            quality_rows.append(
                {
                    "severity": "ERROR",
                    "check": "file_parse",
                    "source_file": path.name,
                    "date": pd.NaT,
                    "ticker": "",
                    "owner": "",
                    "message": str(error),
                }
            )
            continue
        parsed_files += 1
        owner_frames.append(owners)
        account_frames.append(accounts)
        movement_frames.append(movements)
        account_movement_frames.append(account_movements)
        quality_frames.append(quality)

    owners = pd.concat(owner_frames, ignore_index=True) if owner_frames else pd.DataFrame(columns=OWNER_COLUMNS)
    accounts = pd.concat(account_frames, ignore_index=True) if account_frames else pd.DataFrame(columns=ACCOUNT_COLUMNS)
    movements = pd.concat(movement_frames, ignore_index=True) if movement_frames else pd.DataFrame(columns=MOVEMENT_COLUMNS)
    account_movements = pd.concat(account_movement_frames, ignore_index=True) if account_movement_frames else pd.DataFrame(columns=ACCOUNT_MOVEMENT_COLUMNS)

    owners = _aggregate_owner_observations(owners)
    movements = _aggregate_owner_movements(movements)

    owners = _deduplicate(
        owners,
        ["date", "ticker", "owner_normalized"],
        ["shares", "ownership_pct"],
        quality_rows,
        "duplicate_owner_observation",
    )
    accounts = _deduplicate(
        accounts,
        ["date", "ticker", "owner_normalized", "group_no", "account_holder_normalized", "account_name_normalized", "account_occurrence"],
        ["shares"],
        quality_rows,
        "duplicate_account_observation",
    )
    movements = _deduplicate(
        movements,
        ["date", "ticker", "owner_normalized"],
        ["previous_shares", "current_shares", "signal"],
        quality_rows,
        "duplicate_owner_movement",
    )
    account_movements = _deduplicate(
        account_movements,
        ["date", "ticker", "owner_normalized", "group_no", "account_holder_normalized", "account_name_normalized", "account_occurrence"],
        ["previous_shares", "current_shares", "delta_shares"],
        quality_rows,
        "duplicate_account_movement",
    )

    quality = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame(columns=QUALITY_COLUMNS)
    if quality_rows:
        quality = pd.concat([quality, pd.DataFrame(quality_rows, columns=QUALITY_COLUMNS)], ignore_index=True)
    if not quality.empty:
        quality = quality.sort_values(["severity", "source_file", "date"], ascending=[True, True, False], kind="stable").reset_index(drop=True)

    owners = _compact_text_columns(owners)
    accounts = _compact_text_columns(accounts)
    movements = _compact_text_columns(movements)
    account_movements = _compact_text_columns(account_movements)
    quality = _compact_text_columns(quality)

    dates = sorted(pd.Timestamp(value) for value in owners["date"].dropna().unique()) if not owners.empty else []
    metadata = {
        "source_files": len(paths),
        "parsed_files": parsed_files,
        "failed_files": len(paths) - parsed_files,
        "dates": len(dates),
        "earliest_date": dates[0] if dates else None,
        "latest_date": dates[-1] if dates else None,
        "owner_rows": len(owners),
        "account_rows": len(accounts),
        "movement_rows": len(movements),
        "account_movement_rows": len(account_movements),
        "quality_warnings": int(len(quality)),
    }
    return owners, accounts, movements, account_movements, quality, metadata


def load_daily_ownership_folder(
    folder: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    folder = Path(folder)
    files = [
        path
        for pattern in ("*.xlsx", "*.xlsm")
        for path in folder.glob(pattern)
        if not path.name.startswith("~$")
    ]
    return load_daily_ownership_files(
        files,
        cache_dir=folder.parent / ".cache" / "daily_5pct",
    )


def build_account_position_pivots(
    accounts: pd.DataFrame,
    ticker: str,
    owner_normalized: str,
    dimension: str = "institution_account",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Date x Account positions and their daily change from account-level shares.

    The position table is built exclusively from ``Jumlah Saham`` (the normalized
    ``shares`` column).  It intentionally does not use combined beneficial-owner
    holdings, so an account movement cannot be mistaken for investor accumulation
    or selling.
    """
    valid_dimensions = {"institution_account", "institution", "account"}
    if dimension not in valid_dimensions:
        raise ValueError(
            f"Unsupported account dimension {dimension!r}; expected one of {sorted(valid_dimensions)}"
        )
    if accounts.empty:
        return pd.DataFrame(), pd.DataFrame()

    scoped = accounts[
        accounts["ticker"].astype(str).eq(str(ticker))
        & accounts["owner_normalized"].astype(str).eq(str(owner_normalized))
    ].copy()
    if scoped.empty:
        return pd.DataFrame(), pd.DataFrame()

    scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce")
    scoped["shares"] = pd.to_numeric(scoped["shares"], errors="coerce")
    scoped = scoped.dropna(subset=["date", "shares"])
    if scoped.empty:
        return pd.DataFrame(), pd.DataFrame()

    # A report can repeat the previous day's complete account snapshot. Prefer
    # the workbook whose source date is closest to the displayed date so the
    # same position is not counted once from each adjacent workbook.
    if "source_date" in scoped:
        source_dates = pd.to_datetime(scoped["source_date"], errors="coerce")
        source_distance = (source_dates - scoped["date"]).abs()
        nearest_distance = source_distance.groupby(scoped["date"]).transform("min")
        scoped = scoped[
            source_distance.isna()
            | nearest_distance.isna()
            | source_distance.eq(nearest_distance)
        ].copy()

    institutions = scoped["account_holder"].fillna("(MISSING ACCOUNT HOLDER)").astype(str)
    account_names = scoped["account_name"].fillna("(MISSING ACCOUNT NAME)").astype(str)
    if dimension == "institution":
        scoped["account_dimension"] = institutions
    elif dimension == "account":
        scoped["account_dimension"] = account_names
    else:
        scoped["account_dimension"] = institutions.where(
            institutions.eq(account_names),
            institutions + " · " + account_names,
        )

    grouped = (
        scoped.groupby(["date", "account_dimension"], observed=True, as_index=False)["shares"]
        .sum(min_count=1)
    )
    position = grouped.pivot_table(
        index="date",
        columns="account_dimension",
        values="shares",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    position.index = pd.DatetimeIndex(position.index, name="Date")
    position.columns.name = None
    if not position.empty:
        latest_order = position.iloc[-1].sort_values(ascending=False, kind="stable").index
        position = position.loc[:, latest_order]

    movement = position.diff()
    movement.index.name = "Date"
    movement.columns.name = None
    return position, movement


def build_account_hierarchy_pivots(
    accounts: pd.DataFrame,
    ticker: str,
    dimension: str = "institution",
    owners: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-owner account positions and daily changes for one ticker.

    Rows are report dates. Columns are a two-level hierarchy whose first level
    is the beneficial owner and whose second level is either the securities
    institution or account name. Values are always account-level ``shares``
    (normalized ``Jumlah Saham``), never combined investor ownership.
    """
    if dimension not in {"institution", "account"}:
        raise ValueError("Account dimension must be 'institution' or 'account'.")
    if accounts.empty:
        return pd.DataFrame(), pd.DataFrame()

    scoped = accounts[accounts["ticker"].astype(str).eq(str(ticker))].copy()
    if owners:
        owner_set = {str(value) for value in owners}
        scoped = scoped[scoped["owner_normalized"].astype(str).isin(owner_set)]
    if scoped.empty:
        return pd.DataFrame(), pd.DataFrame()

    scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce")
    scoped["shares"] = pd.to_numeric(scoped["shares"], errors="coerce")
    scoped = scoped.dropna(subset=["date", "shares"])
    if scoped.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Adjacent KSEI workbooks can repeat a complete prior-day snapshot. Select
    # the workbook closest to each displayed date independently for every owner.
    if "source_date" in scoped:
        source_dates = pd.to_datetime(scoped["source_date"], errors="coerce")
        source_distance = (source_dates - scoped["date"]).abs()
        nearest_distance = source_distance.groupby(
            [scoped["date"], scoped["owner_normalized"]]
        ).transform("min")
        scoped = scoped[
            source_distance.isna()
            | nearest_distance.isna()
            | source_distance.eq(nearest_distance)
        ].copy()

    owner_display = scoped["owner"].fillna(scoped["owner_normalized"]).astype(str)
    if dimension == "institution":
        account_display = scoped["account_holder"].fillna(
            "(MISSING ACCOUNT HOLDER)"
        ).astype(str)
        second_level = "Institution"
    else:
        account_display = scoped["account_name"].fillna(
            "(MISSING ACCOUNT NAME)"
        ).astype(str)
        second_level = "Account Name"
    scoped = scoped.assign(
        owner_display=owner_display,
        account_display=account_display,
    )

    grouped = (
        scoped.groupby(
            ["date", "owner_display", "account_display"],
            observed=True,
            as_index=False,
        )["shares"]
        .sum(min_count=1)
    )
    position = grouped.pivot_table(
        index="date",
        columns=["owner_display", "account_display"],
        values="shares",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    position.index = pd.DatetimeIndex(position.index, name="Date")
    position.columns = pd.MultiIndex.from_tuples(
        position.columns,
        names=["Beneficial Owner", second_level],
    )

    if not position.empty:
        latest_values = position.iloc[-1]
        owner_totals = latest_values.groupby(level=0).sum().sort_values(
            ascending=False, kind="stable"
        )
        ordered_columns: list[tuple[str, str]] = []
        for owner in owner_totals.index:
            owner_values = latest_values.xs(owner, level=0).sort_values(
                ascending=False, kind="stable"
            )
            ordered_columns.extend((owner, account) for account in owner_values.index)
        position = position.loc[:, ordered_columns]

    movement = position.diff()
    movement.index.name = "Date"
    movement.columns.names = position.columns.names
    return position, movement
