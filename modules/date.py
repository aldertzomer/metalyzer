"""Deterministic year extraction. run(batch) returns FieldResult named year."""
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .contracts import FieldResult, MetadataBatch
from .records import NA_LIKE


YEAR4 = re.compile(r"\b(19\d{2}|20\d{2})\b")
# 2019, 2016-04, 2017-10, 2007-11, 1905
YMD_PREFIX = re.compile(r"^\s*(19\d{2}|20\d{2})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?\s*$")
# 31-12-19 or 15-06-18 (DD-MM-YY)
DMY2 = re.compile(r"^\s*(\d{1,2})-(\d{1,2})-(\d{2})\s*$")

DATE_COL_ORDER = (
    "collection_date",
    "collection_date_start",
    "collection_date_end",
)


def yy_to_yyyy(yy: int, max_year: int = 2026) -> int:
    # Pivot derived from max_year: 00..27 => 20xx; 28..99 => 19xx
    pivot = (max_year % 100) + 1  # 27 for 2026
    return 2000 + yy if yy <= pivot else 1900 + yy


def year_from_value(v: str, min_year: int = 1905, max_year: int = 2026) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in NA_LIKE:
        return None

    # 1) Exact YYYY or YYYY-MM or YYYY-MM-DD
    m = YMD_PREFIX.match(s)
    if m:
        y = int(m.group(1))
        if min_year <= y <= max_year:
            return y

    # 2) Any embedded 4-digit year
    m = YEAR4.search(s)
    if m:
        y = int(m.group(1))
        if min_year <= y <= max_year:
            return y

    # 3) DMY with 2-digit year: use ONLY the last group (YY), never day/month
    m = DMY2.match(s)
    if m:
        yy = int(m.group(3))
        y = yy_to_yyyy(yy, max_year=max_year)
        if min_year <= y <= max_year:
            return y

    return None


def extract_year_from_row(row, min_year: int = 1905, max_year: int = 2026) -> Optional[int]:
    # Prefer the known columns in order
    for col in DATE_COL_ORDER:
        if col in row.index:
            y = year_from_value(row[col], min_year=min_year, max_year=max_year)
            if y is not None:
                return y

    # Fallback: scan other columns that look date-ish, but STILL only parse with year_from_value
    for col in row.index:
        cl = str(col).lower()
        if "date" in cl or "year" in cl:
            y = year_from_value(row[col], min_year=min_year, max_year=max_year)
            if y is not None:
                return y

    return None


@dataclass(frozen=True)
class DateConfig:
    min_year: int = 1905
    max_year: int = 2026

    def __post_init__(self):
        if self.min_year > self.max_year:
            raise ValueError("min_year must not exceed max_year")


def run(batch: MetadataBatch, config: DateConfig = DateConfig()) -> FieldResult:
    """Return four-digit year strings, or '' when no valid year is found."""
    years = [extract_year_from_row(row, config.min_year, config.max_year)
             for _, row in batch.metadata.iterrows()]
    return FieldResult(pd.Series(
        ["" if year is None else str(year) for year in years],
        index=batch.metadata.index, name="year", dtype=object,
    ))
