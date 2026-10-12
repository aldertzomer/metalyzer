"""Render metadata into the shared natural-language representation."""
import math
from html import unescape

import pandas as pd


NA_LIKE = {"", "na", "n/a", "nan", "none", "null", "missing", "not provided", "not collected", "not applicable"}


def is_empty_like(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    if s == "":
        return True
    return s.lower() in NA_LIKE


def build_record(row: pd.Series, max_value_chars: int = 300, max_record_chars: int = 2000,
                 excluded_fields=(), exclude_noisy_fields: bool = True, *, audit=None) -> str:
    from .noisy_fields import select_fields
    items = [(col, val) for col, val in row.items() if col not in excluded_fields and not is_empty_like(val)]
    kept, omitted = select_fields(items) if exclude_noisy_fields else (items, [])
    record, full_length = _render(kept, max_value_chars, max_record_chars)
    if audit is not None:
        before, before_full = _render(items, max_value_chars, max_record_chars)
        audit.update(fields_considered=len(row), nonempty_eligible_fields=len(items),
                     administrative_technical=sum(reason == 'administrative_technical' for reason, _ in omitted),
                     redundant_original=sum(reason == 'redundant_original' for reason, _ in omitted),
                     omissions=tuple(omitted), before_length=len(before), after_length=len(record),
                     before_full_length=before_full, after_full_length=full_length,
                     before_truncated=before_full > max_record_chars,
                     after_truncated=full_length > max_record_chars)
    return record


def _render(items, max_value_chars, max_record_chars):
    """Legacy rendering byte-for-byte, after optional field selection."""
    parts = []
    for col, val in items:
        v = " ".join(unescape(str(val)).split())
        if len(v) > max_value_chars:
            v = v[:max_value_chars] + "…"
        field = str(col).replace("_", " ")
        parts.append(f"{field}: {v}")
    rec = "; ".join(parts)
    full_length = len(rec)
    if len(rec) > max_record_chars:
        rec = rec[:max_record_chars] + "…"
    return (rec if rec.strip() else "metadata: (empty)"), full_length
