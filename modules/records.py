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
                 excluded_fields=()) -> str:
    parts = []
    for col, val in row.items():
        if col in excluded_fields or is_empty_like(val):
            continue
        v = " ".join(unescape(str(val)).split())
        if len(v) > max_value_chars:
            v = v[:max_value_chars] + "…"
        field = str(col).replace("_", " ")
        parts.append(f"{field}: {v}")
    rec = "; ".join(parts)
    if len(rec) > max_record_chars:
        rec = rec[:max_record_chars] + "…"
    return rec if rec.strip() else "metadata: (empty)"
