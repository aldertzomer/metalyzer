#!/usr/bin/env python3
"""
Classify metadata rows with:
  - Source: zero-shot classification (MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
  - Year: deterministic extraction -> 4-digit year (1905–2026)
  - Country: deterministic normalization (handles "USA:WY", "U.S.A;USA", "Canada: Calgary, Alberta", etc.)

Input:
  --metadata  (TSV)
  --sources   (TSV with column 'source' or first column = labels)
Output (TSV):
  id <tab> <one column per source label> <tab> year <tab> country

Notes:
  - Set TOKENIZERS_PARALLELISM=true for speed.
  - The model weights are cached by HF; the progress bar is just loading into memory.
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from typing import Optional

import pandas as pd
import pycountry
from transformers import pipeline


# -------------------------
# Generic NA-like handling
# -------------------------
NA_LIKE = {"", "na", "n/a", "nan", "none", "null", "missing", "not provided", "not collected", "not applicable", "NA"}


def is_empty_like(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    if s == "":
        return True
    return s.lower() in NA_LIKE


def build_record(row: pd.Series, max_value_chars: int = 300, max_record_chars: int = 2000) -> str:
    parts = []
    for col, val in row.items():
        if is_empty_like(val):
            continue
        v = str(val).strip().replace("\n", " ").replace("\r", " ")
        if len(v) > max_value_chars:
            v = v[:max_value_chars] + "…"
        parts.append(f'{col}="{v}"')
    rec = "; ".join(parts)
    if len(rec) > max_record_chars:
        rec = rec[:max_record_chars] + "…"
    return rec if rec.strip() else 'metadata="(empty)"'


# -------------------------
# Year extraction
# -------------------------

import re
from typing import Optional

NA_LIKE = {"", "na", "n/a", "nan", "none", "null", "missing", "not provided", "not collected", "not applicable"}

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



# -------------------------
# Country normalization
# -------------------------
MISSING_COUNTRY = {
    "", "na", "n/a", "nan", "none", "null", "missing", "not collected", "not provided", "unknown"
}

ALIASES = {
    # USA
    "usa": "United States",
    "u s a": "United States",
    "u.s.a": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "u s": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "america": "United States",

    # UK
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",

    # Korea (policy choice)
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "korea": "South Korea",  # change to None if you prefer ambiguous->unknown

    # Vietnam
    "viet nam": "Vietnam",
    "vietnam": "Vietnam",

    # Czech Republic
    "czech republic": "Czechia",

    # Tanzania
    "united republic of tanzania": "Tanzania",
}

SPLIT_RE = re.compile(r"[;|/]+")

COUNTRY_HINTS = ("country", "location", "geographic", "origin")


def _ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize_country(raw: str) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"

    for part in SPLIT_RE.split(s):
        part = part.strip()
        if not part:
            continue

        # remove region/site tails
        token = part.split(":", 1)[0].strip()
        token = token.split(",", 1)[0].strip()

        token_nf = _ascii_fold(token).lower()
        token_nf = re.sub(r"[\.\(\)\[\]]+", " ", token_nf)
        token_nf = re.sub(r"[_\-]+", " ", token_nf)
        token_nf = re.sub(r"\s+", " ", token_nf).strip()

        if token_nf in MISSING_COUNTRY:
            continue

        # alias mapping
        if token_nf in ALIASES:
            mapped = ALIASES[token_nf]
            if mapped is None:
                continue
            return mapped

        # direct match
        c = pycountry.countries.get(name=token)
        if c:
            return c.name

        # fuzzy match
        try:
            matches = pycountry.countries.search_fuzzy(token)
            if matches:
                return matches[0].name
        except LookupError:
            pass

    return "unknown"


#def extract_country_from_row(row: pd.Series, record: str) -> str:
#    # try likely columns first
#    for col in row.index:
#        cl = str(col).lower()
#        if any(h in cl for h in COUNTRY_HINTS):
#            c = normalize_country(row[col])
#            if c != "unknown":
#                return c
#    # fallback: scan record for "country=" or "location=" style fragments
#    # (cheap heuristic: try to normalize the whole record; normalize_country will split and fail fast)
#    return normalize_country(record)
#
## only accept explicit "country-like" fragments from the record
# RE_KV_COUNTRY = re.compile(r'\b(country|location|geo_loc_name|geographic_location)\s*=\s*"([^"]+)"', re.IGNORECASE)

# only accept explicit "country-like" fragments from the record
RE_KV_COUNTRY = re.compile(r'\b(country|location|geo_loc_name|geographic_location)\s*=\s*"([^"]+)"', re.IGNORECASE)


def extract_country_from_row(row: pd.Series, record: str) -> str:
    # 1) try likely columns first (preferred)
    for col in row.index:
        cl = str(col).lower()
        if any(h in cl for h in COUNTRY_HINTS):
            c = normalize_country(row[col])
            if c != "unknown":
                return c

    # 2) fallback: only if record explicitly contains country/location="..."
    m = RE_KV_COUNTRY.search(record)
    if m:
        c = normalize_country(m.group(2))
        if c != "unknown":
            return c

    # 3) otherwise: unknown (DO NOT fuzzy-match entire record)
    return "unknown"



# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Input metadata TSV")
    ap.add_argument("--sources", required=True, help="Sources TSV (labels)")
    ap.add_argument("--out", required=True, help="Output TSV")
    ap.add_argument("--id-col", required=True, help="ID column name in metadata")

    ap.add_argument("--device", type=int, default=0, help="0 for GPU, -1 for CPU")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-value-chars", type=int, default=300)
    ap.add_argument("--max-record-chars", type=int, default=2000)

    args = ap.parse_args()

    df = pd.read_csv(args.metadata, sep="\t", dtype=str, keep_default_na=False)
    src_df = pd.read_csv(args.sources, sep="\t", dtype=str, keep_default_na=False)

    # label column: 'source' if present else first column
    label_col = "source" if "source" in src_df.columns else src_df.columns[0]
    source_labels = [s.strip() for s in src_df[label_col].tolist() if str(s).strip() != ""]

    # prepare records + extracted fields
    records = []
    years = []
    countries = []

    for _, row in df.iterrows():
        rec = build_record(row, max_value_chars=args.max_value_chars, max_record_chars=args.max_record_chars)
        records.append(rec)
        y = extract_year_from_row(row, min_year=1905, max_year=2026)
        years.append("" if y is None else str(y))
        countries.append(extract_country_from_row(row, rec))

    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        device=args.device,
    )

    # Source scores (wide)
    score_rows = []
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]

        source_results = classifier(
            batch,
            candidate_labels=source_labels,
            hypothesis_template="The biological host or environmental source of this sample is {}.",
            multi_label=False,  # nonindependent scores per label
        )

        # print once per batch, per your preference
        print(f"Batch {i // args.batch_size + 1}: source classification done", flush=True)

        if isinstance(source_results, dict):
            source_results = [source_results]

        for r in source_results:
            # HF returns labels sorted; we want fixed column order
            m = {lab: 0.0 for lab in source_labels}
            for lab, sc in zip(r["labels"], r["scores"]):
                m[lab] = float(sc)
            score_rows.append(m)

    scores_df = pd.DataFrame(score_rows)
    scores_df = scores_df[source_labels]  # enforce order

    out_df = pd.concat(
        [
            df[[args.id_col]].astype(str),
            scores_df,
            pd.Series(years, name="year"),
            pd.Series(countries, name="country"),
        ],
        axis=1,
    )

    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out} (n={len(out_df)})", flush=True)


if __name__ == "__main__":
    main()
