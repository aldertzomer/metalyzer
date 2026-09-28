"""Country normalization. run(batch) returns FieldResult named country."""
import re
import unicodedata

import pandas as pd
import pycountry

from .contracts import FieldResult, MetadataBatch


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


# only accept explicit "country-like" fragments from the record
RE_KV_COUNTRY = re.compile(
    r"\b(?:country|location|geo loc name|geographic location)\s*:\s*([^;]+)",
    re.IGNORECASE,
)


def extract_country_from_row(row: pd.Series, record: str) -> str:
    # 1) try likely columns first (preferred)
    for col in row.index:
        cl = str(col).lower()
        if any(h in cl for h in COUNTRY_HINTS):
            c = normalize_country(row[col])
            if c != "unknown":
                return c

    # 2) fallback: only if record explicitly contains a country/location field.
    m = RE_KV_COUNTRY.search(record)
    if m:
        c = normalize_country(m.group(1))
        if c != "unknown":
            return c

    # 3) otherwise: unknown (DO NOT fuzzy-match entire record)
    return "unknown"


def run(batch: MetadataBatch) -> FieldResult:
    """Return standardized country names, or 'unknown' when unresolved."""
    return FieldResult(pd.Series(
        [extract_country_from_row(row, batch.records.loc[key])
         for key, row in batch.metadata.iterrows()],
        index=batch.metadata.index, name="country", dtype=object,
    ))
