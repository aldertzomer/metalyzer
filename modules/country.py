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


# Names where pycountry's display value is undesirable or ambiguous.
DISPLAY_OVERRIDES = {
    "TR": "Turkey",
    "KR": "South Korea",
    "KP": "North Korea",
    "CG": "Republic of the Congo",
    "CD": "Democratic Republic of the Congo",
    "VI": "US Virgin Islands",
    "VG": "British Virgin Islands",
}

# The prefix "original" is not evidence for a geographic origin field.
GEOGRAPHIC_FIELD_PRIORITY = (
    "geo_loc_name", "geographic_location", "geographic_location_name",
    "country", "country_of_origin", "geographic_origin", "location",
)

AMBIGUOUS_COUNTRIES = {"virgin islands"}

SPLIT_RE = re.compile(r"[;|/]+")

COUNTRY_HINTS = ("country", "location", "geographic", "origin")


def _ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))



ADDITIONAL_ALIASES = {
    # A bare Korea denotes South Korea; explicit North Korea always wins.
    "korea": "South Korea",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "korea republic of": "South Korea",
    "north korea": "North Korea",
    "dprk": "North Korea",
    "democratic people s republic of korea": "North Korea",
    "democratic peoples republic of korea": "North Korea",
    "korea democratic people s republic of": "North Korea",
    "korea democratic peoples republic of": "North Korea",
    # Keep the ASCII label even when supplied as Türkiye.
    "turkey": "Turkey",
    "turkiye": "Turkey",
    "republic of turkey": "Turkey",
    "republic of turkiye": "Turkey",
    # Resolve the entire country name before splitting on the comma.
    "congo": "Republic of the Congo",
    "republic of congo": "Republic of the Congo",
    "republic of the congo": "Republic of the Congo",
    "congo brazzaville": "Republic of the Congo",
    "democratic republic of congo": "Democratic Republic of the Congo",
    "democratic republic of the congo": "Democratic Republic of the Congo",
    "congo democratic republic of": "Democratic Republic of the Congo",
    "congo democratic republic of the": "Democratic Republic of the Congo",
    "congo the democratic republic of the": "Democratic Republic of the Congo",
    "congo kinshasa": "Democratic Republic of the Congo",
    "congo dem rep": "Democratic Republic of the Congo",
    "dr congo": "Democratic Republic of the Congo",
    "d r congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    # U.S. Virgin Islands and British Virgin Islands are separate territories.
    "virgin islands u s": "US Virgin Islands",
    "virgin islands us": "US Virgin Islands",
    "u s virgin islands": "US Virgin Islands",
    "us virgin islands": "US Virgin Islands",
    "united states virgin islands": "US Virgin Islands",
    "usvi": "US Virgin Islands",
    "virgin islands british": "British Virgin Islands",
    "british virgin islands": "British Virgin Islands",
    "bvi": "British Virgin Islands",
}


def _key(s: str) -> str:
    """Case-, punctuation-, and accent-insensitive country matching key."""
    return re.sub(r"[^a-z0-9]+", " ", _ascii_fold(s).lower()).strip()


# Apply the same normalization to the historical alias list.
ALIASES = {_key(name): value for name, value in ALIASES.items()}
ALIASES.update({_key(name): value for name, value in ADDITIONAL_ALIASES.items()})


def _resolve_country(token: str, *, fuzzy: bool = False) -> str | None:
    key = _key(token)
    if key in MISSING_COUNTRY or key in AMBIGUOUS_COUNTRIES:
        return None
    if key in ALIASES:
        return ALIASES[key]
    try:
        matched = pycountry.countries.lookup(token)
    except LookupError:
        if not fuzzy:
            return None
        try:
            matches = pycountry.countries.search_fuzzy(token)
        except LookupError:
            return None
        if not matches:
            return None
        matched = matches[0]
    return DISPLAY_OVERRIDES.get(matched.alpha_2, matched.name)


def normalize_country(raw: str) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s or s.lower() in MISSING_COUNTRY:
        return "unknown"

    for part in SPLIT_RE.split(s):
        part = part.strip()
        if not part:
            continue

        # First resolve full names (Congo, Virgin Islands, Korea, etc.).
        country = _resolve_country(part)
        if country:
            return country

        # Only after full-name resolution remove region/site tails.
        token = part.split(":", 1)[0].split(",", 1)[0].strip()
        # Short fragments can give very spurious fuzzy matches ("N" -> UK).
        fuzzy = len(_key(token)) >= 4 and _key(token) not in AMBIGUOUS_COUNTRIES
        country = _resolve_country(token, fuzzy=fuzzy)
        if country:
            return country

    return "unknown"


# Use only dedicated geographic fields. Substring matching on "origin"
# accidentally matches every "original_attributes.*" BioSample column.
RE_KV_COUNTRY = re.compile(
    r"(?:^|;)\\s*(?:country|location|geo loc name|geographic location)\\s*:\\s*([^;]+)",
    re.IGNORECASE,
)


def _field_rank(name: str) -> int | None:
    normalized = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    parts = normalized.split(".")
    leaf = parts[-1]
    if leaf not in GEOGRAPHIC_FIELD_PRIORITY:
        return None
    return (GEOGRAPHIC_FIELD_PRIORITY.index(leaf)
            + (len(parts) - 1) * len(GEOGRAPHIC_FIELD_PRIORITY))


def extract_country_from_row(row: pd.Series, record: str) -> str:
    fields = sorted(
        (rank, i, col)
        for i, col in enumerate(row.index)
        if (rank := _field_rank(col)) is not None
    )
    for _, _, column in fields:
        country = normalize_country(row[column])
        if country != "unknown":
            return country

    for match in RE_KV_COUNTRY.finditer(record):
        country = normalize_country(match.group(1))
        if country != "unknown":
            return country
    return "unknown"


def run(batch: MetadataBatch) -> FieldResult:
    """Return standardized country names, or 'unknown' when unresolved."""
    return FieldResult(pd.Series(
        [extract_country_from_row(row, batch.records.loc[key])
         for key, row in batch.metadata.iterrows()],
        index=batch.metadata.index, name="country", dtype=object,
    ))
