"""TSV loading and conversion to the standard in-memory batch contract."""
from pathlib import Path
import re

import pandas as pd

from .contracts import MetadataBatch, SourceVocabulary
from .records import build_record


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a TSV while discarding invalid UTF-8 bytes from legacy exports."""
    with open(path, encoding="utf-8", errors="ignore", newline="") as handle:
        return pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, **kwargs)


def prepare_batch(metadata: pd.DataFrame, *, max_value_chars: int = 300,
                  max_record_chars: int = 2000) -> MetadataBatch:
    """Assign positional keys once, preserving external IDs and input order."""
    if max_value_chars < 1 or max_record_chars < 1:
        raise ValueError("Record truncation limits must be positive")
    frame = metadata.reset_index(drop=True)
    records = pd.Series(
        [build_record(row, max_value_chars, max_record_chars) for _, row in frame.iterrows()],
        index=frame.index, name="record", dtype=object,
    )
    return MetadataBatch(frame, records)


def parse_taxonomy_anchors(value: str) -> tuple[int, ...]:
    """Parse optional comma-separated NCBI IDs without changing source text."""
    if not value.strip():
        return ()
    parts = [part.strip() for part in value.split(",")]
    if any(not re.fullmatch(r"[0-9]+", part) or int(part) < 1 for part in parts):
        raise ValueError(f"Invalid taxonomy_anchors value: {value!r}")
    return tuple(int(part) for part in parts)


def load(metadata_path: str | Path, sources_path: str | Path, id_col: str, *,
         limit: int | None = None, max_value_chars: int = 300,
         max_record_chars: int = 2000) -> tuple[MetadataBatch, SourceVocabulary]:
    """Read string-valued TSVs without NA coercion; limit applies at read time."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    metadata = read_tsv(metadata_path, nrows=limit)
    if id_col not in metadata:
        raise ValueError(f"ID column {id_col!r} is missing from metadata")
    labels = read_tsv(sources_path)
    column = "source" if "source" in labels else labels.columns[0]
    has_anchors = "taxonomy_anchors" in labels
    rows = [(str(row[column]).strip(), parse_taxonomy_anchors(str(row["taxonomy_anchors"]))
             if has_anchors else ()) for _, row in labels.iterrows() if str(row[column]).strip()]
    sources = SourceVocabulary(tuple(label for label, _ in rows), id_col,
                               tuple(anchors for _, anchors in rows))
    return prepare_batch(metadata, max_value_chars=max_value_chars,
                         max_record_chars=max_record_chars), sources
