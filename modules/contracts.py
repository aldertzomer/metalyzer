"""Shared, model-independent input and output contracts for pipeline stages."""
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


SOURCE_COLUMNS = ("best_hit", "source_method", "source_evidence")
VERIFICATION_COLUMN = "source_verification_score"
RESERVED_COLUMNS = (*SOURCE_COLUMNS, VERIFICATION_COLUMN, "year", "country")


def canonical_source_names(source_labels, id_col):
    """Validate descriptions and return their ordered canonical source names."""
    names = [label.split("(", 1)[0].strip() for label in source_labels]
    if not names or any(not name for name in names):
        raise ValueError("Each source must have a nonempty name before any parentheses")
    if len(set(names)) != len(names):
        raise ValueError("Source names before parentheses must be unique")
    if set(names) & {id_col, *RESERVED_COLUMNS}:
        raise ValueError("Source names must not conflict with output metadata columns")
    if id_col in RESERVED_COLUMNS:
        raise ValueError("ID column must not conflict with output metadata columns")
    return names


@dataclass(frozen=True)
class SourceVocabulary:
    """Ordered full label descriptions and ID column; names strip parenthetical hints."""
    labels: tuple[str, ...]
    id_col: str

    def __post_init__(self):
        object.__setattr__(self, "labels", tuple(self.labels))
        canonical_source_names(self.labels, self.id_col)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(canonical_source_names(self.labels, self.id_col))

    @property
    def columns(self) -> list[str]:
        return [*self.names, *SOURCE_COLUMNS]


@dataclass(frozen=True)
class MetadataBatch:
    """Read-only metadata and rendered text, keyed by unique integer row positions.

    External IDs may repeat. Subsets keep their original row keys. Do not mutate
    the contained frame/series; frozen dataclasses do not freeze pandas objects.
    """
    metadata: pd.DataFrame
    records: pd.Series

    def __post_init__(self):
        index = self.metadata.index
        if not index.is_unique or any(not isinstance(key, int) or key < 0 for key in index):
            raise ValueError("Metadata row keys must be unique nonnegative integers")
        if not self.metadata.columns.is_unique:
            raise ValueError("Metadata columns must be unique")
        if not index.equals(self.records.index):
            raise ValueError("Metadata and records must have identical ordered row keys")
        if not all(isinstance(value, str) for value in self.records):
            raise ValueError("Rendered records must be strings")

    def subset(self, row_keys) -> "MetadataBatch":
        """Select row keys in the requested order, preserving identity."""
        keys = list(row_keys)
        return MetadataBatch(self.metadata.loc[keys], self.records.loc[keys])

    def __len__(self):
        return len(self.metadata)


@dataclass(frozen=True)
class SourceResult:
    """Source table, indexed by input row keys; omitted rows mean unresolved.

    Columns: vocabulary names (float scores or NaN), best_hit (canonical label
    or 'unknown'), source_method (nonempty string), source_evidence (string).
    An explicit 'unknown' is a completed prediction, not an unresolved row.
    """
    table: pd.DataFrame

    def validate(self, batch: MetadataBatch, sources: SourceVocabulary) -> None:
        table = self.table
        if table.columns.tolist() != sources.columns:
            raise ValueError(f"Source result columns must be {sources.columns}")
        if not table.index.is_unique or not table.index.isin(batch.metadata.index).all():
            raise ValueError("Source results contain duplicate or foreign row keys")
        if not table.best_hit.isin([*sources.names, "unknown"]).all():
            raise ValueError("Source results contain labels outside the vocabulary")
        if not all(isinstance(v, str) and v.strip() for v in table.source_method):
            raise ValueError("Every source result needs a nonempty source_method")
        if not all(isinstance(v, str) for v in table.source_evidence):
            raise ValueError("Source evidence must be a string (empty is allowed)")
        scores = table[list(sources.names)]
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in scores.dtypes):
            raise ValueError("Source scores must be numeric or NaN")
        if not ((scores.isna()) | ((scores >= 0) & (scores <= 1))).all().all():
            raise ValueError("Source scores must be between 0 and 1 or NaN")
        if ((scores.isna().any(axis=1)) & ~(scores.isna().all(axis=1))).any():
            raise ValueError("Supply either all candidate scores or all NaN per row")


@dataclass(frozen=True)
class FieldResult:
    """One named string field, with exactly the row keys supplied to the stage."""
    values: pd.Series

    def validate(self, batch: MetadataBatch) -> None:
        if not isinstance(self.values.name, str) or not self.values.name:
            raise ValueError("A field result needs a nonempty column name")
        if (not self.values.index.is_unique
                or len(self.values) != len(batch)
                or not self.values.index.isin(batch.metadata.index).all()):
            raise ValueError("Field results must cover every input row exactly once")
        if not all(isinstance(value, str) for value in self.values):
            raise ValueError("Field values must be strings; use the documented missing value")


@dataclass(frozen=True)
class VerificationResult:
    """Binary entailment scores for assigned sources; NaN means unscored."""
    values: pd.Series

    def validate(self, batch: MetadataBatch) -> None:
        values = self.values
        if (values.name != VERIFICATION_COLUMN or not values.index.is_unique
                or not values.index.equals(batch.metadata.index)):
            raise ValueError("Verification scores must cover the ordered input rows")
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ValueError("Verification scores must be numeric or NaN")
        if not (values.isna() | ((values >= 0) & (values <= 1))).all():
            raise ValueError("Verification scores must be between 0 and 1 or NaN")


class SourceModule(Protocol):
    """A configured stage's run method may implement this callable interface."""
    def __call__(self, batch: MetadataBatch, sources: SourceVocabulary) -> SourceResult: ...


class FieldModule(Protocol):
    def __call__(self, batch: MetadataBatch) -> FieldResult: ...


def empty_source_table(sources: SourceVocabulary, row_keys=()) -> pd.DataFrame:
    """Create an output table with stable dtypes, including for an empty batch."""
    table = pd.DataFrame(float("nan"), index=pd.Index(list(row_keys), dtype="int64"), columns=sources.names)
    for column in SOURCE_COLUMNS:
        table[column] = pd.Series("", index=table.index, dtype=object)
    return table
