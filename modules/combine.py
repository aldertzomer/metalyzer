"""Validate and combine independent results by row key, then serialize TSV."""
from pathlib import Path

import pandas as pd

from .contracts import (
    FieldResult, MetadataBatch, SourceResult, SourceVocabulary, empty_source_table,
)


def unresolved(batch: MetadataBatch, sources: SourceVocabulary,
               *results: SourceResult) -> MetadataBatch:
    """Select rows omitted by every completed source stage, in input order."""
    covered = set()
    for result in results:
        result.validate(batch, sources)
        keys = set(result.table.index)
        if covered & keys:
            raise ValueError("Source stages returned overlapping row keys")
        covered.update(keys)
    return batch.subset([key for key in batch.metadata.index if key not in covered])


def sources(batch: MetadataBatch, vocabulary: SourceVocabulary,
            *results: SourceResult) -> SourceResult:
    """Require exactly one source decision per row; reject overlaps and gaps."""
    if len(unresolved(batch, vocabulary, *results)):
        raise ValueError("Source results do not cover every input row")
    tables = [result.table for result in results if len(result.table)]
    table = (pd.concat(tables).reindex(batch.metadata.index) if tables
             else empty_source_table(vocabulary, batch.metadata.index))
    output = SourceResult(table)
    output.validate(batch, vocabulary)
    return output


def run(batch: MetadataBatch, vocabulary: SourceVocabulary,
        source_result: SourceResult, *fields: FieldResult) -> pd.DataFrame:
    """Return ID, source columns, then fields in argument order, in input order."""
    source_result = sources(batch, vocabulary, source_result)
    if vocabulary.id_col not in batch.metadata:
        raise ValueError(f"ID column {vocabulary.id_col!r} is missing from metadata")
    columns = {vocabulary.id_col, *vocabulary.columns}
    for field in fields:
        field.validate(batch)
        if field.values.name in columns:
            raise ValueError(f"Duplicate output column: {field.values.name}")
        columns.add(field.values.name)
    return pd.concat([
        batch.metadata[[vocabulary.id_col]].astype(str), source_result.table,
        *(field.values.reindex(batch.metadata.index) for field in fields),
    ], axis=1)


def write(table: pd.DataFrame, path: str | Path) -> None:
    """Write the assembled table as TSV, preserving NaN scores as literal NA."""
    table.to_csv(path, sep="\t", index=False, na_rep="NA")


def report(result: SourceResult, method: str) -> None:
    """Print source provenance and unknown counts from completed results."""
    table = result.table
    classified = table[table.source_method == method]
    print("Source classification:\n"
          f"  taxonomy host_tax_id: {(table.source_method == 'host_tax_id').sum()}\n"
          f"  {method.upper()}:                  {len(classified)}\n"
          f"  {method.upper()} -> unknown:        {(classified.best_hit == 'unknown').sum()}", flush=True)
    if method == "llm":
        print(f"  invalid LLM outputs:  {classified.source_evidence.str.startswith('invalid_llm_output=').sum()}", flush=True)
