# Developing a Metalyzer module

Start here and in [`contracts.py`](contracts.py). A stage needs to understand
its input, its output, and its own configuration. It must not import
`metalyzer`, parse command-line arguments, read unrelated files, or call another
classifier. The CLI owns stage order and activation. There is no plugin registry
or automatic discovery: a new stage is wired explicitly into `metalyzer.py`.

## Pipeline and activation

```text
input.load -> MetadataBatch + SourceVocabulary
                  |
                  +-> deterministic_source.run -> resolved SourceResult
                  |           |
                  |     combine.unresolved -> remaining MetadataBatch
                  |           |
                  |     nli.run OR llm.run -> fallback SourceResult
                  |           |
                  |     combine.sources -> complete SourceResult
                  |
                  +-> date.run -> FieldResult("year")
                  +-> country.run -> FieldResult("country")
                              |
                         combine.run -> output DataFrame
                         combine.write -> TSV
```

| Module | Public entry point | Input/configuration | Output and activation |
|---|---|---|---|
| `input` | `load(...)`, `prepare_batch(...)` | TSV paths, ID column, optional row/truncation limits; or a DataFrame | Batch and vocabulary; always used by CLI |
| `records` | `build_record(row, ...)` | One metadata Series, character limits | String used by both text classifiers and country fallback; called by input preparation |
| `date` | `run(batch, config=DateConfig())` | Batch; year bounds default to 1905–2026 | `FieldResult` named `year`; enabled unless `--skip-date` |
| `country` | `run(batch)` | Batch | `FieldResult` named `country`; enabled unless `--skip-country` |
| `deterministic_source` | `run(batch, sources, taxonomy=None)` | Batch, vocabulary, loaded `NCBITaxonomy` or `None` | Partial `SourceResult`; lookup active with valid `--taxonomy-dir` |
| `nli` | `run(batch, sources, config=NLIConfig())` | Batch, vocabulary; device, batch size, score cutoff | Complete result for the supplied batch; `--method nli` (default) |
| `llm` | `run(batch, sources, config=LLMConfig())` | Batch, vocabulary; model, revision, device, batch size, generation limit | Complete result for the supplied batch; `--method llm` |
| `combine` | `unresolved`, `sources`, `run`, `write` | Batch, vocabulary, stage results; path for writing | Validated subsets, source result, assembled DataFrame, TSV |

Taxonomy resolution always precedes text classification. Only unresolved rows
reach the selected text stage. Empty batches and batches fully resolved by
taxonomy never load either model. The two classifiers are mutually exclusive.
`--download-taxonomy DIR` explicitly downloads and validates the dump, then exits.
Loading modules does not download files or load model weights.

## Input contract

`MetadataBatch(metadata: pandas.DataFrame, records: pandas.Series)` contains:

- `metadata`: columns retain their original names. TSV input values are strings;
  empty strings and literal `NA` remain unchanged. Parsers also tolerate common
  missing/numeric values in programmatically constructed batches.
- `records`: one natural-language string per row. Input preparation renders
  fields as `host: Ovis aries; country: USA`, skips missing values, replaces
  underscores in column names, and truncates values/records at 300/2000 characters
  by default. The truncation marker `…` is appended after the character limit.
  An empty row becomes `metadata: (empty)`.
- Both objects have the same ordered, unique, nonnegative integer index. These
  are **internal row keys**, not sample identifiers. `prepare_batch` assigns
  `0..N-1` once; `batch.subset(keys)` preserves the keys. For example, a text
  classifier may receive rows `[1, 4, 9]` and must return those same keys.

Input objects are read-only by convention: do not mutate their DataFrames or
Series. The frozen dataclasses prevent attribute reassignment, not pandas edits.
Do not reset a subset's index, align by sample ID, or assume IDs are unique.
`--limit` is applied while reading the metadata TSV to avoid loading unused rows.

`SourceVocabulary(labels: tuple[str, ...], id_col: str)` contains full source
descriptions in source-file order. `sources.names` strips the first `(` and all
following text, preserving multi-word names. Names must be nonempty, unique,
and must not collide with the ID or output metadata columns. `sources.columns`
is the exact ordered source-result schema. `unknown` is always a valid predicted
label; it gets a score column only if explicitly included in the vocabulary.

## Source output contract

Return `SourceResult(table: pandas.DataFrame)` with this column order:

| Columns | Values |
|---|---|
| One column per `sources.names`, in order | Numeric score in `[0, 1]`, or float NaN for unavailable scores |
| `best_hit` | Exact canonical label, or `unknown` |
| `source_method` | Nonempty provenance string, e.g. `host_tax_id`, `nli`, `llm`, or your new method |
| `source_evidence` | String explaining the decision, or `""` |

Each returned row must use a unique key from the input batch. A partial stage
omits unresolved rows entirely. Returning `best_hit="unknown"` is a **completed
prediction**, so it does not trigger fallback. NLI and LLM return one row per
supplied input row, including unknown calls. For each row, provide all candidate
scores or all NaN; do not mix missing and present scores. Scores need not sum to
one (the combiner also supports future multilabel scoring methods).

Use `empty_source_table(sources, row_keys)` to construct a correctly ordered table
with float NaN scores and empty text fields. Fill `best_hit` and `source_method`
before returning any row. `SourceResult.validate(batch, sources)` checks schema,
keys, labels, provenance, score types/ranges, and missing-score consistency.
The combiner calls this validation even if your module already did.

Existing stage semantics:

- Taxonomy uses only `host_tax_id`, never pathogen/sample `tax_id`. It follows
  merged IDs, translates IDs through scientific names, walks host lineages, and
  respects the selected vocabulary. Unresolved or unavailable labels are omitted.
  Scores are NaN. Evidence is `host_tax_id=9940; Ovis aries`.
- `NCBITaxonomy.normalize_taxid(value)` returns a current valid integer ID or
  `None`; `scientific_name(value)` returns its scientific name or `""`.
  These are available directly in `deterministic_source` for taxid translation
  without classifying a host. `classify_host_taxid` returns a
  `TaxonomySourceResult(source, taxid, scientific_name)` or `None`.
- NLI uses full descriptions as candidate labels. Ties use source-file order;
  a top score below `NLIConfig.min_score` becomes unknown with scores retained.
  `source_evidence` is empty. CPU uses float32; GPU uses checkpoint precision.
- LLM uses full descriptions in its prompt and returns canonical labels. Scores
  are NaN. Invalid responses become unknown with sanitized, truncated
  `invalid_llm_output=...` evidence. `LLMConfig` has no score threshold.

## Field output and combination contracts

Return `FieldResult(values: pandas.Series)`. Its name is the output column name;
its string values cover every row supplied to the stage exactly once. Row order
may differ: combination aligns by key and restores input order. Existing missing
values are `""` for `year` and `"unknown"` for `country`. Date extraction first
checks collection date/start/end, then other date/year fields. Country parsing
uses likely metadata columns and explicit country/location fragments.

`combine.unresolved(batch, sources, *results)` returns a batch containing keys
omitted from all supplied results. `combine.sources(...)` requires exactly one
source decision per input row. Both reject overlapping source results; stages
must implement precedence by receiving only the unresolved subset. Foreign keys,
duplicate keys, invalid labels, missing decisions, and malformed schemas raise
`ValueError` instead of silently joining incorrect rows.

`combine.run(batch, sources, source_result, *fields)` returns a DataFrame ordered
as ID, source columns, then fields in argument order. It rejects incomplete
fields and duplicate output column names. `combine.write(table, path)` writes
tab-separated output without the internal index, using literal `NA` for NaN
scores. Disabled field stages contribute no column. With default activation,
the output schema and classification behavior match the original CLI.

## Develop and test a field module in isolation

```python
import pandas as pd
from modules.contracts import FieldResult, MetadataBatch


def run(batch: MetadataBatch) -> FieldResult:
    # Example new module: report whether explicit host metadata is available.
    host = batch.metadata.get("host", pd.Series("", index=batch.metadata.index))
    values = host.map(lambda value: "yes" if str(value).strip() else "no")
    result = FieldResult(values.rename("has_host"))
    result.validate(batch)
    return result
```

Construct a tiny fixture with `input.prepare_batch(pd.DataFrame(...))`, call
`run(batch)`, and inspect `result.values`. No taxonomy, CLI, or model is needed.
To activate it, add its CLI flag and append its result to `fields` in `main`.
The combiner needs no change.

## Develop a source module in isolation

```python
from modules.contracts import MetadataBatch, SourceResult, SourceVocabulary, empty_source_table


def run(batch: MetadataBatch, sources: SourceVocabulary) -> SourceResult:
    # Example partial rule: only resolve an explicit, exact sheep host.
    keys = [key for key, row in batch.metadata.iterrows()
            if "sheep" in sources.names and row.get("host") == "Ovis aries"]
    table = empty_source_table(sources, keys)
    table["best_hit"] = "sheep"
    table["source_method"] = "exact_host"
    table["source_evidence"] = "host=Ovis aries"
    result = SourceResult(table)
    result.validate(batch, sources)
    return result
```

Wire a partial source stage into `classify_sources` before its fallback, pass it
the remaining batch from `combine.unresolved`, and include its result when
combining. To add an alternative full classifier, extend `--method` and dispatch
it with a stage-specific configuration dataclass. Keep configuration independent
of `argparse.Namespace`. `SourceModule` and `FieldModule` protocols describe
configured callables; configuration can be bound with `functools.partial`.

Test empty input, nonconsecutive/reordered keys, duplicate external IDs, missing
fields, unavailable vocabulary labels, and unknown/unresolved distinctions.
Source implementations should test that each requested row gets the intended
decision and that expensive resources are not loaded for empty input. Mock model
boundaries for offline tests; tiny real tensors cover LLM generation mechanics.
Run the suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

`metalyzer.py` exposes CLI orchestration only. Former helper imports should move
to their owning modules (`modules.records.build_record`,
`modules.nli.parse_source_scores`, etc.). Top-level `taxonomy.py` remains a thin
compatibility import for its previous public API. The experimental
`metalyzer_mistral.py` remains a separate script.
