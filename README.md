# Metalyzer

Metalyzer classifies sequencing-sample metadata into biological host and
environmental source categories. It first resolves explicit host taxonomy IDs
and host names against a local NCBI taxonomy dump and the configured
[source vocabulary](sources.tsv), then sends unresolved records to NLI, a local
Ministral model, or the Mistral API. It also extracts collection year and
country and writes one ordinary TSV for every input mode.

## Installation

The [GPU Conda environment](environment.yml) uses Python 3.11, PyTorch 2.x,
Transformers 5.x, pandas, PyArrow and the local-model tokenizer dependencies.
It requires a compatible NVIDIA driver and cached or downloadable model
weights. The [CPU environment](environment-cpu.yml) supports tests and
`--device -1`. Taxonomy lookup uses Python's standard library. Mistral API
mode additionally needs the optional `mistralai` v2 SDK and an API key file.

```bash
git clone https://github.com/aldertzomer/metalyzer.git
cd metalyzer
conda env create -f environment.yml
conda activate metalyzer
```

For CPU use `conda env create -f environment-cpu.yml`, activate
`metalyzer-cpu`, and pass `--device -1`. A local NCBI taxonomy dump must
contain `nodes.dmp`, `names.dmp`, and `merged.dmp`. To download it once,
run `python metalyzer.py --download-taxonomy taxonomy`.

## Quick start

Classify a TSV with a unique run-ID column:

```bash
python metalyzer.py --metadata my_samples.tsv --id-col run_accession \
  --sources sources.tsv --taxonomy-dir taxonomy --method nli --out results.tsv
```

Classify one local BioSample accession:

```bash
python metalyzer.py --biosample SAMN12345678 --sources sources.tsv \
  --taxonomy-dir taxonomy --method llm --out SAMN12345678.tsv
```

Use `--biosample SAMN... SAMEA... SAMD...` for several accessions or
`--biosample-file accessions.txt` for one accession per line. BioSample mode
uses `accession` as its ID automatically. It reads the local NCBI BioSample
Parquet store at `/mnt/sdd1/data/biosample` by default; use
`--biosample-path /another/store` to override it. Classification does not
query NCBI or ENA for individual accessions. Missing accessions fail before
model loading. [Building the local BioSample database](#building-the-local-biosample-database)
explains one-time setup.

## Input metadata and sources

Ordinary `--metadata` input is a tab-separated file with an ID column named
by `--id-col`; all other columns may contain sample metadata. BioSample input
preserves available structural fields and arbitrary attributes as normal
columns, putting source-informative fields first. Both modes append the same
standard classification and extraction columns. `benchmark.tsv` is a
historical AllTheBacteria (ATB) ENA
metadata export, while current BioSample Parquet records can reflect later
NCBI updates.

[sources.tsv](sources.tsv) defines source labels, optional
`taxonomy_anchors`, and optional `host_aliases`. Numeric `host_tax_id`
and `host_taxid` refer to the biological host. `host` and
`host_scientific_name` can resolve by exact NCBI scientific, common or
synonym name. Generic `organism` and `tax_id` usually identify the sequenced
organism, often a pathogen, and are never host taxonomy evidence.

Aliases such as `canine` or `porcine` are exact, configured operational
assumptions rather than NCBI taxonomic identities. Disable them with
`--disable-host-aliases` for strict taxonomy-only analysis. The conservative
`other_animal` rule for specific, unlisted animal taxa is opt-in via
`--other-animal-taxonomy`; broad taxa and ecologically ambiguous birds
remain for model classification. Taxonomy assignments must match a unique
configured source. Conflicting host evidence falls back to the selected model.
The [host terminology audit](benchmark_smoke/host_name_audit/README.md)
documents these choices.

## Classification methods

Deterministic taxonomy runs before every model. An explicit numeric host ID
has priority, followed by exact NCBI host-name matching and then configured
host aliases. Unique deterministic assignments skip source-model inference.
Unresolved records use one of:

| `--method` | Source classifier | Main setting |
|---|---|---|
| `nli` | Local DeBERTa zero-shot NLI | `--min-score 0.2`; `--batch-size 64` |
| `llm` | Local Ministral generation | `--llm-min-score 0.75`; `--llm-batch-size 1` |
| `mistral` | Paid Mistral API | `--api-key-file key.txt`; `--mistral-model mistral-small-latest` |

Below-threshold or invalid source predictions become `unknown`. Mistral API
request failures stop the run. By default, a separate NLI check scores support
for each final, non-`unknown` source using the same rendered metadata text;
`--disable-verify-source` skips it. The verification score assesses support
for the selected label, not whether it beats every alternative.

## Metadata preprocessing

Default Variant C filtering removes known administrative identifiers,
submission details, technical sequencing and preparation fields, and generic
BioSample registration profiles **from model text**. Equivalent
`original_attributes.*` duplicates are suppressed only when their canonical
field has the same meaning and value. Unique or conflicting original values,
unexpected biological attributes, host/source fields, titles, descriptions,
collection dates and environmental measurements remain available.

The complete input metadata remains in memory and at the front of the normal
output TSV. Filtering preserves the original field order, acts before the
default 300-character value and 2,000-character record limits, and supplies
the same text to source classification and NLI verification. Add
`--disable-noisy-field-exclusion` to restore the previous model rendering
for either input mode and any method. Each run writes `<out>.filtering.json`
and `<out>.excluded_fields.tsv` alongside its normal log.

## Output

The result TSV starts with the original metadata columns, then one score
column per configured source, `best_hit`, `source_method`,
`source_evidence`, `source_llm_score`,
`nli_verification_score`, `year`, and `country`. Scores for
taxonomically assigned rows and for generative model alternatives are `NA`;
the local LLM's generated-label score is recorded separately and is not a
calibrated correctness probability.

`source_method` records provenance: `host_tax_id` for explicit numeric
host IDs, `host_name` for exact NCBI host-name resolution, `host_alias`
for configured terms, `host_other_animal` for the opt-in animal rule, and
`nli`, `llm` or `mistral` for model decisions. `source_evidence`
contains a short explanation or model fallback reason. `year` and
`country` are extracted independently of source classification. The
`<out>.log` file records taxonomy, model and filtering provenance.

## Current benchmark results

**The BioSample metadata has changed since the original AllTheBacteria ENA
snapshot was downloaded on 6 May 2025. Several records had severe host
discrepancies between the old ATB dataset and current BioSample metadata. We
therefore excluded 45 disputed records and reviewed the reference
annotations. The current benchmark contains 1,275 unique sequencing runs.**

The exclusions are the 44 documented CAGED pairs whose historical host is
goat (*Capra hircus*) but current BioSample host is chicken (*Gallus gallus*),
plus `ERR2197811`, whose historical duck evidence conflicts with the current
chicken annotation. No excluded run is reintroduced in the current evaluation.
Among retained runs, 57 source labels were corrected: 41
`unknown→human`, four `unknown→chicken`, one `unknown→cattle`, one
`unknown→water`, one `other_food→chicken`, and nine
`water→environment`. `SRR10257508` remains `other_food`; nine
seasonally ambiguous Troups Creek Wetland isolates are `environment`;
seal isolates are `other_animal`, while seal-study sediment is
`environment`. Previous country and year references are preserved.

Some corrections relied on manual metadata adjudication, including
model-informed decisions. These labels are a **working benchmark reference,
not fully independent validation data**. Previous 1,320-run scores cannot be
compared directly: both the reference population and the model pipeline have
changed. The [current dataset manifest](benchmark_smoke/current_benchmark/inputs/dataset_manifest.json)
records source and derived-file checksums and the exact exclusions.

The current 1,275-run benchmark uses both historical ATB TSV and current
BioSample metadata, all three classifiers, configured host aliases, and
explicit `--other-animal-taxonomy`. Noisy-field filtering is on by default;
the paired off condition passes `--disable-noisy-field-exclusion`. Both
conditions use the same taxonomy configuration, model settings, verification,
and source vocabulary. Detailed results, per-class metrics, matrices,
timings and per-run comparisons are in the
[current benchmark report](benchmark_smoke/current_benchmark/README.md).

<!-- BEGIN CURRENT_BENCHMARK_TABLE -->

| Input | Method | Filter | Accuracy | Assigned precision | Unknown | Taxonomy assignments |
|---|---|---|---:|---:|---:|---:|
| ATB TSV | NLI | On | 85.1% | 88.8% | 92 | 234 |
| ATB TSV | NLI | Off | 85.1% | 89.8% | 109 | 234 |
| ATB TSV | Ministral | On | 89.1% | 98.8% | 164 | 234 |
| ATB TSV | Ministral | Off | 88.5% | 98.6% | 170 | 234 |
| ATB TSV | Mistral API | On | 95.3% | 98.7% | 84 | 234 |
| ATB TSV | Mistral API | Off | 95.1% | 98.4% | 80 | 234 |
| BioSample | NLI | On | 88.9% | 93.2% | 93 | 516 |
| BioSample | NLI | Off | 79.9% | 83.4% | 74 | 516 |
| BioSample | Ministral | On | 90.0% | 94.1% | 72 | 516 |
| BioSample | Ministral | Off | 92.9% | 98.6% | 115 | 516 |
| BioSample | Mistral API | On | 96.7% | 96.9% | 34 | 516 |
| BioSample | Mistral API | Off | 96.4% | 96.5% | 31 | 516 |

<!-- END CURRENT_BENCHMARK_TABLE -->

## Reproducibility and development

The current workflow derives synchronized 1,275-row inputs without editing
the historical snapshots or the adjudicated truth labels. It joins by
`run_accession`, preserves input order and duplicate BioSample requests,
checks file and code signatures before reuse, and writes separate outputs
under `benchmark_smoke/current_benchmark/`.

```bash
python benchmark_smoke/run_current_benchmark.py --phase prepare
python benchmark_smoke/run_current_benchmark.py --phase local
# Only after approval for fresh paid API calls:
python benchmark_smoke/run_current_benchmark.py --phase api --confirm-paid-api
python benchmark_smoke/evaluate_current_benchmark.py
python -m unittest discover -s tests -v
```

The local phase runs NLI and Ministral on the GPU. The API phase is separate
because it incurs charges. The evaluator aligns each BioSample prediction
through the stored run-to-BioSample mapping and checks deterministic
assignments, metadata, year/country values and output schema between filtering
conditions. Manifests record input, source, taxonomy and code hashes, model
settings, GPU details, timings and API usage where available.

### Building the local BioSample database

BioSample lookup requires the large NCBI BioSample XML archive converted to
local Parquet. The one-time builder streams the compressed XML in bounded
memory, retains its download for rebuilds, and creates an accession index:

```bash
python -m modules.biosample_build --output-dir /mnt/sdd1/data/biosample
```

The source archive is
[NCBI BioSample XML](https://ftp.ncbi.nlm.nih.gov/biosample/biosample_set.xml.gz).
Downloading and converting the complete repository may take several hours and
needs substantial space for the compressed XML, Parquet and index. The
[builder documentation](benchmark_smoke/archive/README_before_1275.md#building-the-local-biosample-database)
explains the options and safe replacement behaviour. Subsequent Metalyzer
lookups use only the local store.

## Historical benchmarks

Earlier results remain available as historical evidence, with their own
populations and methods: the [six-method baseline](benchmark_smoke/six_method_evaluation/summary.tsv),
[host-name audit](benchmark_smoke/host_name_audit/README.md),
[host-terminology benchmark](benchmark_smoke/host_terminology_benchmark/README.md),
[TSV host-terminology benchmark](benchmark_smoke/tsv_host_terminology_benchmark/README.md),
and [1,320-run Variant C comparison](benchmark_smoke/noisy_field_exclusion/README.md).
The [archived detailed README](benchmark_smoke/archive/README_before_1275.md)
preserves the earlier methodology and figures. Those scores use older
references and should not be presented as current 1,275-run performance.
