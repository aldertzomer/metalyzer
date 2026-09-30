# Metadata Classification Pipeline (Zero-shot + Deterministic Parsing)

This repository contains a Python pipeline to classify biological metadata records using a combination of:

- Source/host classification with DeBERTa zero-shot NLI (default) or a local Hugging Face LLM
- Deterministic parsing (for year and country)

The pipeline is designed for large-scale datasets (e.g. ENA/SRA metadata) with heterogeneous formatting.

---
## For the impatient

If you build IKEA wardrobes without ever looking at the instructions, and you are more of a try first, read later person, here is the commandline:

```bash
export TOKENIZERS_PARALLELISM=true
python metalyzer.py \
  --metadata benchmark.tsv \
  --sources sources.tsv \
  --out classified.tsv \
  --id-col run_accession \
  --device 0 \
  --batch-size 64 \
  --min-score 0.2
```

## Overview

### Modular development

Metalyzer separates the pipeline into independent stages under
[`modules/`](modules/README.md): date extraction, country normalization,
deterministic source parsing (including taxonomy ID-to-scientific-name lookup),
NLI, local LLM, Mistral API, input preparation, and output combination. `metalyzer.py` contains
command-line options and stage orchestration.

Developers can start with the [module contracts and extension examples](modules/README.md)
without reading the rest of the pipeline. Typed batch/result containers in
[`modules/contracts.py`](modules/contracts.py) define row identity, source scores,
provenance, missing values, and validation. Default commands and TSV columns are
preserved. `--skip-date` and `--skip-country` disable those stages and omit their
output columns. Taxonomy is enabled with `--taxonomy-dir`; `--method nli|llm|mistral`
selects the fallback classifier, which runs only on unresolved rows.

For each metadata row, the pipeline performs:

### 1. Source classification (host taxonomy, then NLI, local LLM or Mistral)

1. If `--taxonomy-dir` is supplied, resolve explicit `host_tax_id` values using local NCBI taxonomy.
2. Unambiguous taxonomy-derived host assignments take precedence over language-model inference.
3. Only unresolved rows reach the selected backend: DeBERTa zero-shot NLI (`--method nli`, the default), a local generative LLM (`--method llm`), or the Mistral API (`--method mistral`).
4. NLI predictions below `--min-score` become `unknown`.

The zero-shot model is:

MoritzLaurer/deberta-v3-large-zeroshot-v2.0

The metadata row is converted into a structured string:

host: Gallus gallus; isolation source: neck skin; country: USA

This is evaluated against candidate labels using NLI.

---

### 2. Year extraction (deterministic)

Extracts a 4-digit year (1905–2030) from:

- collection_date
- 
- collection_date_start
- 
- collection_date_end


Supported formats:

2019

2016-04

31-12-19

15-06-18

2007-11


Handles both:

- 19YY
  
- 20YY


---

### 3. Country normalization (deterministic)

Normalizes messy country fields such as:

USA:WY

U.S.A;USA

Canada: Calgary, Alberta

United Kingdom: Oxford

to standardized country names using:

- alias mapping
  
- pycountry
  
- controlled fallback (no full-text fuzzy matching)
  

---

## Input

### Metadata table (TSV)

Get the metadata table from ENA or ATB, remove all non important columns. Put the source/host interpretable colums as 2nd, 3rd, etc columns for slightly improved performance, then save it as tab delimited file. The first column should start with "run_acc". See for an example benchmark.tsv. 

Example:

|run_acc |   host           |  isolation_source |   collection_date  |  country |
---------|------------------|-------------------|--------------------|----------|
|ERR001  |   Gallus gallus  |  neck skin        |  2019              |  USA     |

For the text sent to NLI, the local LLM, and the Mistral API, Metalyzer omits
the configured ID column, `host_tax_id`, and generic `tax_id`. The first is an
identifier; deterministic taxonomy already handles `host_tax_id`; generic
`tax_id` often names the sequenced organism rather than the sample source.
The original metadata table remains available unchanged to taxonomy, date,
country, and output processing. Other metadata fields remain in the rendered
record. Values are trimmed, CR/LF and repeated whitespace become one space,
and standard HTML entities are decoded (`&amp;` becomes `&`). Punctuation,
capitalization, and field order are preserved. The usual value and record
length limits still apply.


---

### Sources file (TSV)

Put the canonical source name first, followed by human-readable hints in parentheses.
The optional `taxonomy_anchors` column holds comma-separated NCBI Taxonomy IDs
for deterministic `host_tax_id` matching. Leave it blank when no taxon safely
identifies that source; a file containing only `source` is also supported.

| source | taxonomy_anchors |
| --- | --- |
| chicken (poultry host Gallus gallus) | 9031 |
| human (human host Homo sapiens) | 9606 |
| laboratory (artificial or synthetic sequences) | 81077,32630 |

Hints help the text classifiers distinguish similar sources. Taxonomy IDs are
configuration for the deterministic matcher only; NLI, the local LLM and the
Mistral API receive only the `source` descriptions.


---

## Output

A TSV file with:

id    <source scores...>    best_hit    source_method    source_evidence    source_verification_score    year    country

Example:

|run_acc| chicken|human|cattle|best_hit|source_method|source_evidence|source_verification_score|year|country|
|-------|--------|-----|------|--------|-------------|---------------|-------------------------|----|-------|
|ERR001 |  0.85  |0.01 | 0.02 |chicken|nli||0.93|2019|United States|

- One score column per source, named using the text before the first `(` in `sources.tsv`; multi-word names are preserved
- For NLI rows, `best_hit` is the name of the source with the highest score; ties use the first source in `sources.tsv`
- In NLI mode, `--min-score` sets the minimum top score required for `best_hit`; lower-scoring rows are labelled `unknown` while their score columns are retained. The default is `0.2`.
- Full source labels, including parenthetical hints, are still used for classification
- Short source names must be nonempty, unique, and distinct from the ID, `best_hit`, `source_method`, `source_evidence`, `source_verification_score`, `year`, and `country` column names
- Taxonomy-derived rows use `source_method=host_tax_id` and **all source scores are `NA`**, because no NLI inference was performed. They are not artificial probabilities and are not subject to `--min-score`.
- NLI rows use `source_method=nli`, including below-threshold `unknown` calls; their `source_evidence` is blank.
- Local LLM rows use `source_method=llm` and all source scores are `NA`: generated labels have no calibrated candidate probabilities. Valid answers have blank evidence; malformed answers become `unknown` with `source_evidence=invalid_llm_output=...` (sanitized and truncated to 200 characters).
- Mistral API rows use `source_method=mistral` with the same `NA` score convention. `unknown` is always allowed, even when absent from the sources file. Malformed answers have `source_evidence=invalid_mistral_output=...`; request/authentication/quota failures abort instead of producing unknown labels.
- `source_verification_score` is a binary NLI entailment/support score for the assigned `best_hit`, not a calibrated probability that the classification is correct. It is `NA` for `unknown` and when `--disable-verify-source` is set.
- year as 4-digit string
- country as normalized name

Each analysis also writes a human-readable log beside its TSV: `classified.tsv`
produces `classified.tsv.log`. Version 0.2 prints the same progress and summary
messages to the terminal and records provenance in that log: command line,
local Git commit when available, Python/OS, method and model settings, input
paths, a SHA256 of `sources.tsv`, local taxonomy file paths/sizes/timestamps,
and input dimensions. The log also records classification totals and source
verification counts. Warnings appear on screen and in the log; failures exit
nonzero and leave a full Python traceback in the log. API key contents are
never logged. Logging and provenance add no columns to the classification TSV.
Check the installed release with `python metalyzer.py --version`; it exits
without analysis arguments and prints `metalyzer 0.2`.

---

### Source verification

After source classification, Metalyzer uses the same DeBERTa NLI model and
hypothesis wording as `--method nli` to test each assigned source against the
original rendered metadata. It evaluates **only one hypothesis per non-unknown
row**, using the selected source's full description from `sources.tsv`, including
its hints. The score is equivalent to zero-shot `multi_label=True` for that one
candidate. This does not rerun the full NLI classification or compare the
assigned source with alternatives. It applies to `host_tax_id`, `nli`, `llm`,
and `mistral` assignments alike.

Verification runs by default. Pass `--disable-verify-source` to skip it and
write `NA` in `source_verification_score` for every row. An `unknown` assignment
is never scored. On NLI runs, the classifier and verifier share the loaded NLI
pipeline. On LLM and Mistral API runs, verification loads the NLI model for
non-unknown assignments, adding model memory and inference time.

---

## Installation

Requirements:

- Python ≥ 3.10
- Conda environment recommended

For an NVIDIA CUDA GPU, install from Conda:

```bash
conda env create -f environment.yml
conda activate metalyzer
```

`environment.yml` selects conda-forge's `pytorch-gpu`; Conda resolves the CUDA
runtime dependencies for the target system. A compatible NVIDIA driver is
required. The file lists application dependencies rather than fixing every
platform-specific package build or a user-specific installation path.

### CPU-only installation

For a machine without CUDA, create the CPU environment instead:

```bash
conda env create -f environment-cpu.yml
conda activate metalyzer-cpu
```

`environment-cpu.yml` explicitly selects conda-forge's `pytorch-cpu`
metapackage and therefore does not install CUDA, FlashAttention, or Triton.
Run the pipeline with `--device -1`.

Both specifications use Python 3.11, Transformers 5.x, PyTorch 2.x,
`mistral-common>=1.8.6`, `accelerate`, and the SentencePiece/Protobuf tokenizer
dependencies. They are environment specifications,
not exact lockfiles. Taxonomy dump parsing and downloading use the Python
standard library. The pipeline and tests use the dependencies listed in these
environments; the taxonomy feature requires no additional packages.
The Mistral module requires the optional `mistralai` v2 SDK. It is loaded only
when unresolved rows need API inference; NLI, local LLM and taxonomy-only runs
do not require it.

On the Linux GPU server used for the local Ministral test, the CUDA build also
required `cuda-driver-dev=12.9` from conda-forge. Install it there with
`conda install -c conda-forge cuda-driver-dev=12.9` if needed; it is omitted
from the cross-platform environment files because CPU and Windows installations
do not use it.

To update an existing environment, use the matching command:

```bash
conda env update -n metalyzer -f environment.yml
# Or, for the CPU environment:
conda env update -n metalyzer-cpu -f environment-cpu.yml
```

An update may retain packages from an older environment. Create a fresh CPU
environment when switching from a CUDA installation.

---

## Usage

### Optional local NCBI host taxonomy

Download the [official NCBI taxonomy dump](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/)
once (network access occurs only when this command is explicitly requested):

```bash
python metalyzer.py --download-taxonomy taxonomy
```

This fetches `https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz` and
installs `nodes.dmp`, `names.dmp`, and `merged.dmp` in `taxonomy/`. You can
also download and unpack these three files manually. Run the download command
again to refresh the local data; retain a copy of the dump for reproducible runs.

```bash
python metalyzer.py --metadata benchmark.tsv --sources sources.tsv \
  --out classified.tsv --id-col run_accession \
  --taxonomy-dir taxonomy --device -1 --batch-size 10
```

Use `--device 0` for GPU execution. Normal taxonomy lookup is entirely local:
files load once and lineages are cached. Only `host_tax_id` triggers taxonomy
classification; the sample/pathogen `tax_id` is never used as host taxonomy.
Numeric strings, integral decimal values such as `9940.0`, quoted IDs, and
obsolete IDs in `merged.dmp` are supported. Missing or invalid IDs fall back
to the selected text classifier. Omitting `--taxonomy-dir`, or supplying unreadable/malformed files,
produces a warning and uses the selected text classifier for every row.

The `taxonomy_anchors` column in `sources.tsv` controls these assignments.
Each ID maps that taxon and all its descendants to the row's source, regardless
of taxonomic rank. For each valid `host_tax_id`, Metalyzer normalizes merged IDs
and walks its lineage from the most specific taxon toward the root; the first
configured anchor wins. Thus a species anchor such as chicken (`9031`) can
override a broader Aves (`8782`) anchor, and a cat (`9685`) anchor can override
Carnivora (`33554`) in a custom vocabulary. At startup, configured anchors are
checked against the loaded taxonomy dump. Obsolete IDs are normalized and
logged; nonexistent IDs, broken lineages, or a normalized anchor assigned to
multiple sources stop the run with a configuration error. The log reports the
configured, valid, merged, invalid, and conflicting anchor counts. Blank
anchors or no matching lineage for a valid host fall back to the selected
classifier. The generic sample/pathogen `tax_id` is intentionally never source
evidence, since it often identifies the sequenced organism instead of its host.

The repository file configures domestic pig (`9825`), chicken (`9031`), turkey
(`9103`), cattle (`9913`), sheep (`9940`), goat (`9925`), human (`9606`), dog
(`9615`), and cat (`9685`). It also configures the environmental-samples node
`1936016` and artificial/synthetic sequence nodes `81077,32630`. Descendants
of a configured environmental node inherit its source; this applies to soil,
marine or air metagenome nodes only when they actually fall below that anchor
in the supplied NCBI dump. Ordinary organism taxids, such as *E. coli* `562`,
do not imply `laboratory`: a laboratory strain may have come from any source.
The broad Metazoa anchor is left blank for `other_animal` in the repository file
because it would also absorb birds and animal sources whose ecological or food
context taxonomy alone cannot establish. Custom source files can use higher-level
anchors when their classes make that inheritance appropriate. These IDs are
never passed to NLI or generative models; their source descriptions remain
unchanged.

For a sheep host, the output looks like this (all other source scores are also `NA`):

```text
run_accession   chicken turkey  pig cattle  sheep   ... best_hit source_method   source_evidence
SRR17929619     NA      NA      NA  NA      NA      ... sheep    host_tax_id     host_tax_id=9940; Ovis aries
```

Input order, year/country extraction, and NLI score column names are preserved.
Each run reports taxonomy calls, selected classifier calls, and its `unknown` calls.
LLM runs also report the number of invalid model outputs.
If all hosts resolve, the fallback classifier is not loaded. The NLI verifier
still loads for those assignments unless `--disable-verify-source` is set.

Run the offline unit and integration tests with:

```bash
python -m unittest discover -s tests -v
```

### Standard NLI usage

export TOKENIZERS_PARALLELISM=true
```bash
python metalyzer.py \
  --metadata benchmark.tsv \
  --sources sources.tsv \
  --out classified.tsv \
  --id-col run_accession \
  --method nli \
  --device 0 \
  --batch-size 64 \
  --min-score 0.2
```
---
On the supplied benchmark, `--min-score 0.2` identifies 74 of 100 records
labelled `unknown`, but labels 85 known-source records as `unknown`. It gives
83.2% overall accuracy and 88.2% precision among records assigned a source.
Use `--min-score 0.17` to maximize overall benchmark accuracy (84.0%), or
`--min-score 0.3` when higher precision for assigned sources (95.7%) matters
more than coverage. These cutoffs are benchmark-specific and should be
rechecked for a new source list or dataset.

Use `--device 0` for the first GPU (the default), or another nonnegative GPU
index. GPU execution requires a CUDA-enabled PyTorch installation.

For CPU execution, use `--device -1`. For example, to run the benchmark:

```bash
python metalyzer.py --metadata benchmark.tsv --sources sources.tsv --out classified.tsv --id-col run_accession --device -1 --batch-size 10
```

NLI CPU execution explicitly uses float32 to avoid slow float16 inference.
GPU execution uses the model checkpoint's precision (`dtype="auto"`).

## Benchmark results

The saved benchmark TSVs below were generated before source verification was
added, so they do not contain `source_verification_score`.

The following results evaluate the current `benchmark_classified.tsv` using
`sources.tsv` against `benchmark_true_labels.tsv`, with `--min-score 0.2` and
local host taxonomy enabled. The run contains 255 `host_tax_id` assignments and
1,065 NLI assignments. Accuracy is recall: correct calls divided by the number
of true records for a source. Precision is correct calls divided by the number
of calls made for a source.

| Source | True rows | Called rows | Correct | Accuracy | Precision |
|---|---:|---:|---:|---:|---:|
| cat | 20 | 22 | 20 | 100.0% | 90.9% |
| cattle | 99 | 109 | 99 | 100.0% | 90.8% |
| chicken | 101 | 99 | 99 | 98.0% | 100.0% |
| dog | 100 | 100 | 100 | 100.0% | 100.0% |
| environment | 4 | 61 | 1 | 25.0% | 1.6% |
| goat | 100 | 98 | 97 | 97.0% | 99.0% |
| human | 100 | 87 | 87 | 87.0% | 100.0% |
| laboratory | 1 | 1 | 1 | 100.0% | 100.0% |
| other_animal | 100 | 49 | 48 | 48.0% | 98.0% |
| pig | 100 | 112 | 100 | 100.0% | 89.3% |
| sheep | 100 | 95 | 95 | 95.0% | 100.0% |
| turkey | 101 | 87 | 87 | 86.1% | 100.0% |
| unknown | 98 | 136 | 79 | 80.6% | 58.1% |
| wastewater | 5 | 19 | 5 | 100.0% | 26.3% |
| water | 92 | 81 | 78 | 84.8% | 96.3% |
| waterbird | 99 | 50 | 50 | 50.5% | 100.0% |
| wildbird | 100 | 114 | 98 | 98.0% | 86.0% |

The source list yields 1,144 correct calls of 1,320 (**86.7% overall
accuracy**) and assigns a non-unknown source to 1,184 records (89.7%). Precision
among assigned records is 89.9%. The run has 136 NLI calls below the cutoff.

### Confusion matrices

Rows are true sources, columns are predicted sources, and `n` is the number of
true records in the row. The `unknown` prediction is created by the 0.2 score
cutoff, not by a source candidate.

#### Absolute counts

[![Benchmark source confusion matrix: absolute counts](assets/benchmark-confusion-absolute.svg)](assets/benchmark-confusion-absolute.svg)

The image is a full source-by-source table. Click it to inspect at full resolution.

#### Row percentages

[![Benchmark source confusion matrix: row percentages](assets/benchmark-confusion-percent.svg)](assets/benchmark-confusion-percent.svg)

Each row shows the share of records with that true source assigned to every
predicted source. The SVG tables are generated from the benchmark TSV files by
`python render_benchmark_matrices.py`.

### Local Hugging Face LLM classifier

`--method llm` selects local generative classification with
[`mistralai/Ministral-3-8B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
by default. Transformers downloads the tokenizer and weights on first use and
reuses the standard Hugging Face cache on subsequent runs. Inference runs on
your machine; no API or API key is required. The existing environments contain
the required dependencies.

Taxonomy still takes precedence: only unresolved `host_tax_id` rows reach the
LLM. The LLM does not load if taxonomy resolves every row. With verification
enabled, DeBERTa also loads to score the final non-unknown assignments. The
natural-language metadata representation and
deterministic year/country processing are shared with NLI.

The same `sources.tsv` supplies canonical labels and their full descriptions.
The LLM may also answer `unknown` when source evidence is insufficient; this
does not add an `unknown` score column unless that source is in your file.
The prompt tells the model that field names alone are not source evidence and
that ambiguous or absent source values should map to `unknown`. These rules
also apply to the separate Mistral API method.
All LLM score columns are `NA`. `--min-score` applies only to NLI; in LLM mode
it prints an informational message and does not affect predictions.

```bash
python metalyzer.py \
  --metadata benchmark.tsv \
  --sources sources.tsv \
  --out classified_ministral.tsv \
  --id-col run_accession \
  --taxonomy-dir taxonomy \
  --method llm \
  --llm-model mistralai/Ministral-3-8B-Instruct-2512 \
  --device 0 \
  --llm-batch-size 1
```

Use `--device 0` for the first CUDA GPU, another nonnegative index for that
specific GPU, or `--device -1` for CPU. An unavailable GPU causes a clear error
instead of silently switching devices. Omit `--taxonomy-dir taxonomy` if you
have not downloaded taxonomy.

CPU example:

```bash
python metalyzer.py --metadata benchmark.tsv --sources sources.tsv \
  --out ministral_cpu.tsv --id-col run_accession \
  --method llm --llm-model mistralai/Ministral-3-8B-Instruct-2512 \
  --device -1 --llm-batch-size 1
```

For a quick test, add `--limit 10` to either command. `--limit` also works with
NLI and selects the first N metadata rows before classification.

`--llm-model` accepts another compatible Ministral 3 checkpoint;
`--llm-revision` optionally pins both tokenizer and weights to a specific
Hugging Face revision. Models are never substituted automatically. Generation
is greedy and defaults to `--llm-max-new-tokens 16`. The Mistral tokenizer
formats and pads batches of chat conversations directly.

`--llm-batch-size` defaults to 1 independently of NLI's `--batch-size 64`.
The LLM retains checkpoint precision (`dtype="auto"`) on both CPU and GPU and
loads with `low_cpu_mem_usage=True`. GPU loading uses `device_map` for the
requested GPU and requires `accelerate`; the model's Mistral tokenizer requires
`mistral-common`. The checkpoint uses FP8 weights, but actual memory includes
other weights, activations, generation cache, loading overhead and other
processes. The supplied test was run on a larger GPU server; peak memory was
not recorded. A 12-GB fit is not established. CPU execution may be slow and
may need substantially more memory.

#### Local Ministral benchmark results

The supplied `ministral_test.tsv` was generated on a GPU test server with
`mistralai/Ministral-3-8B-Instruct-2512` and local taxonomy. Against
`benchmark_true_labels.tsv`, it contains all 1,320 records: 255 `host_tax_id`
assignments and 1,065 local LLM assignments. It has 58 `unknown` predictions,
including 2 malformed responses recorded as `invalid_llm_output` evidence.
Accuracy below is recall: correct calls divided by true rows. Precision is
correct calls divided by called rows.

| Source | True rows | Called rows | Correct | Accuracy | Precision |
|---|---:|---:|---:|---:|---:|
| cat | 20 | 20 | 20 | 100.0% | 100.0% |
| cattle | 99 | 97 | 97 | 98.0% | 100.0% |
| chicken | 101 | 107 | 101 | 100.0% | 94.4% |
| dog | 100 | 99 | 99 | 99.0% | 100.0% |
| environment | 4 | 9 | 2 | 50.0% | 22.2% |
| goat | 100 | 98 | 98 | 98.0% | 100.0% |
| human | 100 | 118 | 99 | 99.0% | 83.9% |
| laboratory | 1 | 24 | 0 | 0.0% | 0.0% |
| other_animal | 100 | 90 | 89 | 89.0% | 98.9% |
| pig | 100 | 99 | 99 | 99.0% | 100.0% |
| sheep | 100 | 100 | 100 | 100.0% | 100.0% |
| turkey | 101 | 101 | 101 | 100.0% | 100.0% |
| unknown | 98 | 58 | 47 | 48.0% | 81.0% |
| wastewater | 5 | 20 | 5 | 100.0% | 25.0% |
| water | 92 | 68 | 68 | 73.9% | 100.0% |
| waterbird | 99 | 122 | 97 | 98.0% | 79.5% |
| wildbird | 100 | 90 | 86 | 86.0% | 95.6% |

The local Ministral run yields 1,208 correct calls of 1,320 (**91.5% overall
accuracy**) and assigns a non-unknown source to 1,262 records (95.6%). Precision
among assigned records is 92.0%. Of the 98 truly `unknown` records, 24 were
called `laboratory`; the new prompt rules should be judged against this observed
limitation. These are results from the supplied test file; the integrated
modular backend has not been rerun with the full model on this machine, and the
TSV does not record the exact prompt revision used on the server. Use
`python metalyzer.py --method llm` for the integrated local Ministral backend.

For comparison, the earlier Qwen3-4B local run in `benchmark_llm.tsv` achieved
1,223/1,320 (**92.7%**) with 104 `unknown` predictions and 10 invalid outputs.
That file and its [absolute](assets/benchmark-llm-confusion-absolute.svg) and
[percentage](assets/benchmark-llm-confusion-percent.svg) matrices remain as
historical results; Qwen is no longer the `--method llm` backend.

Rows are true sources and columns are predicted sources. Click either image to
inspect the full matrix. The underlying [per-class metrics](benchmark_smoke/local_ministral_evaluation/ministral_test_per_class.tsv),
[counts](benchmark_smoke/local_ministral_evaluation/ministral_test_confusion_counts.tsv),
and [row percentages](benchmark_smoke/local_ministral_evaluation/ministral_test_confusion_percent.tsv)
are also available as TSV files.

##### Absolute counts

[![Local Ministral source confusion matrix: absolute counts](assets/benchmark-ministral-confusion-absolute.svg)](assets/benchmark-ministral-confusion-absolute.svg)

##### Row percentages

[![Local Ministral source confusion matrix: row percentages](assets/benchmark-ministral-confusion-percent.svg)](assets/benchmark-ministral-confusion-percent.svg)

### Experimental Mistral API classifier

The modular pipeline now supports `--method mistral`; see the
[Mistral module usage](#mistral-api-module) below. The benchmark results in this
section came from the original standalone script, not the new module.

`metalyzer_mistral.py` is an experimental alternative that asks the Mistral
API to choose one controlled source label. It performs well but of course it is not free. It requires the `mistralai` Python
package and a text file containing a Mistral API key.

```bash
conda install mistralai # in the metalyzer environment

python metalyzer_mistral.py \
  --metadata benchmark.tsv \
  --sources sources_mistral.tsv \
  --out benchmark_mistral.tsv \
  --id-col run_accession \
  --api-key-file ~/mistral.key
```

The Mistral output contains a predicted `source` label rather than a score per
candidate, so it does not support a score cutoff. `sources_mistral.tsv`
includes `unknown` as a controlled label.

#### Mistral benchmark results

`benchmark_mistral.tsv` was re-evaluated against the current
`benchmark_true_labels.tsv` using `sources_mistral.tsv`. It contains all 1,320
benchmark records and has 1,267 correct calls: **96.0% overall accuracy**.
Accuracy is recall: correct calls divided by the number of true records for a
source. Precision is correct calls divided by the number of calls made for a
source.

| Source | True rows | Called rows | Correct | Accuracy | Precision |
|---|---:|---:|---:|---:|---:|
| cat | 20 | 20 | 20 | 100.0% | 100.0% |
| cattle | 99 | 99 | 99 | 100.0% | 100.0% |
| chicken | 101 | 106 | 101 | 100.0% | 95.3% |
| dog | 100 | 100 | 100 | 100.0% | 100.0% |
| environment | 4 | 14 | 4 | 100.0% | 28.6% |
| goat | 100 | 100 | 100 | 100.0% | 100.0% |
| human | 100 | 127 | 100 | 100.0% | 78.7% |
| laboratory | 1 | 2 | 1 | 100.0% | 50.0% |
| other_animal | 100 | 86 | 86 | 86.0% | 100.0% |
| pig | 100 | 102 | 100 | 100.0% | 98.0% |
| sheep | 100 | 100 | 100 | 100.0% | 100.0% |
| turkey | 101 | 101 | 101 | 100.0% | 100.0% |
| unknown | 98 | 69 | 69 | 70.4% | 100.0% |
| wastewater | 5 | 0 | 0 | 0.0% | N/A |
| water | 92 | 92 | 92 | 100.0% | 100.0% |
| waterbird | 99 | 94 | 94 | 94.9% | 100.0% |
| wildbird | 100 | 108 | 100 | 100.0% | 92.6% |

| True source | Mistral predicted calls |
|---|---|
| cat | cat: 20 |
| cattle | cattle: 99 |
| chicken | chicken: 101 |
| dog | dog: 100 |
| environment | environment: 4 |
| goat | goat: 100 |
| human | human: 100 |
| laboratory | laboratory: 1 |
| other_animal | human: 12; other_animal: 86; pig: 1; wildbird: 1 |
| pig | pig: 100 |
| sheep | sheep: 100 |
| turkey | turkey: 101 |
| unknown | chicken: 5; environment: 9; human: 11; laboratory: 1; pig: 1; unknown: 69; wildbird: 2 |
| wastewater | environment: 1; human: 4 |
| water | water: 92 |
| waterbird | waterbird: 94; wildbird: 5 |
| wildbird | wildbird: 100 |

---

### Mistral API module

`--method mistral` uses [`modules/mistral.py`](modules/mistral.py) with the
standard pipeline inputs and output columns. Taxonomy still takes precedence;
only unresolved rows are sent to Mistral. Date and country extraction remain
local. The API module itself does not load PyTorch, Transformers, or local model
weights; the default source verification step does load DeBERTa for non-unknown
assignments. Use `--disable-verify-source` when running without local NLI weights.

Install the optional SDK in your active environment:

```bash
conda install -c conda-forge 'mistralai>=2,<3'
```

When API access is available, start with a small run:

```bash
python metalyzer.py \
  --metadata benchmark.tsv \
  --sources sources.tsv \
  --out benchmark_mistral_modular.tsv \
  --id-col run_accession \
  --method mistral \
  --api-key-file ~/mistral.key \
  --mistral-concurrency 1 \
  --mistral-retries 0 \
  --limit 10
```

Add `--taxonomy-dir taxonomy` if you have a local dump. The key file accepts
either the raw key or `MISTRAL_API_KEY=...` (optionally quoted). It is read only
when API requests are needed and is never included in metadata or output.

As with the local LLM, **`unknown` is always an allowed answer**. It is added
to both the prompt and the JSON schema when missing from your source list.
There is no need to edit `sources.tsv`, and no extra `unknown` score column
appears unless your file explicitly includes that source. An explicit unknown
answer has blank evidence. Invalid JSON, unexpected labels or missing responses
become unknown with sanitized `invalid_mistral_output=...` evidence (at most
200 characters after the prefix), without another paid request.

The schema follows the [official Mistral structured-output SDK example](https://github.com/mistralai/client-python/blob/main/examples/mistral/chat/structured_outputs_with_json_schema.py).
Output includes `best_hit`, `source_method=mistral`, `source_evidence`, `year`
and `country`, with all candidate scores `NA`. `--min-score` has no effect.
`--device`, `--batch-size`, and `--llm-*` configure the local backends only.

| Option | Default | Meaning |
|---|---|---|
| `--mistral-model` | `mistral-small-latest` | Model sent to the API; choose a fixed model ID for reproducibility |
| `--mistral-concurrency` | `8` | Maximum concurrent requests/worker tasks |
| `--mistral-retries` | `5` | Additional attempts for timeouts or HTTP 408/429/500/502/503/504 |
| `--mistral-timeout` | `60` | Seconds per request attempt |
| `--mistral-random-seed` | `12345` | Seed sent to Mistral |
| `--mistral-max-tokens` | `32` | Maximum generated tokens per response |
| `--mistral-progress-every` | `10` | Print progress after this many completed rows |

Retries use capped exponential backoff; SDK retries are disabled to keep the
attempt limit predictable. Authentication errors and other permanent failures
stop immediately. Exhausted retries, including quota/rate-limit errors, stop
the run and cancel remaining workers; they never become classification labels.
The output file is written only after the full run succeeds, so API failures
leave an existing output file untouched. Completed API calls may still be billed;
this module does not checkpoint/resume partial runs.

Validation uses mocked API responses, including concurrency, timeouts, retries,
taxonomy precedence and output serialization. No live Mistral call was made for
this implementation; the module's classification accuracy has not been benchmarked.

## Performance Notes

- Source classification runs on the selected CPU or GPU
- Year and country parsing are CPU-light
- Batch size can be increased for better GPU utilization
- current implementation has a single CPU bottleneck. 

---

## Design Rationale

Why not classify year/country with NLI?

- Year and country are usually explicitly present
- Zero-shot classification:
  - is slower
  - introduces unnecessary errors
- Deterministic parsing is:
  - faster
  - more accurate
  - reproducible

---

## Known Limitations

- Ambiguous entries like "Korea" default to "South Korea"
- Missing or noisy metadata may result in:
  - year = ""
  - country = "unknown"

---

## License

MIT License
