# Metadata Classification Pipeline (Zero-shot + Deterministic Parsing)

This repository contains a Python pipeline to classify biological metadata records using a combination of:

- Zero-shot classification (for source/host)
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

For each metadata row, the pipeline performs:

### 1. Source classification (host taxonomy, then NLI)

1. If `--taxonomy-dir` is supplied, resolve explicit `host_tax_id` values using local NCBI taxonomy.
2. Unambiguous taxonomy-derived host assignments take precedence over language-model inference.
3. Only unresolved rows are classified in batches using DeBERTa zero-shot NLI.
4. NLI predictions below `--min-score` become `unknown`.

The zero-shot model is:

MoritzLaurer/deberta-v3-large-zeroshot-v2.0

The metadata row is converted into a structured string:

host="Gallus gallus"; isolation_source="neck skin"; country="USA"

This is evaluated against candidate labels using NLI.

---

### 2. Year extraction (deterministic)

Extracts a 4-digit year (1905–2026) from:

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


---

### Sources file (TSV)

Put the main source first, the hints in parenthesis, good hints are necessary:  e.g. cat/cattle get mixed up. Stool (thing you sit on, or feces), guinea pig might get classified as pig. Try to catch these mistakes. See for an example sources.tsv.

source

chicken (poultry host)

human (human host)

cattle (bovine host)


---

## Output

A TSV file with:

id    <source scores...>    best_hit    source_method    source_evidence    year    country

Example:

|run_acc| chicken|human|cattle|best_hit|source_method|source_evidence|year|country|
|-------|--------|-----|------|--------|-------------|---------------|----|-------|
|ERR001 |  0.85  |0.01 | 0.02 |chicken|nli||2019|United States|

- One score column per source, named using the text before the first `(` in `sources.tsv`; multi-word names are preserved
- `best_hit` is the name of the source with the highest score; ties use the first source in `sources.tsv`
- `--min-score` sets the minimum top score required for `best_hit`; lower-scoring rows are labelled `unknown` while their score columns are retained. The default is `0.2`.
- Full source labels, including parenthetical hints, are still used for classification
- Short source names must be nonempty, unique, and distinct from the ID, `best_hit`, `source_method`, `source_evidence`, `year`, and `country` column names
- Taxonomy-derived rows use `source_method=host_tax_id` and **all source scores are `NA`**, because no NLI inference was performed. They are not artificial probabilities and are not subject to `--min-score`.
- NLI rows use `source_method=nli`, including below-threshold `unknown` calls; their `source_evidence` is blank.
- year as 4-digit string
- country as normalized name

---

## Benchmark results

The following results compare `benchmark_classified.tsv` (using `sources.tsv`)
with `benchmark_classified_rich.tsv` (using `sources_rich.tsv`). Both were
evaluated against `benchmark_true_labels.tsv` with `--min-score 0.2` and local
host taxonomy enabled. Each run contains 255 `host_tax_id` assignments and
1,065 NLI assignments. Accuracy is recall: correct calls divided by the number
of true records for a source. Precision is correct calls divided by the number
of calls made for a source.

| Source | Standard accuracy | Standard precision | Rich accuracy | Rich precision |
|---|---:|---:|---:|---:|
| cat | 100.0% | 83.3% | 100.0% | 60.6% |
| cattle | 99.0% | 89.1% | 100.0% | 79.8% |
| chicken | 98.0% | 98.0% | 98.0% | 100.0% |
| dog | 96.0% | 100.0% | 100.0% | 97.1% |
| environment | 75.0% | 7.7% | 25.0% | 14.3% |
| goat | 97.0% | 100.0% | 94.0% | 97.9% |
| human | 76.0% | 100.0% | 72.0% | 100.0% |
| laboratory | 0.0% | N/A | 100.0% | 50.0% |
| other_animal | 57.0% | 98.3% | 51.0% | 98.1% |
| pig | 99.0% | 87.6% | 98.0% | 98.0% |
| sheep | 95.0% | 100.0% | 95.0% | 99.0% |
| turkey | 93.1% | 100.0% | 99.0% | 100.0% |
| unknown | 86.7% | 65.4% | 86.7% | 73.3% |
| wastewater | 100.0% | 25.0% | 100.0% | 27.8% |
| water | 73.9% | 59.6% | 83.7% | 56.6% |
| waterbird | 67.7% | 100.0% | 63.6% | 100.0% |
| wildbird | 81.0% | 94.2% | 98.0% | 95.1% |

The standard source list yields 1,140 correct calls of 1,320 (**86.4% overall
accuracy**) and assigns a non-unknown source to 1,190 records (90.2%). The rich
source list yields 1,158 correct calls (**87.7% overall accuracy**) and assigns
a non-unknown source to 1,204 records (91.2%). Their precision among assigned
records is 88.7% and 89.1%, respectively. The standard run has 130 NLI calls
below the cutoff; the rich run has 116.

### Confusion matrices

Rows are true labels and entries list `predicted_label: count`. Only nonzero
entries are shown. The `unknown` column is created by the 0.2 score cutoff,
not by a source candidate.

#### Standard source list

| True source | Predicted calls |
|---|---|
| cat | cat: 20 |
| cattle | cattle: 98; pig: 1 |
| chicken | chicken: 99; environment: 1; pig: 1 |
| dog | dog: 96; water: 4 |
| environment | cattle: 1; environment: 3 |
| goat | cat: 1; cattle: 1; goat: 97; other_animal: 1 |
| human | cattle: 1; human: 76; unknown: 10; water: 13 |
| laboratory | unknown: 1 |
| other_animal | environment: 4; other_animal: 57; pig: 10; unknown: 2; water: 27 |
| pig | cattle: 1; pig: 99 |
| sheep | cattle: 4; pig: 1; sheep: 95 |
| turkey | environment: 7; turkey: 94 |
| unknown | cat: 1; cattle: 4; chicken: 2; environment: 2; pig: 1; unknown: 85; water: 1; wildbird: 2 |
| wastewater | wastewater: 5 |
| water | environment: 7; unknown: 2; wastewater: 15; water: 68 |
| waterbird | cat: 2; environment: 15; unknown: 12; waterbird: 67; wildbird: 3 |
| wildbird | unknown: 18; water: 1; wildbird: 81 |

#### Rich source list

| True source | Predicted calls |
|---|---|
| cat | cat: 20 |
| cattle | cattle: 99 |
| chicken | chicken: 99; environment: 1; pig: 1 |
| dog | dog: 100 |
| environment | cattle: 1; environment: 1; water: 2 |
| goat | cat: 2; cattle: 2; goat: 94; other_animal: 1; unknown: 1 |
| human | cattle: 10; human: 72; unknown: 5; water: 13 |
| laboratory | laboratory: 1 |
| other_animal | cat: 8; cattle: 3; environment: 1; other_animal: 51; unknown: 7; water: 30 |
| pig | cattle: 1; pig: 98; sheep: 1 |
| sheep | dog: 3; goat: 1; pig: 1; sheep: 95 |
| turkey | turkey: 100; unknown: 1 |
| unknown | cattle: 8; environment: 2; laboratory: 1; unknown: 85; wildbird: 2 |
| wastewater | wastewater: 5 |
| water | unknown: 2; wastewater: 13; water: 77 |
| waterbird | cat: 3; environment: 2; goat: 1; unknown: 13; water: 14; waterbird: 63; wildbird: 3 |
| wildbird | unknown: 2; wildbird: 98 |

The new taxonomy stage resolves explicit hosts before NLI; those calls have
`source_method=host_tax_id` and `NA` score columns. The remaining errors are
concentrated in text-derived and ambiguous categories, especially environmental
and water-associated records.

### Experimental Mistral API classifier

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

Both specifications use Python 3.11, Transformers 5.x, PyTorch 2.x, and the
SentencePiece/Protobuf tokenizer dependencies. They are environment specifications,
not exact lockfiles. Taxonomy parsing, downloading, and the tests use the Python
standard library, so the taxonomy feature requires no additional packages.
The experimental Mistral classifier still requires the optional `mistralai` package.

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
to NLI. Omitting `--taxonomy-dir`, or supplying unreadable/malformed files,
produces a warning and retains NLI-only classification.

Anchors are resolved by scientific name: chicken (`Gallus gallus`), turkey
(`Meleagris gallopavo`), cattle (`Bos taurus`), sheep (`Ovis aries`), goat
(`Capra hircus`), human (`Homo sapiens`), dog (`Canis lupus familiaris`), cat
(`Felis catus`), and domestic pig (`Sus scrofa domesticus`). Descendants inherit
the most specific matching class. A class must exist in the selected sources
file before it can be emitted.

Other clearly identified non-bird animals can become `other_animal`.
Other birds fall back to NLI for the ecological `wildbird`/`waterbird` distinction.
Generic `Sus scrofa` and non-domestic descendants also fall back: taxonomy alone
may not distinguish wild boar from domestic pig. Broad ancestors such as
Mammalia or Metazoa remain unresolved, as do unknown/deleted IDs, broken
lineages, and non-animal hosts. If anchors are missing or ambiguous, generic
`other_animal` mapping is disabled to prevent false assignments. Environmental
classes (`water`, `wastewater`, `environment`, `laboratory`) remain text-derived.

For a sheep host, the output looks like this (all other source scores are also `NA`):

```text
run_accession   chicken turkey  pig cattle  sheep   ... best_hit source_method   source_evidence
SRR17929619     NA      NA      NA  NA      NA      ... sheep    host_tax_id     host_tax_id=9940; Ovis aries
```

Input order, year/country extraction, and NLI score column names are preserved.
Each run reports taxonomy calls, NLI calls, and NLI calls below the cutoff.
If all hosts resolve, the model is not loaded at all.

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

CPU execution explicitly uses float32 to avoid slow float16 inference.
GPU execution uses the model checkpoint's precision (`dtype="auto"`).

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
