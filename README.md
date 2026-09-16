# Metadata Classification Pipeline (Zero-shot + Deterministic Parsing)

This repository contains a Python pipeline to classify biological metadata records using a combination of:

- Zero-shot classification (for source/host)
- Deterministic parsing (for year and country)

The pipeline is designed for large-scale datasets (e.g. ENA/SRA metadata) with heterogeneous formatting.

---

## Overview

For each metadata row, the pipeline performs:

### 1. Source classification (NLI)
Uses a zero-shot model:

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

id    <source scores...>    best_hit    year    country

Example:

|run_acc| chicken|human|cattle|best_hit|year|country|
|-------|--------|-----|------|--------|----|-------|
|ERR001 |  0.85  |0.01 | 0.02 |chicken|2019|United States|

- One score column per source, named using the text before the first `(` in `sources.tsv`; multi-word names are preserved
- `best_hit` is the name of the source with the highest score; ties use the first source in `sources.tsv`
- `--min-score` sets the minimum top score required for `best_hit`; lower-scoring rows are labelled `unknown` while their score columns are retained. The default is `0.2`.
- Full source labels, including parenthetical hints, are still used for classification
- Short source names must be nonempty, unique, and distinct from the ID, `best_hit`, `year`, and `country` column names
- year as 4-digit string
- country as normalized name

---

## Benchmark results

The following results compare `benchmark_classified.tsv` (using `sources.tsv`)
with `benchmark_classified_rich.tsv` (using `sources_rich.tsv`). Both were
evaluated against `benchmark_true_labels.tsv` with `--min-score 0.2`.
Accuracy is recall: correct calls divided by the number of true records for a
source. Precision is correct calls divided by the number of calls made for a
source. The benchmark contains no true `environment` or `laboratory` records,
so their recall is not applicable and every call to either is a false positive.

| Source | Standard accuracy | Standard precision | Rich accuracy | Rich precision |
|---|---:|---:|---:|---:|
| cat | 95.0% | 63.3% | 100.0% | 54.1% |
| cattle | 98.0% | 77.8% | 97.0% | 99.0% |
| chicken | 99.0% | 92.5% | 72.0% | 97.3% |
| dog | 95.0% | 100.0% | 100.0% | 95.2% |
| environment | — | 0.0% | — | 0.0% |
| goat | 98.0% | 96.1% | 94.0% | 87.9% |
| human | 75.0% | 100.0% | 72.0% | 100.0% |
| laboratory | — | 0.0% | — | 0.0% |
| other_animal | 35.0% | 100.0% | 37.0% | 100.0% |
| pig | 100.0% | 82.0% | 100.0% | 94.3% |
| sheep | 61.0% | 100.0% | 44.0% | 100.0% |
| turkey | 99.0% | 100.0% | 75.0% | 100.0% |
| unknown | 74.0% | 46.5% | 90.0% | 70.3% |
| water | 84.0% | 100.0% | 87.0% | 88.8% |
| waterbird | 82.0% | 86.3% | 63.0% | 100.0% |
| wildbird | 79.0% | 79.0% | 98.0% | 65.8% |

The standard source list yields 83.2% overall accuracy and assigns a
non-unknown source to 88.0% of records. The rich source list yields 79.5%
overall accuracy and assigns a non-unknown source to 90.3% of records.

### Confusion matrices

Rows are true labels and entries list `predicted_label: count`. Only nonzero
entries are shown. The `unknown` column is created by the 0.2 score cutoff,
not by a source candidate.

#### Standard source list

| True source | Predicted calls |
|---|---|
| cat | cat: 19; pig: 1 |
| cattle | cattle: 98; pig: 2 |
| chicken | chicken: 99; pig: 1 |
| dog | cattle: 1; dog: 95; unknown: 4 |
| goat | cat: 1; cattle: 1; goat: 98 |
| human | cattle: 13; human: 75; unknown: 12 |
| other_animal | cat: 5; environment: 8; other_animal: 35; pig: 10; unknown: 39; wildbird: 3 |
| pig | pig: 100 |
| sheep | cat: 1; cattle: 4; pig: 7; sheep: 61; unknown: 1; waterbird: 13; wildbird: 13 |
| turkey | environment: 1; turkey: 99 |
| unknown | cat: 1; cattle: 6; chicken: 5; environment: 2; laboratory: 9; pig: 1; unknown: 74; wildbird: 2 |
| water | cattle: 3; chicken: 1; environment: 10; unknown: 2; water: 84 |
| waterbird | cat: 3; goat: 4; unknown: 8; waterbird: 82; wildbird: 3 |
| wildbird | chicken: 2; unknown: 19; wildbird: 79 |

#### Rich source list

| True source | Predicted calls |
|---|---|
| cat | cat: 20 |
| cattle | cattle: 97; dog: 1; pig: 2 |
| chicken | chicken: 72; environment: 26; pig: 1; wildbird: 1 |
| dog | dog: 100 |
| goat | cat: 4; cattle: 1; goat: 94; unknown: 1 |
| human | environment: 3; human: 72; unknown: 14; water: 11 |
| other_animal | cat: 11; environment: 42; other_animal: 37; unknown: 7; wildbird: 3 |
| pig | pig: 100 |
| sheep | dog: 4; goat: 12; pig: 1; sheep: 44; unknown: 1; wildbird: 38 |
| turkey | environment: 25; turkey: 75 |
| unknown | chicken: 2; environment: 2; laboratory: 2; pig: 2; unknown: 90; wildbird: 2 |
| water | environment: 11; unknown: 2; water: 87 |
| waterbird | cat: 2; environment: 16; goat: 1; unknown: 11; waterbird: 63; wildbird: 7 |
| wildbird | unknown: 2; wildbird: 98 |

The rich hints increase calls to `environment` from 21 to 125, despite the
benchmark containing no true environment records. This accounts for much of
the decline in rich-list accuracy, especially for chicken, turkey,
other_animal, sheep, water, and waterbird.

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

`benchmark_mistral.tsv` was evaluated against `benchmark_true_labels.tsv`
using `sources_mistral.tsv`. It contains all 1,320 benchmark records and has
1,260 correct calls: **95.5% overall accuracy**. Accuracy is recall: correct
calls divided by the number of true records for a source. Precision is correct
calls divided by the number of calls made for a source.

| Source | True rows | Called rows | Correct | Accuracy | Precision |
|---|---:|---:|---:|---:|---:|
| cat | 20 | 20 | 20 | 100.0% | 100.0% |
| cattle | 100 | 99 | 99 | 99.0% | 100.0% |
| chicken | 100 | 106 | 100 | 100.0% | 94.3% |
| dog | 100 | 100 | 100 | 100.0% | 100.0% |
| environment | 0 | 14 | 0 | — | 0.0% |
| goat | 100 | 100 | 100 | 100.0% | 100.0% |
| human | 100 | 127 | 100 | 100.0% | 78.7% |
| laboratory | 0 | 2 | 0 | — | 0.0% |
| other_animal | 100 | 86 | 86 | 86.0% | 100.0% |
| pig | 100 | 102 | 100 | 100.0% | 98.0% |
| sheep | 100 | 100 | 100 | 100.0% | 100.0% |
| turkey | 100 | 101 | 100 | 100.0% | 99.0% |
| unknown | 100 | 69 | 69 | 69.0% | 100.0% |
| water | 100 | 92 | 92 | 92.0% | 100.0% |
| waterbird | 100 | 94 | 94 | 94.0% | 100.0% |
| wildbird | 100 | 108 | 100 | 100.0% | 92.6% |

The benchmark has no true `environment` or `laboratory` records, so every call
to either is counted as a false positive.

| True source | Mistral predicted calls |
|---|---|
| cat | cat: 20 |
| cattle | cattle: 99; environment: 1 |
| chicken | chicken: 100 |
| dog | dog: 100 |
| goat | goat: 100 |
| human | human: 100 |
| other_animal | human: 12; other_animal: 86; pig: 1; wildbird: 1 |
| pig | pig: 100 |
| sheep | sheep: 100 |
| turkey | turkey: 100 |
| unknown | chicken: 6; environment: 9; human: 11; laboratory: 2; pig: 1; unknown: 69; wildbird: 2 |
| water | environment: 4; human: 4; water: 92 |
| waterbird | turkey: 1; waterbird: 94; wildbird: 5 |
| wildbird | wildbird: 100 |

---

## Installation

Requirements:

- Python ≥ 3.10
- Conda environment recommended

Install from conda

```conda env create -f environment.yml```

Install dependencies by hand

```conda install -c conda-forge pandas pycountry```
```pip install transformers torch```

---

## Usage

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
