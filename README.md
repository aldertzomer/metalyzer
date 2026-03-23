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

id    <source scores...>    year    country

Example:

ERR001    0.85    0.01    0.02    2019    United States

- One column per source label
- year as 4-digit string
- country as normalized name

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

python metalyzer.py \
  --metadata metadata.tsv \
  --sources sources.tsv \
  --out classified.tsv \
  --id-col run_acc \
  --device 0 \
  --batch-size 64

---

## Performance Notes

- Source classification runs on GPU
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
