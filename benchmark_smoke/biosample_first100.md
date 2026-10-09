# BioSample list-input smoke run

The first 100 **rows** of repository `benchmark.tsv` were read with Python's
CSV reader and their `sample_accession` values passed together to `--biosample`,
in order, retaining duplicates. The extracted values are saved in
`biosample_first100_accessions.txt`.

Run on 2026-10-09 with the local 60,153,296-row BioSample Parquet repository:

```bash
PATH="$CONDA_PREFIX/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python metalyzer.py --biosample $(cat benchmark_smoke/biosample_first100_accessions.txt) \
  --sources sources.tsv --out benchmark_smoke/biosample_first100_llm.tsv \
  --method llm --taxonomy-dir /home/zomer007/metalyzer/taxonomy
```

The environment's CUDA 12.9 `ptxas` must precede the server's CUDA 11.2 assembler
on PATH; CUDA 11.2 does not support this GPU's `sm_89` architecture. This is a
runtime environment adjustment, with no model or confidence-setting changes.
No network archive download or per-accession network lookup was performed.

Results: 100 requested/resolved rows; 149 extracted metadata columns; approximately
22 seconds for indexed lookup. All rows used Ministral with the default 0.75
threshold, with 100 retained assignments, no abstentions or invalid outputs, and
100 NLI verification scores. These BioSample records contained no `host_tax_id`
column, so there were no deterministic taxonomy assignments. Predictions were
20 cat, 79 cattle, and one environment. These are smoke-run counts, not an accuracy
measurement against reference labels.

The saved regular TSV was checked for requested accession order, row count and
normal Metalyzer score/evidence/year/country columns. Its normal log records
lookup provenance, cached-model configuration, taxonomy and run totals.
