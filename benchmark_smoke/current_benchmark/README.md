# Current 1,275-run benchmark

This is the current benchmark. The historical AllTheBacteria (ATB) ENA export
was downloaded on 6 May 2025; NCBI BioSample records subsequently changed.
The reference revision excludes 44 documented CAGED run/BioSample pairs with
historical goat and current chicken hosts, and `ERR2197811` with historical
duck and current chicken evidence. No excluded run is scored. Of the 1,275
retained runs, 57 source labels were corrected by metadata adjudication:
41 unknown→human, four unknown→chicken, one unknown→cattle, one
unknown→water, one other_food→chicken, and nine water→environment. Country
values have been encoded as ISO 3166-1 alpha-2; year labels were preserved. Some adjudications were model-informed, so
this is a **working reference**, not independent validation data.

The [dataset manifest](inputs/dataset_manifest.json) provides the source and
derived SHA256 checksums, accession coverage, and exclusion checks. The
[run manifest](manifest.json) records settings, code/taxonomy signatures,
GPU details, commands, timings, and output hashes. The [API plan](api_plan.json)
counts the exact source-model requests before any paid run. Results are always
joined by sequencing-run accession through the
[run-to-BioSample mapping](inputs/run_to_biosample.tsv); the repeated
BioSample accession is retained for its two distinct runs.

## Settings and execution

The 12-run matrix crosses ATB TSV/current BioSample, NLI/local Ministral/Mistral
API, and Variant C noisy-field filtering on/off. Host aliases are enabled and
the optional conservative `--other-animal-taxonomy` rule is explicitly on in
all runs. The off condition passes `--disable-noisy-field-exclusion`; source
taxonomy and metadata are otherwise identical. DeBERTa uses a 0.2 threshold
and batch size 64; local Ministral uses 0.75 and batch size 1; API requests
use `mistral-small-latest` with seed 12345. NLI source verification, date and
country extraction are enabled; new country outputs use ISO 3166-1 alpha-2 (or `unknown`). Values are limited to 300 characters and
records to 2,000 characters. The command list and package versions are in the
run manifest. The API phase requires explicit approval for fresh paid calls.
The evaluation estimates API spend from reported token usage at Mistral's
[published Small 4 standard rates](https://docs.mistral.ai/inference/pricing)
of US$0.15 per million input tokens and US$0.60 per million output tokens
(checked 13 October 2026). This is an estimate because `mistral-small-latest`
is an alias and billing can differ.
On this workstation, local Ministral required
`TRITON_PTXAS_PATH=/home/zomer007/miniforge3/envs/metalyzer/bin/ptxas`
because the default CUDA 11.2 assembler does not support the GPU's `sm_89`
architecture; the Conda assembler reports CUDA 12.9. This environment setting
only selects the kernel compiler and does not change model prompts or scores.

```bash
python benchmark_smoke/run_current_benchmark.py --phase prepare
python benchmark_smoke/run_current_benchmark.py --phase local
# Run only after approving fresh paid API calls:
python benchmark_smoke/run_current_benchmark.py --phase api --confirm-paid-api
python benchmark_smoke/evaluate_current_benchmark.py
```

The runner resumes verified completed outputs. It rejects reuse if a source
input, source definitions, taxonomy dump, BioSample store, model setting or
classification code signature changes. The original 1,320-row input files,
truth history, and previous prediction files are unchanged. The synchronized
inputs are [historical ATB TSV](inputs/benchmark_atb_1275.tsv),
[BioSample requests](inputs/benchmark_accessions_1275.tsv), and a
[copy of the current truth set](inputs/benchmark_true_labels_1275.tsv).

## Results

<!-- BEGIN CURRENT_REPORT_TABLE -->

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

<!-- END CURRENT_REPORT_TABLE -->

<!-- BEGIN CURRENT_REPORT_ANALYSIS -->

| Input | Method | Δ accuracy | Δ assigned precision | Wrong → correct | Correct → wrong | Median model-text reduction | Δ runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| ATB TSV | NLI | +0.0% | -1.0% | 13 | 13 | 94 chars | -14.3 s |
| ATB TSV | Ministral | +0.6% | +0.2% | 22 | 14 | 94 chars | -5.9 s |
| ATB TSV | Mistral API | +0.2% | +0.2% | 8 | 5 | 94 chars | -10.2 s |
| BioSample | NLI | +8.9% | +9.8% | 115 | 1 | 1282 chars | -153.5 s |
| BioSample | Ministral | -2.9% | -4.5% | 14 | 51 | 1282 chars | -59.4 s |
| BioSample | Mistral API | +0.3% | +0.5% | 9 | 5 | 1282 chars | -24.6 s |

The completed API runs made 3,715 HTTP requests, including 115 retries. Reported token usage gives an estimated US$0.52 at the published Small 4 standard rates; this is not a billing statement.

<!-- END CURRENT_REPORT_ANALYSIS -->

## Detailed evaluation

[Summary metrics](evaluation/summary.tsv), [run outcomes](evaluation/outcomes.tsv),
[paired filtering metrics](evaluation/paired_summary.tsv), and
[taxonomy-pathway metrics](evaluation/pathway_metrics.tsv) are the compact
machine-readable tables. Each completed run has its own per-class metrics,
confusion matrix counts, normalized confusion matrix, and aligned per-record
table under `evaluation/<input>/<method>/<condition>/`. Corresponding count
and percentage heatmaps are in [figures](figures/). The
`evaluation/paired_<input>_<method>.tsv` files compare conditions by run ID;
`evaluation/new_errors_<input>_<method>.tsv` lists newly wrong runs. The
[example table](evaluation/error_examples.tsv) contains selected changed
predictions and full rendered model text in both conditions. Each output TSV
also has a filtering audit and field-exclusion summary alongside its run log.

The evaluation checks that paired conditions have the same row order, complete
1,275-run coverage, unchanged raw metadata and output schema, identical
country/year values, and identical deterministic taxonomy assignments.
BioSample results use the explicit run-to-accession mapping and never rely on
accidental row alignment. Accuracy includes `unknown` reference labels;
assigned precision excludes predictions of `unknown`. Paired confidence
intervals and McNemar tests are descriptive because this working benchmark
has informed pipeline development. API output can vary even with a fixed
seed, so paired differences cannot all be attributed to filtering.


## Country-code migration (October 2026)

The authoritative `benchmark_true_labels.tsv` and its 1,275-row synchronized copy now use ISO 3166-1 alpha-2 codes (e.g., `USA → US`, `United Kingdom → GB`, `Turkey → TR`). Unknown locations remain `unknown`. Historical ATB input metadata is unchanged. Previously committed prediction outputs are historical snapshots whose `country` values still use the old representation; do not compare them directly with the converted reference. Source-classification predictions remain independently interpretable. Recompute or convert only the country column before a new country-accuracy evaluation; rerunning a source classifier is unnecessary for that conversion. The existing run manifest and output hashes remain a record of the historical run, so a fresh benchmark work directory is needed for a new benchmark under the updated code.
