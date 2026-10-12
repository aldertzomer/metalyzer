# Variant C: paired noisy-field exclusion benchmark

This is a fresh, twelve-run comparison of the same current Metalyzer taxonomy
configuration with original model text and default Variant C filtering. It
uses 1,320 accession-aligned requests from `benchmark.tsv` and
`benchmark_accessions.tsv`, scored against the unchanged
`benchmark_true_labels.tsv`. All methods ran on the RTX 4070 Ti SUPER.
Model weights were loaded locally; the four Mistral API runs used
`mistral-small-latest`, seed 12345, and the configured credential. The
[manifest](manifest.json) records every command, status, input and code hash,
taxonomy file provenance, GPU, and elapsed time. The selected BioSample
metadata was read once from the indexed local Parquet repository and cached
for these runs; every source prediction was computed afresh.

Both conditions explicitly enable `--other-animal-taxonomy`, leave configured
host aliases enabled, use NLI cutoff 0.2 and batch size 64, local Ministral
cutoff 0.75 and batch size 1, 300/2,000-character limits, NLI verification,
and date/country extraction. The animal rule remains opt-in in ordinary runs.
Filtering is the only intended classification-input difference.

## Main result

Each cell gives accuracy / assigned precision. Precision excludes predictions
of `unknown`. “Full” scores all 1,320 original references. “Adjusted”
excludes exactly the 44 validated CAGED old-goat/current-chicken
run/BioSample pairs, leaving 1,276; it retains all 41
human-evidence/reference-unknown records.

| Input | Method | Full unfiltered | Full filtered | Adjusted unfiltered | Adjusted filtered |
|---|---|---:|---:|---:|---:|
| TSV | NLI | 87.50% / 89.35% | 88.41% / 89.74% | 87.07% / 88.95% | 88.01% / 89.36% |
| TSV | Local Ministral | 92.05% / 98.61% | 92.80% / 98.88% | 91.77% / 98.55% | 92.55% / 98.83% |
| TSV | Mistral API | 97.35% / 97.18% | 97.95% / 97.82% | 97.26% / 97.08% | 97.88% / 97.73% |
| BioSample | NLI | 72.88% / 75.84% | 81.52% / 85.17% | 75.39% / 78.62% | 84.33% / 88.33% |
| BioSample | Local Ministral | 85.83% / 90.46% | 83.11% / 86.70% | 88.79% / 93.88% | 85.97% / 89.87% |
| BioSample | Mistral API | 88.41% / 88.30% | 88.94% / 88.96% | 91.46% / 91.42% | 92.01% / 92.11% |

Variant C helps BioSample NLI and harms BioSample local Ministral on this
benchmark. Do not treat the other small changes as independent validation or
claim that filtering improves every model. The dataset was used to inspect
and develop Metalyzer, and fixed-seed API calls can still vary. The previous
host-terminology TSV and BioSample local scores match the current unfiltered
accuracy to the displayed precision. Earlier API results predate this
taxonomy configuration and are not substituted for these fresh API calls.

## Paired changes and uncertainty

The difference and percentile bootstrap interval are percentage points of
full-benchmark accuracy (filtered minus unfiltered). McNemar's exact p-value
describes paired discordance for local methods; all inference is exploratory.
The [paired summary](evaluation/paired_summary.tsv) also reports adjusted
results and the 1,042 TSV / 759 BioSample records reaching the source model
separately.

| Input / method | Changed calls | Wrong → correct | Correct → wrong | Accuracy difference [95% paired CI] | McNemar p |
|---|---:|---:|---:|---:|---:|
| TSV NLI | 50 | 27 | 15 | +0.91 [−0.08, +1.89] | 0.088 |
| TSV Ministral | 42 | 25 | 15 | +0.76 [−0.15, +1.67] | 0.154 |
| TSV API | 17 | 12 | 4 | +0.61 [0.00, +1.21] | 0.077 |
| BioSample NLI | 169 | 115 | 1 | +8.64 [+7.12, +10.23] | <0.001 |
| BioSample Ministral | 93 | 11 | 47 | −2.73 [−3.86, −1.59] | <0.001 |
| BioSample API | 20 | 12 | 5 | +0.53 [−0.08, +1.14] | 0.143 |

Most changed calls occur only among records sent to the model. Every paired
deterministic `best_hit`, method, evidence and LLM score is identical; so are
year/country values and the normal output schema. The full
[per-record TSV comparisons](evaluation/paired_tsv_nli.tsv),
[TSV Ministral comparisons](evaluation/paired_tsv_llm.tsv),
[TSV API comparisons](evaluation/paired_tsv_mistral.tsv),
[BioSample NLI comparisons](evaluation/paired_biosample_nli.tsv),
[BioSample Ministral comparisons](evaluation/paired_biosample_llm.tsv), and
[BioSample API comparisons](evaluation/paired_biosample_mistral.tsv) include
run and BioSample accessions, truth, predictions, confidence and provenance.
The [paired summary](evaluation/paired_summary.tsv) gives wrong-to-different-
wrong, unchanged-correct/wrong, and transitions involving `unknown`.
Separate `evaluation/human_reference_unknown_*.tsv` tables keep the 41
historical-reference-unknown records visible.

## Taxonomy, text size, and throughput

TSV has 215 numeric-ID, 4 exact-name, 13 configured-alias and 46 opt-in
other-animal assignments: 278 deterministic assignments, 1,042 source-model
records. BioSample has 1, 401, 117 and 42 respectively: 561 deterministic
assignments, 759 source-model records. These counts are identical in both
conditions and in each method. NLI verification continues to cover final
assigned sources; its call count varies with the number of assigned model
predictions. [Pathway metrics](evaluation/pathway_metrics.tsv) give precision
and accuracy by source method in each run and denominator.

For TSV input, 3,924 administrative field occurrences are omitted across
1,320 records: `sample_accession` (1,320), `experiment_alias` (1,320), and
`library_name` (1,284). The model-record median falls from 555 to 461
characters. Neither TSV condition has a 2,000-character-truncated record or
measured NLI 512-token exposure.

For BioSample input, 27,719 administrative/technical field occurrences and
9,375 equivalent original-attribute occurrences are omitted across 1,320
records. The 759 source-model records account for 16,830 and 4,518 of those
omissions. Their median falls from 1,496 to 214 characters, and the 37
source-model records hitting the 2,000-character limit fall to zero. The
pre-truncation NLI token-pair exposure falls from 272/759 to 0/759. The
all-record median falls from 1,538 to 285 characters; all-record truncation
falls from 109 to zero. These counts come from
[TSV filtering statistics](filtered/tsv_nli.tsv.filtering.json),
[BioSample filtering statistics](filtered/biosample_nli.tsv.filtering.json),
[TSV excluded fields](filtered/tsv_nli.tsv.excluded_fields.tsv), and
[BioSample excluded fields](filtered/biosample_nli.tsv.excluded_fields.tsv).
Raw metadata and normal output columns are preserved.

| Input / method | Total seconds unfiltered → filtered | Source-model seconds | Verification calls unfiltered → filtered |
|---|---:|---:|---:|
| TSV NLI | 116.6 → 99.8 | 86.0 → 70.8 | 1,211 → 1,228 |
| TSV Ministral | 432.7 → 445.0 | 401.1 → 414.2 | 1,150 → 1,156 |
| TSV API | 107.0 → 105.5 | 71.4 → 70.2 | 1,242 → 1,236 |
| BioSample NLI | 275.1 → 118.4 | 176.7 → 37.2 | 1,246 → 1,227 |
| BioSample Ministral | 429.9 → 375.5 | 332.6 → 289.9 | 1,205 → 1,248 |
| BioSample API | 166.5 → 137.3 | 62.5 → 48.1 | 1,291 → 1,286 |

Source-stage times include loading and preprocessing. These timing
differences are observations on this machine, not controlled GPU-kernel
benchmarks. [All-run metrics](evaluation/summary.tsv) contain exact counts,
per-run times, calls, text lengths and token-limit exposure.

The four paid API runs each completed their expected number of source calls
with no retry: 1,042 calls per TSV run and 759 per BioSample run. API-reported
total tokens fell from 951,636 to 902,060 for TSV and from 921,385 to 590,698
for BioSample. These are observed usage counts, not a fixed monetary estimate.
The individual `.log` files record request, retry, prompt and completion-token
counts without credentials. API variability limits causal interpretation of
small paired changes.

The filtered TSV Ministral run completed and its output, manifest entry,
filtering sidecar and evaluation are intact. An accidental concurrent restart
truncated its early progress log; the committed `.log` preserves the completed
run from batch 522 onward and marks that gap explicitly. This does not affect
the 1,320 scored predictions.

## Class metrics, confusion matrices, and examples

Each run has [per-class recall, precision and one-vs-rest accuracy](evaluation/),
a full count matrix and a true-source-normalized percentage matrix under
`evaluation/<input>/<method>/<condition>/<denominator>/`. The 48 SVG
visualizations are under `figures/original/` and
`figures/metadata_version_adjusted/`. The table links the adjusted normalized
figures; switch `percent` to `counts`, or
`metadata_version_adjusted` to `original`, for the corresponding view.

| Input / method | Unfiltered adjusted matrix | Filtered adjusted matrix |
|---|---|---|
| TSV NLI | [percent](figures/metadata_version_adjusted/tsv_nli_unfiltered_percent.svg) | [percent](figures/metadata_version_adjusted/tsv_nli_filtered_percent.svg) |
| TSV Ministral | [percent](figures/metadata_version_adjusted/tsv_llm_unfiltered_percent.svg) | [percent](figures/metadata_version_adjusted/tsv_llm_filtered_percent.svg) |
| TSV API | [percent](figures/metadata_version_adjusted/tsv_mistral_unfiltered_percent.svg) | [percent](figures/metadata_version_adjusted/tsv_mistral_filtered_percent.svg) |
| BioSample NLI | [percent](figures/metadata_version_adjusted/biosample_nli_unfiltered_percent.svg) | [percent](figures/metadata_version_adjusted/biosample_nli_filtered_percent.svg) |
| BioSample Ministral | [percent](figures/metadata_version_adjusted/biosample_llm_unfiltered_percent.svg) | [percent](figures/metadata_version_adjusted/biosample_llm_filtered_percent.svg) |
| BioSample API | [percent](figures/metadata_version_adjusted/biosample_mistral_unfiltered_percent.svg) | [percent](figures/metadata_version_adjusted/biosample_mistral_filtered_percent.svg) |

The [error examples](evaluation/error_examples.tsv) contain complete
unfiltered and filtered model text for representative wrong-to-correct,
correct-to-wrong and `unknown` transitions in each method. For BioSample
NLI, `SAMN06660983` and `SAMN07633659` change from incorrect
`environment` to correct `chicken` calls; their original text contains the
`Pathogen.env` registration profile. In BioSample Ministral,
`SAMN13392879` and `SAMN13392878` change from correct `dog` calls to
`unknown`. Multiple fields change together, so the examples do not isolate
one field as the cause. The full per-class and paired tables show where
filtering helps and where it loses useful context.

Reproduce the benchmark and evaluation from the repository root:

```bash
python benchmark_smoke/run_noisy_field_benchmarks.py
python benchmark_smoke/evaluate_noisy_fields.py
```

The runner skips only completed runs whose input and code signature matches
the manifest. Use `--work-dir` for a separate experiment.
