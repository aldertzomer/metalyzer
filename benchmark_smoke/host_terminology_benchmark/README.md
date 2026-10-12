# Full local BioSample host-terminology benchmarks

Fresh NLI and Ministral runs on 1,320 local BioSample requests with the combined
configuration: exact taxonomy, configured aliases, and explicitly enabled
`--other-animal-taxonomy`. The latter remains disabled by default. All settings
match earlier runs: NLI batch 64 and cutoff 0.2, Ministral batch 1 and cutoff
0.75, 300/2,000-character limits, verification and year/country extraction.
CUDA was verified on an RTX 4070 Ti SUPER (16 GB). Cached model weights were
loaded offline. No API benchmark was run and no old results were replaced.

| Method / version | Accuracy, 1,320 | Assigned precision | Adjusted accuracy, 1,276 | Adjusted assigned precision |
|---|---:|---:|---:|---:|
| Saved NLI | 60.98% | 64.43% | 63.01% | 66.75% |
| New NLI | 72.88% | 75.84% | 75.39% | 78.62% |
| Saved Ministral | 85.83% | 90.60% | 88.79% | 94.04% |
| New Ministral | 85.83% | 90.46% | 88.79% | 93.88% |

The adjusted evaluation excludes only the 44 documented CAGED pairs validated
against historical goat/current chicken metadata. Original predictions and
reference labels are unchanged. The 41 human-evidence/reference-unknown rows
remain visible in separate diagnostic TSVs and remain in both denominators.

All 759 pending records received the same model labels as before. Aliases
improve 21 NLI decisions and two Ministral decisions with no losses. The
additional animal rule improves 34 NLI decisions but loses the wild-goat row
`SAMN08235377` (reference goat versus taxonomically resolved `Capra aegagrus`).
Ministral has no gains on these animals, and its other newly incorrect reference
match is `SAMEA7632583`: host name resolves human while reference remains unknown.
Its two gains and two losses leave overall accuracy unchanged.

The 117 aliases are all reference-consistent; the additional animal rule is
41/42 consistent and remains opt-in. The CPU audit covers default alias-only
resolution, while these full model scores measure the combined experiment.

Source inference used 759 records per backend, 561 fewer (42.5%) than the saved
runs and 159 fewer (17.3%) than the prior exact-host-name resolver. NLI source
stage: 175.8 s; verification: 20.1 s; total: 414.2 s. Ministral source stage:
331.2 s; verification: 21.3 s; total: 570.5 s. Stage times include weight
loading and tokenization; they are not GPU kernel-only times. Old total timings
were 539.9 s and 785.9 s, but differences in hardware/cache prevent attributing
all timing changes to this implementation. Verification still covers final
assignments and is not included in the source-model call reduction.

- [Fresh NLI output](benchmark_biosample_nli_host_terminology.tsv)
- [Fresh Ministral output](benchmark_biosample_llm_host_terminology.tsv)
- [Commands, GPU details, input/code hashes](manifest.json)
- [Original and adjusted evaluation summary](evaluation/summary.tsv)
- [Pathway precision and accuracy](evaluation/pathway_metrics.tsv)
- [Performance](evaluation/performance.tsv)
- [All per-record old/new comparisons](evaluation/prediction_comparison.tsv)
- [Change counts by pathway](evaluation/prediction_change_summary.tsv)
- [NLI adjusted confusion matrix](evaluation/metadata_version_adjusted/nli_confusion_percent.svg)
- [Ministral adjusted confusion matrix](evaluation/metadata_version_adjusted/llm_confusion_percent.svg)

Reproduce from the repository root:

```bash
python benchmark_smoke/run_host_terminology_benchmarks.py --other-animal-taxonomy
python benchmark_smoke/evaluate_host_terminology.py
```

Use a separate `--out-dir` for different settings; completed results are reused
only when the input, taxonomy and code signature matches.
