# Full TSV host terminology benchmarks

Fresh local NLI and Ministral runs on all 1,320 rows of `benchmark.tsv`, keeping
the original metadata column order and using sequencing run IDs. No BioSample
metadata is read or merged. Earlier predictions, inputs and labels remain
untouched. The combined configuration matches the new BioSample experiment:
exact taxonomy, operational aliases, and explicitly enabled additional animal
assignments. The animal rule remains disabled by default in normal runs.

CUDA was verified on an RTX 4070 Ti SUPER (16 GB). Cached weights were loaded
offline. NLI batch size 64, cutoff 0.2; Ministral batch size 1, cutoff 0.75;
record limits 300/2,000 characters; normal verification and date/country
extraction enabled. No paid API run was made.

| Method / version | Accuracy, 1,320 | Assigned precision | Adjusted accuracy, 1,276 | Adjusted assigned precision |
|---|---:|---:|---:|---:|
| Old NLI | 86.14% | 88.23% | 85.66% | 87.78% |
| New NLI | 87.50% | 89.35% | 87.07% | 88.95% |
| Old Ministral | 92.05% | 98.69% | 91.77% | 98.64% |
| New Ministral | 92.05% | 98.61% | 91.77% | 98.55% |

There are 278 deterministic assignments: 215 numeric IDs, four exact names,
13 aliases (eight Canine→dog, five Ovine→sheep), and 46 additional animals.
Alias assignments are all reference-consistent. Animal assignments match 45/46
references, with the known `Capra aegagrus` / goat disagreement. The default
alias-only audit has 232 deterministic records, excluding the opt-in rule.

All 1,042 remaining model decisions have the same labels as before. Aliases
correct five old NLI decisions and one Ministral decision, with no new errors.
The animal rule corrects 14 NLI decisions but introduces the wild-goat
disagreement. Ministral's alias gain is offset by that same animal disagreement,
so combined accuracy remains unchanged. Exact names add coverage without
changing the four already-correct source labels.

Adjusted scoring excludes precisely the 44 documented CAGED run/BioSample pairs.
These rows retain older ENA goat metadata and remain correct under TSV input.
The exclusion permits comparison with current BioSample's chicken metadata;
it does not relabel anything. The separate 41 human-evidence/reference-unknown
rows remain scored and have separate diagnostic tables.

Source-model records decrease from 1,105 to 1,042 (63 fewer, 5.7%). NLI source
stage: 85.5 s, verification: 7.3 s, total: 114.3 s. Ministral source stage:
442.0 s, verification: 8.9 s, total: 472.6 s. Stage times include weight
loading and tokenization. The older total times were 113.1 s and 452.4 s;
this rerun did not demonstrate an overall speedup. Hardware/cache differences
and taxonomy-loading overhead prevent attributing timing differences solely
to the smaller inference workload.

- [Fresh NLI output](benchmark_tsv_nli_host_terminology.tsv)
- [Fresh Ministral output](benchmark_tsv_llm_host_terminology.tsv)
- [Input/code hashes, GPU details and commands](manifest.json)
- [Original/adjusted scores](evaluation/summary.tsv)
- [Pathway metrics](evaluation/pathway_metrics.tsv)
- [Prediction changes](evaluation/prediction_comparison.tsv)
- [Change counts by pathway](evaluation/prediction_change_summary.tsv)
- [Performance](evaluation/performance.tsv)
- [NLI adjusted confusion matrix](evaluation/metadata_version_adjusted/nli_confusion_percent.svg)
- [Ministral adjusted confusion matrix](evaluation/metadata_version_adjusted/llm_confusion_percent.svg)

Reproduce from the repository root:

```bash
python benchmark_smoke/audit_host_terminology.py --input-mode metadata
python benchmark_smoke/run_host_terminology_benchmarks.py --input-mode metadata --other-animal-taxonomy
python benchmark_smoke/evaluate_host_terminology.py --input-mode metadata
```

Taxonomy defaults to `/home/zomer007/metalyzer/taxonomy`; override it when needed.
Use a separate output directory when changing settings.
