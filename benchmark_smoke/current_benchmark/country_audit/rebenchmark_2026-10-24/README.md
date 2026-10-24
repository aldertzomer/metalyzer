# Current country-only benchmark (2026-10-24)

This run uses repository revision `c12bcce4c729c32f6ce3856393443b31278adbe7`, the current 1,275-run ISO2 reference, historical ATB TSV metadata, and the existing cached BioSample frame. It calls only `modules.country.run`; no NLI, Ministral, or Mistral API inference was used. Inputs and runner hashes are in [summary.json](summary.json).

The 87 runs with `final_country=unknown` are **unscored**, following the current reference policy. Accuracy is calculated over the other 1,188 sequencing runs; the full per-run tables retain all 1,275 records.

| Input | Correct / scored | Accuracy | Scored mismatches | Predicted `unknown` among scored |
| --- | ---: | ---: | ---: | ---: |
| Historical ATB TSV | 1,169 / 1,188 | 98.4% | 19 | 19 |
| Current BioSample cache | 1,188 / 1,188 | 100.0% | 0 | 0 |

Review the [ATB per-run results](atb_countries.tsv), [ATB mismatches](atb_mismatches.tsv), [BioSample per-run results](biosample_countries.tsv), and [BioSample mismatches](biosample_mismatches.tsv). The latter contains only its header because no scored BioSample run disagreed with the reference. Raw geographic fields are included in both per-run tables and mismatch tables. `correct` is blank for unscored rows.

From the repository root, rerun with the project Python environment:

```bash
PYTHONPATH=. python benchmark_smoke/current_benchmark/country_audit/rebenchmark_2026-10-24/run.py
```

The reference includes metadata-based adjudications, so the BioSample agreement is not an independent estimate of country-extraction accuracy. The older TSVs in the parent directory remain frozen historical outputs from an earlier reference version.
