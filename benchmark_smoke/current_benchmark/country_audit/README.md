# Country-only audit

Run on 2026-10-24 from repository revision `f963e902368c8c90ab76ef4613587a1a20810cfb`, using the project conda environment (Python 3.14.3). This applies `modules.country.run` directly to the 1,275 aligned ATB TSV records and the existing BioSample pickle cache. It does not run NLI, Ministral, or the Mistral API. Reference countries are the current `final_country` ISO 3166-1 alpha-2 labels, aligned by `run_accession` through `run_to_biosample.tsv`.

| Input | Correct | Agreement | Mismatches |
| --- | ---: | ---: | ---: |
| Historical ATB TSV | 1,234 / 1,275 | 96.8% | 41 |
| Current BioSample cache | 1,244 / 1,275 | 97.6% | 31 |

The complete per-run tables are [ATB countries](atb_countries.tsv) and [BioSample countries](biosample_countries.tsv). The [ATB mismatches](atb_mismatches.tsv) and [BioSample mismatches](biosample_mismatches.tsv) retain the original geographic fields for individual review. The reference is a working benchmark that includes manually adjudicated labels; a mismatch does not by itself establish a parser error.

Input SHA-256 checksums:

| Input | SHA-256 |
| --- | --- |
| `inputs/benchmark_true_labels_1275.tsv` | `4f602c6caea98aec93bfca07b2aae8f334d2d44fd631e93292e92f0225527e95` |
| `inputs/run_to_biosample.tsv` | `8710bafbe991d370a6e454b556407db2f4ddfe202ab3d1dd29ec362f8e1418dd` |
| `inputs/benchmark_atb_1275.tsv` | `def594eb97b6fa6b1e0cede25dc68bc811e3cb5663e219d4600eb20467e21786` |
| `input_cache/biosample.pkl` | `c4d97b27d9ba45ddf0235e410fb7dd0346a58c1e199a57b775877fbe5089cfad` |

## Reference revision after this audit (2026-10-24)

The four `*_countries.tsv` and `*_mismatches.tsv` files above are **frozen historical audit output** and retain the original reference labels. Following a separate user-requested review, 23 records where *both* independent metadata sources produced `unknown` were conservatively changed to `unknown` in the authoritative benchmark reference and its synchronized copy. Their previous values, per-accession source fields, and justification are recorded in [reference_country_changes_2026-10-24.tsv](reference_country_changes_2026-10-24.tsv).

Against the **revised** references, those *same frozen predictions* would yield 1,257/1,275 ATB matches (98.6%) and 1,267/1,275 BioSample matches (99.4%); the mismatch files themselves were not modified. These values are re-scored agreement with an updated reference, **not an improvement in country extraction**. The existing pipeline-run manifest still refers to its original historical predictions; new evaluation runs must use compatible manifests and avoid overwriting the historical snapshot.

## Eight explicit geographic conflicts reviewed (2026-10-24)

See [review_eight_conflicts_2026-10-24.tsv](review_eight_conflicts_2026-10-24.tsv) for accession-by-accession evidence and decisions.

- **Six Brazil/USA conflicts:** original ATB metadata records `Brazil` for CCAMP-991, CCAMP-980, CCAMP-730, CCAMP-162, CCAMP-1466 and CCAMP-764, while the cached BioSample `geo_loc_name` is `USA`. The 2018 genome announcement explicitly describes this CCAMP collection as C. jejuni isolated in Brazil and lists all six strains. Preserve `BR` in the benchmark references. Do not add isolate-specific overrides to the generic country parser. Source: https://doi.org/10.1128/genomea.00250-18 .
- **One UK sample (ERR946747 / SAMEA3487068):** BioSample geographic location is `United Kingdom`; study ERP010862/PRJEB9722 is linked to a paper on retail chicken produced in Northern Ireland. Revise reference `unknown` to `GB`. Source: https://doi.org/10.1186/s13099-015-0067-z .
- **One US/Canada conflict (SRR23525304 / SAMN33356324):** ATB gives `USA:NY` while BioSample gives `Canada`. The sample is comminuted turkey from a USDA-FSIS sequencing study, but that does not establish whether the contradictory locations refer to sample collection, origin of the food, or another provenance dimension. Keep the historical `US` reference provisionally and flag for source verification; do not assert that either metadata field is wrong.

The review changed exactly one additional reference label (`ERR946747: unknown -> GB`) after the 23 unsupported references had been changed to `unknown`. The frozen predictions therefore would match the final reviewed reference in **1,256/1,275 ATB rows (98.5%)** and **1,268/1,275 BioSample rows (99.5%)**. These are rescored historical predictions; the country extraction itself was not rerun. Remaining disagreements: 19 ATB, 7 BioSample. All raw historical audit TSVs above are intentionally retained without rewriting.
