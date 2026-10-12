# Offline host-name taxonomy audit

This audit exercises the new resolver on all 1,320 requests through the local BioSample module.
No NLI, Ministral, Mistral API, or verification inference was performed. Raw metadata was checked for changes.

| Audit measure | Records |
|---|---:|
| Usable numeric host taxonomy ID | 2 |
| Exact scientific-name match | 479 |
| Any unambiguous scientific/common/synonym name match | 616 |
| Missing/unresolved/ambiguous name | 704 |
| Ambiguous exact name | 0 |
| Numeric/textual conflict | 0 |
| Explicit numeric assignment | 1 |
| Exact host-name assignment | 401 |
| Total deterministic assignments | 402 |
| Model fallback required | 918 |

Resolution counts overlap: a matched name need not belong to a configured source lineage.
The source vocabulary remains the authority; unconfigured wildlife stays unresolved.

Actual example `SAMN04435360` resolves to turkey (`host_name`, taxid 9103).
Actual `SAMN37735409` resolves to chicken (`host_name`, taxid 9031).
A CPU CLI smoke run in Mistral mode completed both without an API key or model loading,
with verification explicitly disabled for that smoke check. Verification is otherwise unchanged.

All 44 documented CAGED pairs resolve to chicken. Their current records were checked for
`host=Gallus gallus`, `ifsac_category=chicken`, and `last_update` on 18 June 2025.
The historical ENA source is the AllTheBacteria May 2025 snapshot.

## Separate evaluations of the saved predictions

**These are the earlier saved model predictions, not results of a new model run.**
The adjusted evaluation excludes only the exact 44 verified run/accession pairs.
The 41 reference-unknown/human-evidence records remain scored, with unchanged labels.

| Input | Method | Original accuracy (1,320) | Metadata-version-adjusted accuracy (1,276) |
|---|---|---:|---:|
| TSV | NLI | 86.14% | 85.66% |
| TSV | Local LLM | 92.05% | 91.77% |
| TSV | Mistral API | 97.27% | 97.18% |
| BioSample | NLI | 60.98% | 63.01% |
| BioSample | Local LLM | 85.83% | 88.79% |
| BioSample | Mistral API | 88.94% | 92.01% |

Files:

- [Per-record deterministic audit](audit.tsv)
- [Audit summary and input hashes](summary.json)
- [Combined saved-prediction evaluations](saved_prediction_summary.tsv)
- [Original evaluation](saved_prediction_evaluation/original/summary.tsv)
- [Metadata-version-adjusted evaluation](saved_prediction_evaluation/metadata_version_adjusted/summary.tsv)
- [Normal audit log and taxonomy file provenance](audit.tsv.log)

The original classification output files and reference labels were not overwritten.
A new full model benchmark would be required to measure the complete updated pipeline.
