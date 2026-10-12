# BioSample host terminology audit

This CPU-only audit reads all 1,320 requests from the local indexed Parquet store,
preserves the existing run/accession mapping and duplicate request, and verifies
that the raw BioSample dataframe is unchanged. No models or APIs were used.
The baseline assignments exactly match the previous saved host-name audit.

| Configuration | Numeric | Scientific | Common/synonym | Alias | Other animal | Model fallback |
|---|---:|---:|---:|---:|---:|---:|
| Current | 1 | 295 | 106 | 0 | 0 | 918 |
| Aliases | 1 | 295 | 106 | 117 | 0 | 801 |
| Other animal | 1 | 295 | 106 | 0 | 42 | 876 |
| Combined | 1 | 295 | 106 | 117 | 42 | 759 |

All configurations have zero ambiguous/conflicting host evidence in this dataset.
These are source-assignment counts; successfully resolving a taxon does not
guarantee a configured source. Failure to resolve, or context-sensitive taxa,
still use the model.

New aliases assign dog 22, pig 39, sheep 35, goat 20, cat 1. All 117 agree with
unchanged references. All 83 `Bovine` records remain exact NCBI `host_name`
assignments; bovine is never unnecessarily assigned by an operational alias.
Aliases are explicit assumptions about veterinary terminology rather than
NCBI-derived taxids. They can be disabled with `--disable-host-aliases`.

The optional specific-unlisted-animal rule assigns 42 records, including all
five alpacas. Forty-one agree with references. `SRR12495807` / `SAMN08235377`
has `host=Capra aegagrus` (taxid 9923, wild goat), outside the configured
`Capra hircus` lineage. It becomes `other_animal`, while the reference says
goat. No species-specific exception was introduced; the rule remains disabled
by default. Goose, duck, Columbidae, broad taxa and ancestors of configured
domestic anchors stay unresolved.

The 44 documented CAGED pairs were checked against live `Gallus gallus`,
`ifsac_category=chicken` and the 18 June 2025 update date. Their assignments
remain chicken. Only these exact pairs are excluded from adjusted scoring.
The separate 41 reference-unknown / human-evidence records remain visible and
scored with their original references.

- [All configurations, per record](audit.tsv)
- [Summary counts](summary.tsv), [settings and hashes](summary.json)
- [Newly assigned disagreements](combined_new_disagreements.tsv)
- [Original request mapping](accession_mapping.tsv)
- [Audit log and taxonomy file provenance](audit.tsv.log)

Reproduce from the repository root:

```bash
python benchmark_smoke/audit_host_terminology.py --taxonomy-dir /path/to/taxonomy
```
