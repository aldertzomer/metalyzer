# TSV host terminology audit

This CPU-only audit uses all 1,320 rows of the original `benchmark.tsv`, with
unchanged column order, record formatting and sequencing run IDs. It does not
load models, read BioSample Parquet metadata or change references. The existing
BioSample/run mapping is retained only for consistent evaluation exclusions.

| Configuration | Numeric IDs | Scientific names | Common/synonyms | Aliases | Other animal | Model fallback |
|---|---:|---:|---:|---:|---:|---:|
| Current exact taxonomy | 215 | 1 | 3 | 0 | 0 | 1,101 |
| Plus aliases | 215 | 1 | 3 | 13 | 0 | 1,088 |
| Plus opt-in other animal | 215 | 1 | 3 | 0 | 46 | 1,055 |
| Combined | 215 | 1 | 3 | 13 | 46 | 1,042 |

The historical saved TSV runs had 215 numeric taxonomy assignments. Exact
host-name resolution adds four, and aliases add 13 more: eight Canine→dog and
five Ovine→sheep, all matching references. They appear in `host_scientific_name`
in this export, and remain operational descriptors rather than scientific taxa.
The 215 numeric assignments retain precedence. There are no host conflicts or
ambiguous names in any configuration.

The additional animal rule adds 46 assignments, with 45 matching references.
The remaining row is `SRR12495807` / `SAMN08235377`, where `Capra aegagrus`
resolves outside the configured domestic-goat lineage and becomes `other_animal`
despite a goat reference. The rule remains independently opt-in; no specific
species or benchmark exceptions were added to production code.

The original 1,320-record evaluation is retained. Adjusted evaluation excludes
only the same 44 documented CAGED run/BioSample pairs; human-evidence records
whose reference says unknown remain included without relabelling.

- [All four configurations](audit.tsv)
- [Counts](summary.tsv), [input hashes and provenance](summary.json)
- [New assignment disagreements](combined_new_disagreements.tsv)
- [Request mapping in TSV input order](accession_mapping.tsv)

Reproduce:

```bash
python benchmark_smoke/audit_host_terminology.py --input-mode metadata
```
