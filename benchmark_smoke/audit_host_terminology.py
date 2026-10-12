"""Compare four deterministic host configurations using live local BioSamples."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from modules import biosample, deterministic_source as ds, input, runlog
from benchmark_smoke.metadata_version_evaluation import validated_conflict_runs


def audit(taxonomy_dir, store, out):
    out.mkdir(parents=True, exist_ok=True)
    mapping = input.read_tsv(ROOT / 'benchmark_smoke/host_name_audit/accession_mapping.tsv')
    metadata = input.read_tsv(ROOT / 'benchmark.tsv')
    truth = input.read_tsv(ROOT / 'benchmark_true_labels.tsv')
    if mapping.run_accession.duplicated().any() or len(mapping) != 1320:
        raise ValueError('Expected the existing 1,320 unique benchmark requests')
    expected = metadata.set_index('run_accession').sample_accession
    if mapping.accession.tolist() != expected.loc[mapping.run_accession].tolist():
        raise ValueError('Stored BioSample mapping differs from benchmark metadata')
    references = truth.set_index('run_accession').final_species
    started = time.monotonic()
    with runlog.analysis_log(out / 'audit.tsv'):
        runlog.emit('CPU-only four-configuration audit; no models or network')
        frame = biosample.load_biosamples(mapping.accession.tolist(), store)
        before = frame.copy(deep=True)
        taxonomy = ds.NCBITaxonomy(taxonomy_dir)
        sources = input.load_sources(ROOT / 'sources.tsv', 'accession')
        anchors = ds.configured_anchors(sources, taxonomy, report=True)
        for name in ds.FILES:
            path = (taxonomy_dir / name).resolve()
            stat = path.stat()
            runlog.emit(f'Taxonomy file: {path}; bytes={stat.st_size}; mtime_ns={stat.st_mtime_ns}')
        excluded = validated_conflict_runs(
            truth.to_dict('records'), metadata.to_dict('records'), mapping.to_dict('records'),
            input.read_tsv(ROOT / 'benchmark_smoke/biosample_diagnostics/goat_host_conflicts.tsv').to_dict('records'),
            biosample_rows=frame.to_dict('records'))
        batch = input.prepare_batch(frame, id_col='accession')
        baseline = None
        summaries, all_rows = [], []
        for config, aliases, animals in [('current', False, False), ('aliases', True, False),
                                        ('other_animal', False, True), ('combined', True, True)]:
            diagnostics = []
            ds.run(batch, sources, taxonomy, anchor_sources=anchors, diagnostics=diagnostics,
                   use_host_aliases=aliases, other_animal=animals)
            rows = []
            for mapped, item in zip(mapping.to_dict('records'), diagnostics):
                key, resolution = item['row_key'], item['resolution']
                host = frame.iloc[key].get('host', '')
                matches = [taxonomy.resolve_host_name(value) for _, value in ds.host_name_values(frame.iloc[key])]
                row = dict(configuration=config, **mapped, host=host,
                           host_scientific_name=frame.iloc[key].get('host_scientific_name', ''),
                           match_type=resolution.match_type or ' | '.join(dict.fromkeys(m.match_type for m in matches)),
                           resolved_taxids=' | '.join(map(str, resolution.name_taxids or resolution.numeric_taxids)),
                           assigned_source=resolution.result.source if resolution.result else '',
                           source_method=resolution.method, evidence=resolution.evidence,
                           ambiguous=resolution.ambiguous, conflict=resolution.conflict,
                           reference=references.loc[mapped['run_accession']],
                           excluded_caged=mapped['run_accession'] in excluded,
                           newly_assigned=False, reference_disagreement=False,
                           requires_model=resolution.result is None)
                rows.append(row)
            if baseline is None:
                baseline = rows
                old = input.read_tsv(ROOT / 'benchmark_smoke/host_name_audit/audit.tsv')
                if (old.best_hit.tolist() != [r['assigned_source'] for r in rows]
                        or old.source_method.tolist() != [r['source_method'] for r in rows]):
                    raise ValueError('Baseline differs from the saved current taxonomy audit')
            for row, prior in zip(rows, baseline):
                row['newly_assigned'] = prior['requires_model'] and not row['requires_model']
                row['reference_disagreement'] = bool(row['assigned_source'] and row['assigned_source'] != row['reference'])
            newly = [r for r in rows if r['newly_assigned']]
            counts = Counter(r['source_method'] for r in rows)
            summary = dict(configuration=config, records=len(rows), numeric=counts['host_tax_id'],
                           scientific=sum(r['source_method'] == 'host_name' and r['match_type'] == 'scientific name' for r in rows),
                           common_synonym=sum(r['source_method'] == 'host_name' and r['match_type'] == 'common name/synonym' for r in rows),
                           aliases=counts['host_alias'], other_animal=counts['host_other_animal'],
                           pending=counts[''], ambiguous=sum(r['ambiguous'] for r in rows),
                           conflicts=sum(bool(r['conflict']) for r in rows), newly_assigned=len(newly),
                           newly_assigned_reference_disagreements=sum(r['reference_disagreement'] for r in newly),
                           new_source_distribution=dict(Counter(r['assigned_source'] for r in newly)),
                           veterinary_new_assignments={term: sum(r['newly_assigned'] and ds.normalized_host_name(r['host']) == term.casefold() for r in rows)
                                                      for term in ['Bovine', 'Canine', 'Porcine', 'Ovine', 'Caprine', 'Feline']})
            pd.DataFrame(rows).to_csv(out / f'{config}_audit.tsv', sep='\t', index=False)
            pd.DataFrame([r for r in newly if r['reference_disagreement']], columns=rows[0].keys()).to_csv(
                out / f'{config}_new_disagreements.tsv', sep='\t', index=False)
            summaries.append(summary)
            all_rows.extend(rows)
            runlog.emit(json.dumps(summary, indent=2))
        pd.testing.assert_frame_equal(frame, before)
        mapping.to_csv(out / 'accession_mapping.tsv', sep='\t', index=False)
        (out / 'biosample_accessions.txt').write_text(''.join(value + '\n' for value in mapping.accession))
        pd.DataFrame(all_rows).to_csv(out / 'audit.tsv', sep='\t', index=False)
        (out / 'summary.json').write_text(json.dumps(dict(
            configurations=summaries, excluded_caged_runs=sorted(excluded),
            original_records=1320, adjusted_records=1320-len(excluded),
            elapsed_seconds=time.monotonic()-started, biosample_store=str(store.resolve()),
            taxonomy_dir=str(taxonomy_dir.resolve()),
            input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ['sources.tsv', 'benchmark.tsv', 'benchmark_true_labels.tsv', 'benchmark_accessions.tsv']}), indent=2)+'\n')
        pd.DataFrame(summaries).drop(columns=['new_source_distribution', 'veterinary_new_assignments']).to_csv(
            out / 'summary.tsv', sep='\t', index=False)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taxonomy-dir', type=Path, default=Path('/home/zomer007/metalyzer/taxonomy'))
    parser.add_argument('--biosample-path', type=Path, default=biosample.DEFAULT_STORE)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'benchmark_smoke/host_terminology_audit')
    args = parser.parse_args()
    audit(args.taxonomy_dir, args.biosample_path, args.out_dir)


if __name__ == '__main__':
    main()
