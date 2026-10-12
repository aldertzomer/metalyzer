"""CPU-only deterministic BioSample audit and separately adjusted saved evaluations.

No model inference is performed. Run from the repository root:
  python benchmark_smoke/audit_host_taxonomy.py --taxonomy-dir /path/to/taxonomy
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from modules import biosample, deterministic_source, input, runlog
from benchmark_smoke.benchmark_evaluation import read_tsv, unique_by_id
from benchmark_smoke.metadata_version_evaluation import validated_conflict_runs
from benchmark_smoke.evaluate_local_backends import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taxonomy-dir', type=Path, required=True)
    parser.add_argument('--biosample-path', type=Path, default=biosample.DEFAULT_STORE)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'benchmark_smoke/host_name_audit')
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metadata = read_tsv(ROOT / 'benchmark.tsv')
    requests = read_tsv(ROOT / 'benchmark_accessions.tsv')
    truth_rows = read_tsv(ROOT / 'benchmark_true_labels.tsv')
    by_run = unique_by_id(metadata, 'metadata')
    unique_by_id(requests, 'accession requests')
    truth_by_run = unique_by_id(truth_rows, 'truth')
    if not set(by_run) == set(truth_by_run) == {r['run_accession'] for r in requests}:
        raise ValueError('Benchmark input and truth run IDs differ')
    mapping = [dict(run_accession=r['run_accession'], accession=by_run[r['run_accession']]['sample_accession']) for r in requests]
    pd.DataFrame(mapping).to_csv(out / 'accession_mapping.tsv', sep='\t', index=False)
    with runlog.analysis_log(out / 'audit.tsv'):
        runlog.emit('Offline deterministic host taxonomy audit; no model inference')
        for name in deterministic_source.FILES:
            path = (args.taxonomy_dir / name).resolve()
            stat = path.stat()
            runlog.emit(f'Taxonomy file {name}: {path}; {stat.st_size} bytes; modified {datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}')
        frame = biosample.load_biosamples([r['accession'] for r in mapping], args.biosample_path)
        before = frame.copy(deep=True)
        taxonomy = deterministic_source.NCBITaxonomy(args.taxonomy_dir)
        sources = input.load_sources(ROOT / 'sources.tsv', 'accession')
        anchors = deterministic_source.configured_anchors(sources, taxonomy, report=True)
        diagnostics = []
        result = deterministic_source.run(input.prepare_batch(frame, id_col='accession'), sources,
                                          taxonomy, anchor_sources=anchors, diagnostics=diagnostics)
        pd.testing.assert_frame_equal(frame, before)
        audit = []
        for item, mapped in zip(diagnostics, mapping):
            key, resolution = item['row_key'], item['resolution']
            audit.append(dict(**mapped, host=frame.iloc[key].get('host', ''),
                              host_scientific_name=frame.iloc[key].get('host_scientific_name', ''),
                              supplied_host_tax_id=frame.iloc[key].get('host_tax_id', ''),
                              supplied_host_taxid=frame.iloc[key].get('host_taxid', ''),
                              usable_numeric_host=bool(resolution.numeric_taxids),
                              numeric_taxids=' | '.join(map(str, resolution.numeric_taxids)),
                              resolvable_scientific_host=resolution.scientific_name_resolved,
                              resolved_name_taxids=' | '.join(map(str, resolution.name_taxids)),
                              ambiguous_name=resolution.ambiguous, conflict=resolution.conflict,
                              numeric_textual_conflict=resolution.numeric_text_conflict,
                              best_hit=resolution.result.source if resolution.result else '',
                              source_method=resolution.method, source_evidence=resolution.evidence,
                              requires_model=resolution.result is None))
        pd.DataFrame(audit).to_csv(out / 'audit.tsv', sep='\t', index=False)
        resolutions = [r['resolution'] for r in diagnostics]
        summary = dict(records=len(frame), usable_numeric_host_taxid=sum(bool(r.numeric_taxids) for r in resolutions),
                       resolvable_scientific_host_name=sum(r.scientific_name_resolved for r in resolutions),
                       resolvable_any_host_name=sum(bool(r.name_taxids) for r in resolutions),
                       unresolved_or_ambiguous_host_name=sum(not r.name_taxids or r.ambiguous for r in resolutions),
                       ambiguous_host_name=sum(r.ambiguous for r in resolutions),
                       conflicting_host_evidence=sum(bool(r.conflict) for r in resolutions),
                       numeric_textual_conflicts=sum(r.numeric_text_conflict for r in resolutions),
                       explicit_numeric_assignments=int(result.table.source_method.eq('host_tax_id').sum()),
                       exact_host_name_assignments=int(result.table.source_method.eq('host_name').sum()),
                       total_deterministic_assignments=len(result.table), model_fallback_records=len(frame) - len(result.table))
        runlog.emit(json.dumps(summary, indent=2))
        conflicts = read_tsv(ROOT / 'benchmark_smoke/biosample_diagnostics/goat_host_conflicts.tsv')
        excluded = validated_conflict_runs(truth_rows, metadata, mapping, conflicts,
                                          biosample_rows=frame.to_dict('records'))
        for item in audit:
            if item['run_accession'] in excluded and (item['best_hit'] != 'chicken' or item['source_method'] != 'host_name'):
                raise ValueError('A documented current chicken host was not deterministically assigned chicken')
        # Score the existing full predictions, without changing them or references.
        truth = pd.DataFrame(truth_rows)
        evaluation = []
        for label, exclusions in [('original', None), ('metadata_version_adjusted', excluded)]:
            directory = out / 'saved_prediction_evaluation' / label
            directory.mkdir(parents=True, exist_ok=True)
            for mode in ['', 'biosample_']:
                for suffix in ['nli', 'llm', 'mistral_api']:
                    filename = f'benchmark_{mode}{suffix}.tsv'
                    measured = evaluate(truth, ROOT / filename, 'best_hit', directory, False, mapping, exclusions)
                    measured['file'] = filename
                    evaluation.append(measured)
            group = evaluation[-6:]
            pd.DataFrame(group).drop(columns='source_methods').to_csv(directory / 'summary.tsv', sep='\t', index=False)
            (directory / 'summary.json').write_text(json.dumps(group, indent=2) + '\n')
        summary.update(excluded_documented_records=len(excluded), adjusted_records=len(frame) - len(excluded),
                       biosample_store=str(args.biosample_path.resolve()), taxonomy_dir=str(args.taxonomy_dir.resolve()),
                       elapsed_seconds=time.monotonic() - started,
                       input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                     for name in ['benchmark.tsv', 'benchmark_accessions.tsv', 'benchmark_true_labels.tsv', 'sources.tsv']})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        pd.DataFrame(evaluation).drop(columns='source_methods').to_csv(out / 'saved_prediction_summary.tsv', sep='\t', index=False)
        runlog.emit(f'Audit complete: {out}; original denominator {len(frame)}, metadata-version-adjusted denominator {len(frame)-len(excluded)}')


if __name__ == '__main__':
    main()
