"""Evaluate new and old BioSample predictions with exact pair exclusions and routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from modules import input
from benchmark_smoke.benchmark_evaluation import align_predictions
from benchmark_smoke.evaluate_local_backends import evaluate
from benchmark_smoke.metadata_version_evaluation import validated_conflict_runs
from render_benchmark_matrices import LABELS, render_svg


def route_metrics(rows):
    assigned = rows.best_hit.ne('unknown')
    correct = rows.best_hit.eq(rows.final_species)
    return dict(records=len(rows), correct=int(correct.sum()),
                overall_accuracy=float(correct.mean()) if len(rows) else None,
                assigned=int(assigned.sum()), correct_assigned=int((correct & assigned).sum()),
                assigned_precision=float(correct[assigned].mean()) if assigned.any() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'benchmark_smoke/host_terminology_audit')
    parser.add_argument('--benchmark-dir', type=Path, default=ROOT / 'benchmark_smoke/host_terminology_benchmark')
    args = parser.parse_args()
    out = args.benchmark_dir / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.benchmark_dir / 'manifest.json').read_text())
    if any(manifest['runs'].get(method, {}).get('status') != 'complete' for method in ['nli', 'llm']):
        raise ValueError('Both full benchmarks must finish, including verification, before evaluation')
    configuration = 'combined' if manifest['inputs']['other_animal_taxonomy'] else 'aliases'
    audit = input.read_tsv(args.audit_dir / f'{configuration}_audit.tsv')
    truth = input.read_tsv(ROOT / 'benchmark_true_labels.tsv')
    metadata = input.read_tsv(ROOT / 'benchmark.tsv')
    mapping = input.read_tsv(args.audit_dir / 'accession_mapping.tsv').to_dict('records')
    conflicts = input.read_tsv(ROOT / 'benchmark_smoke/biosample_diagnostics/goat_host_conflicts.tsv')
    excluded = validated_conflict_runs(truth.to_dict('records'), metadata.to_dict('records'), mapping,
                                      conflicts.to_dict('records'))
    # The audit independently checked the live host, category and update date.
    if sorted(excluded) != json.loads((args.audit_dir / 'summary.json').read_text())['excluded_caged_runs']:
        raise ValueError('Exclusion list differs from the live-validated CPU audit')
    human_evidence = input.read_tsv(ROOT / 'benchmark_smoke/biosample_diagnostics/unknown_human_evidence.tsv')
    human_runs = set(human_evidence.run_accession)
    if (len(human_runs) != 41 or human_runs & excluded
            or not truth.set_index('run_accession').loc[list(human_runs)].final_species.eq('unknown').all()):
        raise ValueError('The 41 documented human-evidence reference-unknown records must remain scored unchanged')
    summary, strata, performance, changed = [], [], [], []
    for method in ['nli', 'llm']:
        new_path = args.benchmark_dir / f'benchmark_biosample_{method}_host_terminology.tsv'
        old_path = ROOT / f'benchmark_biosample_{method}.tsv'
        for version, path in [('old_saved', old_path), ('new_combined' if configuration == 'combined' else 'new_aliases', new_path)]:
            predictions = input.read_tsv(path)
            _, aligned = align_predictions(truth.to_dict('records'), predictions.to_dict('records'), mapping)
            merged = truth.merge(pd.DataFrame(aligned.values()), on='run_accession', validate='one_to_one')
            merged = merged.merge(audit[['run_accession', 'match_type', 'host', 'newly_assigned', 'source_method']].rename(
                columns={'source_method': 'audit_method'}), on='run_accession', validate='one_to_one')
            if version != 'old_saved':
                expected = audit[audit.source_method.ne('')].set_index('run_accession')
                actual = merged.set_index('run_accession').loc[expected.index]
                if (actual.best_hit.tolist() != expected.assigned_source.tolist()
                        or actual.source_method.tolist() != expected.source_method.tolist()):
                    raise ValueError('Benchmark deterministic outputs differ from audited assignments')
            for label, exclusions in [('original', None), ('metadata_version_adjusted', excluded)]:
                destination = out / label
                destination.mkdir(exist_ok=True)
                measured = evaluate(truth, path, 'best_hit', destination, False, mapping, exclusions)
                measured.update(version=version, method=method, file=str(path.relative_to(ROOT)))
                summary.append(measured)
                rows = merged if exclusions is None else merged[~merged.run_accession.isin(exclusions)]
                for route, group in rows.groupby('source_method'):
                    strata.append(dict(version=version, method=method, evaluation=label, pathway=route, **route_metrics(group)))
                if version != 'old_saved':
                    names = rows[rows.source_method.eq('host_name')]
                    for match_type, group in names.groupby('match_type'):
                        strata.append(dict(version=version, method=method, evaluation=label,
                                           pathway=f'host_name: {match_type}', **route_metrics(group)))
                    matrix = pd.crosstab(rows.final_species, rows.best_hit).reindex(index=LABELS, columns=LABELS, fill_value=0)
                    totals = matrix.sum(axis=1).to_dict()
                    values = {key: matrix.loc[key].to_dict() for key in LABELS}
                    for percentages, suffix in [(False, 'absolute'), (True, 'percent')]:
                        render_svg(destination / f'{method}_confusion_{suffix}.svg',
                                   f'BioSample {method.upper()} · {configuration} · {label} (n={len(rows)})',
                                   totals, values, percentages)
            if version == 'old_saved':
                previous = merged.set_index('run_accession')
            else:
                merged[merged.run_accession.isin(human_runs)][[
                    'run_accession', 'accession', 'host', 'final_species', 'best_hit', 'source_method', 'nli_verification_score'
                ]].to_csv(out / f'{method}_reference_unknown_human_evidence.tsv', sep='\t', index=False)
                for row in merged.to_dict('records'):
                    old = previous.loc[row['run_accession']]
                    changed.append(dict(method=method, run_accession=row['run_accession'], accession=row['accession'],
                                        host=row['host'], reference=row['final_species'], old_best_hit=old.best_hit,
                                        new_best_hit=row['best_hit'], source_method=row['source_method'],
                                        changed=row['best_hit'] != old.best_hit,
                                        old_correct=old.best_hit == row['final_species'],
                                        new_correct=row['best_hit'] == row['final_species']))
        log = Path(str(new_path) + '.log').read_text()
        def duration(marker):
            match = re.search(re.escape(marker) + r': ([0-9.]+)', log)
            return float(match[1]) if match else None
        selected = merged.source_method.eq(method)
        baseline = ROOT / 'benchmark_smoke/rerun_20261012/manifest.json'
        prior_manifest = json.loads(baseline.read_text()) if baseline.exists() else {}
        performance.append(dict(method=method, gpu=manifest['gpu'], gpu_memory_bytes=manifest['gpu_memory_bytes'],
                                total_seconds=manifest['runs'][method]['elapsed_seconds'],
                                source_model_stage_seconds_including_load=duration('Source model stage seconds (including weight loading)'),
                                verification_stage_seconds_including_any_load=duration('NLI verification stage seconds (including any weight loading)'),
                                source_model_rows=int(selected.sum()),
                                old_source_model_rows=int(previous.source_method.eq(method).sum()),
                                source_model_rows_avoided=int(previous.source_method.eq(method).sum()-selected.sum()),
                                old_total_seconds=prior_manifest.get('runs', {}).get('biosample_'+method, {}).get('elapsed_seconds')))
    pd.DataFrame(summary).drop(columns='source_methods').to_csv(out / 'summary.tsv', sep='\t', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    pd.DataFrame(strata).to_csv(out / 'pathway_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(changed).to_csv(out / 'prediction_comparison.tsv', sep='\t', index=False)
    changes = pd.DataFrame(changed)
    changes['improved'] = ~changes.old_correct & changes.new_correct
    changes['worsened'] = changes.old_correct & ~changes.new_correct
    changes.groupby(['method', 'source_method']).agg(
        records=('run_accession', 'size'), changed=('changed', 'sum'),
        improved=('improved', 'sum'), worsened=('worsened', 'sum')).to_csv(
            out / 'prediction_change_summary.tsv', sep='\t')
    pd.DataFrame(performance).to_csv(out / 'performance.tsv', sep='\t', index=False)
    print(pd.DataFrame(summary)[['method', 'version', 'records', 'overall_accuracy', 'precision_assigned']].to_string(index=False))


if __name__ == '__main__':
    main()
