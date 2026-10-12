"""Validate six completed benchmark runs, publish results, and refresh README figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from benchmark_smoke.benchmark_evaluation import read_tsv
from benchmark_smoke.evaluate_local_backends import evaluate
from render_benchmark_matrices import load_matrix, render_svg

RUNS = [('metadata', 'nli', 'benchmark_nli.tsv', 'TSV · NLI', 'benchmark-tsv-nli'),
        ('metadata', 'llm', 'benchmark_llm.tsv', 'TSV · Local Ministral', 'benchmark-tsv-llm'),
        ('metadata', 'mistral', 'benchmark_mistral_api.tsv', 'TSV · Mistral API', 'benchmark-tsv-mistral-api'),
        ('biosample', 'nli', 'benchmark_biosample_nli.tsv', 'BioSample · NLI', 'benchmark-biosample-nli'),
        ('biosample', 'llm', 'benchmark_biosample_llm.tsv', 'BioSample · Local Ministral', 'benchmark-biosample-llm'),
        ('biosample', 'mistral', 'benchmark_biosample_mistral_api.tsv', 'BioSample · Mistral API', 'benchmark-biosample-mistral-api')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'benchmark_smoke/rerun_20261012')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    manifest = json.loads((work / 'manifest.json').read_text())
    for name, expected in manifest['inputs'].items():
        if name != 'taxonomy' and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input changed during benchmark: {name}')
    truth = pd.read_csv(ROOT / 'benchmark_true_labels.tsv', sep='\t', dtype=str, keep_default_na=False)
    mapping = read_tsv(work / 'accession_mapping.tsv')
    reports = work / 'evaluation'
    reports.mkdir(exist_ok=True)
    figures = work / 'figures'
    figures.mkdir(exist_ok=True)
    summary = []
    for mode, method, filename, title, prefix in RUNS:
        if manifest['runs'].get(mode + '_' + method, {}).get('status') != 'complete':
            raise ValueError(f'Benchmark not complete: {mode}_{method}')
        path = work / filename
        log = Path(str(path) + '.log').read_text()
        if 'Taxonomy configuration:' not in log or 'Cannot load taxonomy' in log:
            raise ValueError(f'Taxonomy was not successfully loaded: {filename}')
        result = evaluate(truth, path, 'best_hit', reports, False, mapping if mode == 'biosample' else None)
        result.update(input_mode=mode, method=method, file=filename,
                      elapsed_seconds=manifest['runs'][mode + '_' + method]['elapsed_seconds'])
        summary.append(result)
        totals, matrix = load_matrix(path, work / 'accession_mapping.tsv' if mode == 'biosample' else None)
        if sum(totals.values()) != len(truth):
            raise ValueError('Confusion matrix has incorrect total')
        for percentages, suffix in [(False, 'absolute'), (True, 'percent')]:
            render_svg(figures / f'{prefix}-confusion-{suffix}.svg', title + ' source confusion matrix', totals, matrix, percentages)
    pd.DataFrame(summary).drop(columns='source_methods').to_csv(reports / 'summary.tsv', sep='\t', index=False)
    (reports / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # All inputs, outputs and figures validated before publishing over prior results.
    for _, _, filename, _, _ in RUNS:
        shutil.copy2(work / filename, ROOT / filename)
        shutil.copy2(work / (filename + '.log'), ROOT / (filename + '.log'))
    destination = ROOT / 'benchmark_smoke/six_method_evaluation'
    destination.mkdir(exist_ok=True)
    for path in reports.iterdir():
        shutil.copy2(path, destination / path.name)
    shutil.copy2(work / 'accession_mapping.tsv', destination / 'accession_mapping.tsv')
    for path in figures.iterdir():
        shutil.copy2(path, ROOT / 'assets' / path.name)
    rows = ['## Benchmark results', '',
            'The six full source-classification benchmarks below were rerun on 12 October 2026',
            'using the reordered [metadata](benchmark.tsv) and',
            '[accession list](benchmark_accessions.tsv), with 1,320 records and',
            '1,319 distinct BioSamples. All runs loaded the same local NCBI taxonomy',
            'and validated the 12 anchors configured in [sources.tsv](sources.tsv).', '',
            '`benchmark_accessions.tsv` contains sequencing run IDs. These were mapped to',
            'BioSample IDs using `sample_accession` in `benchmark.tsv`; the order and',
            'duplicate BioSample request were preserved. BioSample metadata came exclusively',
            'from the local indexed Parquet repository. It was not supplemented with TSV',
            'metadata. The [run-to-BioSample mapping](benchmark_smoke/six_method_evaluation/accession_mapping.tsv)',
            'is retained for reproducible evaluation.', '',
            'Only source labels are scored against `final_species` in',
            '[benchmark_true_labels.tsv](benchmark_true_labels.tsv). Truth-file order does not',
            'affect matching: TSV outputs join by `run_accession`; BioSample outputs join',
            'through the accession mapping, preserving repeated requests. Every run was',
            'checked for complete coverage. Country, year, LLM confidence and NLI verification',
            'scores are not scoring targets here.', '',
            'NLI uses `--min-score 0.2`; local Ministral uses `--llm-min-score 0.75`.',
            'Mistral API uses `mistral-small-latest`, seed 12345, and the default API settings.',
            'All methods retain normal NLI source verification, date/country extraction,',
            'run logs, and TSV output. API results reflect the service available on the run date.', '',
            '| Input | Method | Correct / total | Accuracy | Assigned precision | Unknown | Taxonomy calls |',
            '|---|---|---:|---:|---:|---:|---:|']
    for result, (_, _, filename, title, _) in zip(summary, RUNS):
        mode, method = title.split(' · ')
        rows.append(f'| {mode} | [{method}]({filename}) | {result["correct"]} / {result["records"]} | '
                    f'{100*result["overall_accuracy"]:.2f}% | {100*result["precision_assigned"]:.2f}% | '
                    f'{result["unknown_predictions"]} | {result["taxonomy_calls"]} |')
    rows += ['', 'Accuracy includes correct `unknown` predictions. Assigned precision excludes',
             'predictions of `unknown`. Enabling taxonomy only resolves records with a usable',
             '`host_tax_id`; the taxonomy-call column reports actual deterministic assignments.',
             'Input modes differ in metadata content and availability of host taxonomy IDs,',
             'so these results compare the complete pipelines on their respective inputs.', '',
             'Rows in each confusion matrix are true sources; columns are predicted sources.',
             'The embedded figures show row percentages. Absolute-count figures and per-source',
             'recall/precision tables are linked below. All source metrics are also available',
             'in the [six-run summary](benchmark_smoke/six_method_evaluation/summary.tsv).', '']
    for (_, _, filename, title, prefix), result in zip(RUNS, summary):
        rows += [f'### {title} benchmark', '',
                 f'[![{title} source confusion matrix](assets/{prefix}-confusion-percent.svg)](assets/{prefix}-confusion-percent.svg)', '',
                 f'[Absolute counts](assets/{prefix}-confusion-absolute.svg) · '
                 f'[Per-source metrics](benchmark_smoke/six_method_evaluation/{Path(filename).stem}_per_class.tsv) · '
                 f'[Confusion counts TSV](benchmark_smoke/six_method_evaluation/{Path(filename).stem}_confusion_counts.tsv)', '']
    rows += ['Regenerate all six runs, metrics and figures from the repository root:', '',
             '```bash', 'python benchmark_smoke/run_six_benchmarks.py',
             'python benchmark_smoke/publish_six_benchmarks.py', '```', '',
             'The runner preserves prior outputs under `benchmark_smoke/rerun_20261012/previous_outputs/`',
             'and resumes completed runs only when the input hashes match. Its default',
             'taxonomy and API-key paths match this server; override `--taxonomy-dir` and',
             '`--api-key-file` for another installation. The active environment must provide',
             'GPU access, cached local models, and the optional Mistral SDK. API runs require',
             'network access and may incur charges.', '', '---', '']
    readme = ROOT / 'README.md'
    old = readme.read_text()
    backup = work / 'previous_outputs/README.md'
    if not backup.exists():
        shutil.copy2(readme, backup)
    start, end = old.index('## Benchmark results'), old.index('## Development and tests')
    readme.write_text(old[:start] + '\n'.join(rows) + '\n' + old[end:])
    print(pd.DataFrame(summary)[['input_mode', 'method', 'correct', 'records', 'overall_accuracy', 'precision_assigned', 'unknown_predictions', 'taxonomy_calls']].to_string(index=False))
    print('Published six outputs, twelve SVG figures, metrics and README benchmark section.')


if __name__ == '__main__':
    main()
