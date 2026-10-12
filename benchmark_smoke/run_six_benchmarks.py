"""Run both benchmark input modes with all three backends and local taxonomy.

Run from the repository root. Existing results stay untouched until evaluated
and published; completed runs in the work directory can be resumed safely.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_tsv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'benchmark_smoke/rerun_20261012')
    parser.add_argument('--taxonomy-dir', type=Path, default=Path('/home/zomer007/metalyzer/taxonomy'))
    parser.add_argument('--api-key-file', type=Path, default=Path('/home/zomer007/mistral.key'))
    args = parser.parse_args()
    os.chdir(ROOT)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    metadata = read_tsv('benchmark.tsv')
    requests = read_tsv('benchmark_accessions.tsv')
    truth = read_tsv('benchmark_true_labels.tsv')
    ids = []
    for name, rows in [('metadata', metadata), ('accessions', requests), ('truth', truth)]:
        keys = [row['run_accession'].strip() for row in rows]
        if not all(keys) or len(keys) != len(set(keys)):
            raise ValueError(f'Blank or duplicate run accession in {name}')
        ids.append(set(keys))
    if not ids[0] == ids[1] == ids[2]:
        raise ValueError('Metadata, accession list and truth must cover identical run accessions')
    by_run = {row['run_accession']: row['sample_accession'].strip() for row in metadata}
    mapping = [{'run_accession': row['run_accession'], 'accession': by_run[row['run_accession']]} for row in requests]
    if any(not row['accession'] for row in mapping):
        raise ValueError('Missing BioSample accession in benchmark metadata')
    accession_file = work / 'biosample_accessions.txt'
    accession_file.write_text(''.join(row['accession'] + '\n' for row in mapping))
    with (work / 'accession_mapping.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['run_accession', 'accession'], delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(mapping)
    for name in ('nodes.dmp', 'names.dmp', 'merged.dmp'):
        if not (args.taxonomy_dir / name).is_file():
            raise FileNotFoundError(args.taxonomy_dir / name)
    signature = {name: digest(name) for name in ['benchmark.tsv', 'benchmark_accessions.tsv', 'benchmark_true_labels.tsv', 'sources.tsv']}
    signature['taxonomy'] = [(name, (args.taxonomy_dir / name).stat().st_size, (args.taxonomy_dir / name).stat().st_mtime_ns) for name in ['nodes.dmp', 'names.dmp', 'merged.dmp']]
    signature = json.loads(json.dumps(signature))
    manifest_file = work / 'manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest['inputs'] != signature:
            raise ValueError('Work directory belongs to different inputs; choose a new --work-dir')
    else:
        manifest = {'inputs': signature, 'started': datetime.now(timezone.utc).isoformat(), 'records': len(requests), 'runs': {}}
        manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    backups = work / 'previous_outputs'
    backups.mkdir(exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', '')
    failed = []
    for mode in ['metadata', 'biosample']:
        for method, suffix in [('nli', 'nli'), ('llm', 'llm'), ('mistral', 'mistral_api')]:
            filename = f'benchmark_{suffix}.tsv' if mode == 'metadata' else f'benchmark_biosample_{suffix}.tsv'
            output = work / filename
            for prior in (ROOT / filename, ROOT / (filename + '.log')):
                if prior.exists() and not (backups / prior.name).exists():
                    shutil.copy2(prior, backups / prior.name)
            key = mode + '_' + method
            if manifest['runs'].get(key, {}).get('status') == 'complete' and output.exists():
                print(f'Reusing completed {key}: {output}', flush=True)
                continue
            command = [sys.executable, str(ROOT / 'metalyzer.py')]
            command += (['--metadata', str(ROOT / 'benchmark.tsv'), '--id-col', 'run_accession'] if mode == 'metadata'
                        else ['--biosample-file', str(accession_file)])
            command += ['--sources', str(ROOT / 'sources.tsv'), '--out', str(output), '--method', method,
                        '--taxonomy-dir', str(args.taxonomy_dir)]
            if method == 'mistral':
                command += ['--api-key-file', str(args.api_key_file)]
            print(f'Starting {key}: {len(requests)} records; taxonomy={args.taxonomy_dir}', flush=True)
            started = time.monotonic()
            with (work / (filename + '.console.log')).open('w') as handle:
                result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            manifest['runs'][key] = {'status': 'complete' if result.returncode == 0 else 'failed',
                                    'command': command, 'output': str(output), 'returncode': result.returncode,
                                    'elapsed_seconds': time.monotonic() - started,
                                    'finished': datetime.now(timezone.utc).isoformat()}
            manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
            print(f'{key}: {manifest["runs"][key]["status"]}; elapsed {time.monotonic()-started:.1f}s', flush=True)
            if result.returncode:
                failed.append(key)
    if failed:
        raise SystemExit('Failed runs: ' + ', '.join(failed) + '; see console logs; successful runs are retained')
    print('All six benchmark runs complete', flush=True)


if __name__ == '__main__':
    main()
