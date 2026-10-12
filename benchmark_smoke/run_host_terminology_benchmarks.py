"""Run fresh BioSample NLI/Ministral benchmarks offline; never call the paid API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taxonomy-dir', type=Path, default=Path('/home/zomer007/metalyzer/taxonomy'))
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'benchmark_smoke/host_terminology_audit')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'benchmark_smoke/host_terminology_benchmark')
    parser.add_argument('--other-animal-taxonomy', action='store_true')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no expensive CPU inference will be started')
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(other_animal_taxonomy=args.other_animal_taxonomy,
                     input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                   for name in ['sources.tsv', 'benchmark.tsv', 'benchmark_accessions.tsv', 'benchmark_true_labels.tsv']},
                     code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [ROOT / 'metalyzer.py', *sorted((ROOT / 'modules').glob('*.py'))]},
                     taxonomy_files={name: dict(bytes=(args.taxonomy_dir / name).stat().st_size,
                                               mtime_ns=(args.taxonomy_dir / name).stat().st_mtime_ns)
                                     for name in ['nodes.dmp', 'names.dmp', 'merged.dmp']})
    audit = json.loads((args.audit_dir / 'summary.json').read_text())
    if signature['input_sha256'] != audit['input_sha256']:
        raise ValueError('Inputs differ from the CPU audit; repeat the audit first')
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['inputs'] != signature:
            raise ValueError('Existing benchmark directory has different settings or code; choose another directory')
    else:
        manifest = dict(inputs=signature, gpu=torch.cuda.get_device_name(0),
                        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        started=datetime.now(timezone.utc).isoformat(), runs={})
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', '')
    for method in ['nli', 'llm']:
        output = out / f'benchmark_biosample_{method}_host_terminology.tsv'
        if manifest['runs'].get(method, {}).get('status') == 'complete':
            print(f'Reusing completed {method}', flush=True)
            continue
        if output.exists():
            raise FileExistsError(f'Unexpected existing output: {output}')
        command = [sys.executable, str(ROOT / 'metalyzer.py'), '--biosample-file',
                   str((args.audit_dir / 'biosample_accessions.txt').resolve()), '--sources', str(ROOT / 'sources.tsv'),
                   '--out', str(output), '--method', method, '--taxonomy-dir', str(args.taxonomy_dir),
                   '--device', '0', '--batch-size', '64', '--llm-batch-size', '1', '--llm-min-score', '0.75']
        if args.other_animal_taxonomy:
            command.append('--other-animal-taxonomy')
        print(f'Starting {method}: GPU={manifest["gpu"]}; 1,320 requests; verification enabled', flush=True)
        started = time.monotonic()
        with (out / f'{output.name}.console.log').open('w') as handle:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        manifest['runs'][method] = dict(status='complete' if result.returncode == 0 else 'failed',
                                        returncode=result.returncode, command=command,
                                        elapsed_seconds=time.monotonic()-started,
                                        finished=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'{method}: {manifest["runs"][method]["status"]}; {time.monotonic()-started:.1f}s', flush=True)
        if result.returncode:
            raise RuntimeError(f'{method} failed; inspect {output.name}.console.log')


if __name__ == '__main__':
    main()
