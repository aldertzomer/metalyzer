"""Fresh 12-run Variant C comparison; same taxonomy, local models and paid API.

The selected BioSample metadata is read once from the local indexed Parquet
repository and cached for this benchmark only. Every prediction is fresh.
"""
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(mode, method, condition, output, args):
    command = (['--metadata', str(ROOT/'benchmark.tsv'), '--id-col', 'run_accession'] if mode=='tsv'
               else ['--biosample-file', str(args.work_dir/'biosample_accessions.txt')])
    command += ['--sources',str(ROOT/'sources.tsv'),'--out',str(output),'--method',method,
                '--taxonomy-dir',str(args.taxonomy_dir),'--other-animal-taxonomy',
                '--device','0','--batch-size','64','--llm-batch-size','1','--llm-min-score','0.75',
                '--min-score','0.2','--max-value-chars','300','--max-record-chars','2000']
    if condition=='unfiltered': command.append('--disable-noisy-field-exclusion')
    if method=='mistral':
        command += ['--api-key-file',str(args.api_key_file),'--mistral-model','mistral-small-latest',
                    '--mistral-random-seed','12345']
    return command


def child(args):
    import metalyzer
    from modules import biosample
    command=arguments(args.input_mode,args.method,args.condition,args.output,args)
    if args.input_mode=='tsv':
        metalyzer.main(command)
    else:
        import pandas as pd
        frame=pd.read_pickle(args.work_dir/'input_cache/biosample.pkl')
        def cached(accessions, path):
            if [str(value).strip() for value in accessions] != frame.accession.tolist():
                raise ValueError('Cached metadata does not match requested accession order')
            from modules.runlog import emit
            emit('Input mode: BioSample; benchmark-only selected metadata cache from local Parquet')
            emit(f'BioSample store: {biosample.DEFAULT_STORE}; requested/resolved: {len(frame)}')
            emit('BioSample accession index: '+str(biosample.index_path(biosample.DEFAULT_STORE)))
            return frame.copy(deep=True)
        with patch.object(biosample,'load_biosamples',side_effect=cached):
            metalyzer.main(command)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,default=ROOT/'benchmark_smoke/noisy_field_exclusion')
    parser.add_argument('--taxonomy-dir',type=Path,default=Path('/home/zomer007/metalyzer/taxonomy'))
    parser.add_argument('--api-key-file',type=Path,default=Path('/home/zomer007/mistral.key'))
    parser.add_argument('--run-one',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--input-mode',choices=['tsv','biosample'])
    parser.add_argument('--method',choices=['nli','llm','mistral'])
    parser.add_argument('--condition',choices=['filtered','unfiltered'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    args.work_dir=args.work_dir.resolve()
    if args.run_one:
        child(args)
        return
    import torch
    import pandas as pd
    from modules import biosample,input,mistral,deterministic_source as ds
    from benchmark_smoke.metadata_version_evaluation import validated_conflict_runs
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; benchmarks not started')
    work=args.work_dir
    work.mkdir(parents=True,exist_ok=True)
    for name in ['filtered','unfiltered','evaluation','figures','input_cache']:
        (work/name).mkdir(exist_ok=True)
    signature=dict(input_sha256={name:digest(ROOT/name) for name in ['benchmark.tsv','benchmark_accessions.tsv','benchmark_true_labels.tsv','sources.tsv']},
                   code_sha256={str(path.relative_to(ROOT)):digest(path) for path in [ROOT/'metalyzer.py',*sorted((ROOT/'modules').glob('*.py'))]},
                   taxonomy={name:dict(path=str((args.taxonomy_dir/name).resolve()),bytes=(args.taxonomy_dir/name).stat().st_size,
                                      mtime_ns=(args.taxonomy_dir/name).stat().st_mtime_ns) for name in ds.FILES},
                   biosample_store=[dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
                                    for path in biosample.store_files(biosample.DEFAULT_STORE)],
                   other_animal_taxonomy=True,host_aliases=True,
                   filter_profile='variant_c_v1',verification=True,date_country=True,
                   nli_batch_size=64,nli_min_score=.2,llm_batch_size=1,llm_min_score=.75,
                   mistral_model='mistral-small-latest',mistral_seed=12345,
                   max_value_chars=300,max_record_chars=2000)
    manifest_path=work/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['inputs']!=signature: raise ValueError('Inputs/code changed; choose another benchmark directory')
    else:
        manifest=dict(inputs=signature,gpu=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                      torch_version=torch.__version__,cuda_version=torch.version.cuda,
                      started=datetime.now(timezone.utc).isoformat(),runs={})
    metadata=input.read_tsv(ROOT/'benchmark.tsv')
    requests=input.read_tsv(ROOT/'benchmark_accessions.tsv')
    truth=input.read_tsv(ROOT/'benchmark_true_labels.tsv')
    for frame in [metadata,requests,truth]:
        if len(frame)!=1320 or frame.run_accession.duplicated().any(): raise ValueError('Expected 1,320 unique run IDs')
    if set(metadata.run_accession)!=set(requests.run_accession) or set(metadata.run_accession)!=set(truth.run_accession):
        raise ValueError('Input run IDs differ')
    mapping=pd.DataFrame(dict(run_accession=requests.run_accession,
                             accession=metadata.set_index('run_accession').loc[requests.run_accession].sample_accession.to_numpy()))
    mapping.to_csv(work/'accession_mapping.tsv',sep='\t',index=False)
    (work/'biosample_accessions.txt').write_text(''.join(value+'\n' for value in mapping.accession))
    cache=work/'input_cache/biosample.pkl'
    if not cache.exists():
        print('Reading 1,320 requested BioSample records once from local Parquet',flush=True)
        frame=biosample.load_biosamples(mapping.accession.tolist(),biosample.DEFAULT_STORE)
        frame.to_pickle(cache)
    else: frame=pd.read_pickle(cache)
    if frame.accession.tolist()!=mapping.accession.tolist(): raise ValueError('Cache request order differs')
    excluded=validated_conflict_runs(truth.to_dict('records'),metadata.to_dict('records'),mapping.to_dict('records'),
              input.read_tsv(ROOT/'benchmark_smoke/biosample_diagnostics/goat_host_conflicts.tsv').to_dict('records'),biosample_rows=frame.to_dict('records'))
    manifest['excluded_caged_runs']=sorted(excluded)
    manifest['selected_metadata_cache_sha256']=digest(cache)
    api_error=None
    try:
        mistral.read_api_key(args.api_key_file)
        from mistralai.client import Mistral
    except Exception as exc: api_error=type(exc).__name__+'; API credentials or SDK unavailable'
    manifest['api_credentials_available']=api_error is None
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    for mode in ['tsv','biosample']:
        for method in ['nli','llm','mistral']:
            for condition in ['unfiltered','filtered']:
                key=f'{mode}_{method}_{condition}'
                output=work/condition/f'{mode}_{method}.tsv'
                if manifest['runs'].get(key,{}).get('status')=='complete' and output.exists(): continue
                if method=='mistral' and api_error:
                    manifest['runs'][key]=dict(status='incomplete',reason=api_error)
                    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
                    continue
                if output.exists(): raise FileExistsError(output)
                command=[sys.executable,str(Path(__file__).resolve()),'--run-one','--work-dir',str(work),
                         '--taxonomy-dir',str(args.taxonomy_dir),'--input-mode',mode,'--method',method,
                         '--condition',condition,'--output',str(output)]
                if method=='mistral': command+=['--api-key-file',str(args.api_key_file)]
                print('Starting '+key,flush=True)
                started=time.monotonic()
                with (Path(str(output)+'.console.log')).open('w') as handle:
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                manifest['runs'][key]=dict(status='complete' if result.returncode==0 else 'incomplete',
                                          returncode=result.returncode,elapsed_seconds=time.monotonic()-started,
                                          command=[sys.executable,str(ROOT/'metalyzer.py'),*arguments(mode,method,condition,output,args)],
                                          benchmark_child_command=command,finished=datetime.now(timezone.utc).isoformat())
                manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
                print(f'{key}: {manifest["runs"][key]["status"]}; {time.monotonic()-started:.1f}s',flush=True)
    print('All 12 benchmark outcomes recorded',flush=True)


if __name__=='__main__': main()
