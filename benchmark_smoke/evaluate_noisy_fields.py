"""Accession-aligned Variant C evaluation, paired exploratory statistics and examples."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from modules import input,deterministic_source as ds
from benchmark_smoke.benchmark_evaluation import align_predictions
from benchmark_smoke.evaluate_local_backends import evaluate
from benchmark_smoke.metadata_version_evaluation import validated_conflict_runs
from render_benchmark_matrices import LABELS,render_svg


def paired_statistics(rows):
    before=rows.unfiltered_correct
    after=rows.filtered_correct
    improved=int((~before & after).sum())
    worsened=int((before & ~after).sum())
    n=len(rows)
    difference=(improved-worsened)/n if n else 0.0
    if n:
        draws=np.random.default_rng(12345).multinomial(n,[worsened/n,1-(improved+worsened)/n,improved/n],size=10000)
        ci=np.quantile((draws[:,2]-draws[:,0])/n,[.025,.975]).tolist()
    else: ci=[0.0,0.0]
    discordant=improved+worsened
    p=min(1.0,2*sum(math.comb(discordant,i) for i in range(min(improved,worsened)+1))/(2**discordant)) if discordant else 1.0
    unknown_before=rows.unfiltered_best_hit.eq('unknown')
    unknown_after=rows.filtered_best_hit.eq('unknown')
    return dict(records=n,accuracy_difference=difference,paired_bootstrap_95pct_low=ci[0],
                paired_bootstrap_95pct_high=ci[1],mcnemar_exact_p=p,
                wrong_to_correct=improved,correct_to_wrong=worsened,
                wrong_to_different_wrong=int((~before & ~after & rows.changed_prediction).sum()),
                unchanged_correct=int((before & after & ~rows.changed_prediction).sum()),
                unchanged_wrong=int((~before & ~after & ~rows.changed_prediction).sum()),
                changed_predictions=int(rows.changed_prediction.sum()),
                unknown_to_assigned=int((unknown_before & ~unknown_after).sum()),
                assigned_to_unknown=int((~unknown_before & unknown_after).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,default=ROOT/'benchmark_smoke/noisy_field_exclusion')
    args=parser.parse_args()
    work=args.work_dir.resolve()
    manifest=json.loads((work/'manifest.json').read_text())
    truth=input.read_tsv(ROOT/'benchmark_true_labels.tsv')
    metadata=input.read_tsv(ROOT/'benchmark.tsv')
    mapping=input.read_tsv(work/'accession_mapping.tsv').to_dict('records')
    excluded=validated_conflict_runs(truth.to_dict('records'),metadata.to_dict('records'),mapping,
              input.read_tsv(ROOT/'benchmark_smoke/biosample_diagnostics/goat_host_conflicts.tsv').to_dict('records'))
    if sorted(excluded)!=manifest['excluded_caged_runs']: raise ValueError('Live-validated exclusions differ')
    human=input.read_tsv(ROOT/'benchmark_smoke/biosample_diagnostics/unknown_human_evidence.tsv')
    human_runs=set(human.run_accession)
    if len(human_runs)!=41 or human_runs & excluded: raise ValueError('Human-reference cohort must remain included')
    if not truth.set_index('run_accession').loc[list(human_runs)].final_species.eq('unknown').all(): raise ValueError('Human cohort was relabelled')
    cache=pd.read_pickle(work/'input_cache/biosample.pkl')
    frames={'tsv':metadata,'biosample':cache}
    scope_mapping={'tsv':[dict(run_accession=r.run_accession,accession=r.sample_accession) for r in metadata.itertuples()],
                   'biosample':mapping}
    record_sets={}
    for mode,frame in frames.items():
        for condition in ['unfiltered','filtered']:
            batch=input.prepare_batch(frame,id_col='run_accession' if mode=='tsv' else 'accession',
                                      exclude_noisy_fields=condition=='filtered')
            record_sets[(mode,condition)]={mapped['run_accession']:text for mapped,text in zip(scope_mapping[mode],batch.records)}
    summaries=[]; pairs=[]; examples=[]; pathways=[]; outcomes=[]
    for mode in ['tsv','biosample']:
        for method in ['nli','llm','mistral']:
            aligned_conditions={}
            for condition in ['unfiltered','filtered']:
                key=f'{mode}_{method}_{condition}'
                state=manifest['runs'].get(key,dict(status='incomplete',reason='not started'))
                outcomes.append(dict(input=mode,method=method,condition=condition,status=state['status'],reason=state.get('reason',''),recorded=True))
                if state['status']!='complete': continue
                path=work/condition/f'{mode}_{method}.tsv'
                predictions=input.read_tsv(path)
                _,aligned=align_predictions(truth.to_dict('records'),predictions.to_dict('records'),mapping)
                merged=truth.merge(pd.DataFrame(aligned.values()),on='run_accession',validate='one_to_one')
                if 'accession' not in merged: merged['accession']=merged.run_accession.map({r['run_accession']:r['accession'] for r in mapping})
                aligned_conditions[condition]=merged.set_index('run_accession')
                stats=json.loads(Path(str(path)+'.filtering.json').read_text())
                log=Path(str(path)+'.log').read_text()
                def duration(marker):
                    found=re.search(re.escape(marker)+r': ([0-9.]+)',log)
                    return float(found[1]) if found else None
                usage=re.search(r'Mistral API usage: (\{[^\n]+\})',log)
                api=json.loads(usage[1]) if usage else {}
                pending=merged.source_method.eq(method)
                if method=='mistral' and api.get('completed')!=int(pending.sum()): raise ValueError('API completed requests differ from classified rows')
                for evaluation,exclude in [('original',None),('metadata_version_adjusted',excluded)]:
                    out=work/'evaluation'/mode/method/condition/evaluation
                    out.mkdir(parents=True,exist_ok=True)
                    measured=evaluate(truth,path,'best_hit',out,False,mapping,exclude)
                    selected=merged if exclude is None else merged[~merged.run_accession.isin(exclude)]
                    measured.update(input=mode,method=method,condition=condition,
                        file=str(path.relative_to(ROOT)),
                        source_model_calls_full=int(pending.sum()),
                        nli_verification_calls_full=int(merged.nli_verification_score.ne('NA').sum()),
                        total_seconds=state['elapsed_seconds'],
                        source_model_seconds=duration('Source model stage seconds (including weight loading)'),
                        verification_seconds=duration('NLI verification stage seconds (including any weight loading)'),
                        all_record_median_chars=stats['all_records']['after_length_median'],
                        all_record_mean_chars=stats['all_records']['after_length_mean'],
                        all_record_truncated_fraction=stats['all_records']['after_truncated']/len(merged),
                        model_record_median_chars=stats['source_model_records']['after_length_median'],
                        model_record_mean_chars=stats['source_model_records']['after_length_mean'],
                        model_record_truncated_fraction=stats['source_model_records']['after_truncated']/int(pending.sum()),
                        nli_exposed_model_records=stats['nli_token_exposure_source_model'].get('records_any_candidate_exceeds_limit'),
                        api_requests=api.get('requests'),api_retries=api.get('retries'),
                        api_prompt_tokens=api.get('prompt_tokens'),api_completion_tokens=api.get('completion_tokens'),
                        api_total_tokens=api.get('total_tokens'))
                    summaries.append(measured)
                    matrix=pd.crosstab(selected.final_species,selected.best_hit).reindex(index=LABELS,columns=LABELS,fill_value=0)
                    perclass=input.read_tsv(out/f'{path.stem}_per_class.tsv').set_index('source')
                    perclass=perclass.reindex(LABELS)
                    perclass[['true_rows','called_rows','correct']]=perclass[['true_rows','called_rows','correct']].fillna(0)
                    # Explicit one-vs-rest accuracy; true-source recall is already present.
                    perclass['one_vs_rest_accuracy']=[(len(selected)-matrix.loc[label].sum()-matrix[label].sum()+2*matrix.loc[label,label])/len(selected) for label in perclass.index]
                    perclass.to_csv(out/f'{path.stem}_per_class.tsv',sep='\t',index_label='source',na_rep='NA')
                    figure_dir=work/'figures'/evaluation
                    figure_dir.mkdir(parents=True,exist_ok=True)
                    totals=matrix.sum(axis=1).to_dict()
                    cells={label:matrix.loc[label].to_dict() for label in LABELS}
                    for percentages,suffix in [(False,'counts'),(True,'percent')]:
                        render_svg(figure_dir/f'{mode}_{method}_{condition}_{suffix}.svg',
                                   f'{mode.upper()} {method.upper()} · {condition} · {evaluation} (n={len(selected)})',totals,cells,percentages)
                    for route,group in selected.groupby('source_method'):
                        assigned=group.best_hit.ne('unknown'); correct=group.best_hit.eq(group.final_species)
                        pathways.append(dict(input=mode,method=method,condition=condition,evaluation=evaluation,
                                             pathway=route,records=len(group),accuracy=float(correct.mean()),
                                             assigned_precision=float(correct[assigned].mean()) if assigned.any() else None))
            if len(aligned_conditions)!=2: continue
            before=aligned_conditions['unfiltered']; after=aligned_conditions['filtered'].loc[before.index]
            if before.columns.tolist()!=after.columns.tolist(): raise ValueError('Output schemas differ')
            deterministic=before.source_method.isin(ds.DETERMINISTIC_METHODS)
            if not deterministic.equals(after.source_method.isin(ds.DETERMINISTIC_METHODS)): raise ValueError('Taxonomy coverage changed')
            for column in ['best_hit','source_method','source_evidence','source_llm_score']:
                if not before.loc[deterministic,column].equals(after.loc[deterministic,column]): raise ValueError('Taxonomy assignment changed')
            for column in ['year','country']:
                if not before[column].equals(after[column]): raise ValueError(f'{column} changed')
            paired=pd.DataFrame(dict(run_accession=before.index,accession=before.accession,
                true_source=before.final_species,unfiltered_best_hit=before.best_hit,filtered_best_hit=after.best_hit,
                unfiltered_source_method=before.source_method,filtered_source_method=after.source_method,
                unfiltered_correct=before.best_hit.eq(before.final_species),filtered_correct=after.best_hit.eq(after.final_species),
                unfiltered_llm_score=before.source_llm_score,filtered_llm_score=after.source_llm_score,
                unfiltered_verification=before.nli_verification_score,filtered_verification=after.nli_verification_score,
                model_classified=~deterministic,changed_prediction=before.best_hit.ne(after.best_hit))).reset_index(drop=True)
            paired.to_csv(work/'evaluation'/f'paired_{mode}_{method}.tsv',sep='\t',index=False)
            paired[paired.run_accession.isin(human_runs)].to_csv(work/'evaluation'/f'human_reference_unknown_{mode}_{method}.tsv',sep='\t',index=False)
            for evaluation,exclude in [('original',None),('metadata_version_adjusted',excluded)]:
                selected=paired if exclude is None else paired[~paired.run_accession.isin(exclude)]
                for scope,group in [('all_records',selected),('source_model_records',selected[selected.model_classified])]:
                    pairs.append(dict(input=mode,method=method,evaluation=evaluation,scope=scope,**paired_statistics(group)))
            cases={'wrong_to_correct':~paired.unfiltered_correct & paired.filtered_correct,
                   'correct_to_wrong':paired.unfiltered_correct & ~paired.filtered_correct,
                   'unknown_to_assigned':paired.unfiltered_best_hit.eq('unknown') & paired.filtered_best_hit.ne('unknown'),
                   'assigned_to_unknown':paired.unfiltered_best_hit.ne('unknown') & paired.filtered_best_hit.eq('unknown'),
                   'environment_corrected':paired.unfiltered_best_hit.eq('environment') & paired.filtered_correct & paired.changed_prediction}
            for case,mask in cases.items():
                for row in paired[mask].head(2).to_dict('records'):
                    old_text=record_sets[(mode,'unfiltered')][row['run_accession']]
                    new_text=record_sets[(mode,'filtered')][row['run_accession']]
                    examples.append(dict(input=mode,method=method,case=case,**row,
                                         unfiltered_model_text=old_text,filtered_model_text=new_text,
                                         had_pathogen_env_profile='Pathogen.env' in old_text or 'Pathogen: environmental/food/other' in old_text,
                                         original_truncated=old_text.endswith('…'),filtered_truncated=new_text.endswith('…')))
    (work/'evaluation/summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    pd.DataFrame(summaries).drop(columns=['source_methods'],errors='ignore').to_csv(work/'evaluation/summary.tsv',sep='\t',index=False,na_rep='NA')
    pd.DataFrame(pairs).to_csv(work/'evaluation/paired_summary.tsv',sep='\t',index=False,na_rep='NA')
    pd.DataFrame(pathways).to_csv(work/'evaluation/pathway_metrics.tsv',sep='\t',index=False,na_rep='NA')
    pd.DataFrame(outcomes).to_csv(work/'evaluation/outcomes.tsv',sep='\t',index=False,na_rep='NA')
    pd.DataFrame(examples).to_csv(work/'evaluation/error_examples.tsv',sep='\t',index=False,na_rep='NA')
    print(pd.DataFrame(summaries)[['input','method','condition','evaluation','overall_accuracy','precision_assigned']].to_string(index=False))
    print('All paired deterministic assignments, date/country outputs and output schemas verified unchanged.')


if __name__=='__main__': main()
