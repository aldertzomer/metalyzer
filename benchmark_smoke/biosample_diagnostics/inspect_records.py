import sys
sys.path.insert(0,'/home/zomer007/PycharmProjects/metalyzer')
from pathlib import Path
from collections import Counter
import pandas as pd
from modules import input, deterministic_source
from modules.records import is_empty_like, build_record
root=Path('/home/zomer007/PycharmProjects/metalyzer')
out=root/'benchmark_smoke/biosample_diagnostics'
bio=pd.read_pickle(out/'selected_metadata.pkl')
rows=input.read_tsv(out/'record_comparison.tsv')
sources=input.load_sources(root/'sources.tsv','accession')
taxonomy=deterministic_source.NCBITaxonomy('/home/zomer007/metalyzer/taxonomy')
anchors=deterministic_source.configured_anchors(sources,taxonomy)
print('HOST_TAXID_FIELD', 'host_taxid' in bio,'NONEMPTY',int(bio.host_taxid.map(lambda x:not is_empty_like(x)).sum()))
print('HOST_TAXID_VALUES',bio.host_taxid.value_counts().to_dict())
variant=bio.copy()
variant['host_tax_id']=variant['host_taxid']
result=deterministic_source.run(input.prepare_batch(variant,id_col='accession'),sources,taxonomy,anchor_sources=anchors)
rows['alias_taxonomy_call']=''
rows.loc[result.table.index,'alias_taxonomy_call']=result.table.best_hit.values
ix=rows.alias_taxonomy_call!=''
print('ALIAS_ONLY_TAXONOMY',int(ix.sum()),'correct',int((rows.loc[ix,'alias_taxonomy_call']==rows.loc[ix,'final_species']).sum()))
for method in ['nli','llm','mistral']:
 before=rows['bio_'+method+'_best_hit']
 after=before.where(~ix,rows.alias_taxonomy_call)
 print('SAVED_PREDICTION_COUNTERFACTUAL',method,'before',int((before==rows.final_species).sum()),'after',int((after==rows.final_species).sum()),'fixed',int(((before!=rows.final_species)&(after==rows.final_species)).sum()),'broken',int(((before==rows.final_species)&(after!=rows.final_species)).sum()))
print('ALIAS_TAXONOMY_MISMATCHES')
print(rows.loc[ix&(rows.alias_taxonomy_call!=rows.final_species),['run_accession','accession','final_species','alias_taxonomy_call','bio_metadata_host','tsv_metadata_host_scientific_name']].head(8).to_string(index=False))
print('EXAMPLE RECORDS')
for name,mask in [('goat_conflict',(rows.final_species=='goat')&(rows.bio_metadata_host=='Gallus gallus')),
                  ('nli_turkey',(rows.final_species=='turkey')&(rows.bio_nli_best_hit=='cat')),
                  ('nli_wildbird',(rows.final_species=='wildbird')&(rows.bio_nli_best_hit=='environment')&(rows.bio_mistral_best_hit=='wildbird')),
                  ('unknown_human',(rows.final_species=='unknown')&(rows.bio_mistral_best_hit=='human')),
                  ('water_environment',(rows.final_species=='water')&(rows.bio_mistral_best_hit=='environment')),
                  ('llm_wildbird_unknown',(rows.final_species=='wildbird')&(rows.bio_llm_best_hit=='unknown')&(rows.bio_mistral_best_hit=='wildbird')),
                  ('other_animal',(rows.final_species=='other_animal')&(rows.bio_nli_best_hit=='environment')&(rows.bio_mistral_best_hit=='other_animal'))]:
 sample=rows[mask].iloc[0]
 print('\nCASE',name,sample.run_accession,sample.accession,'truth',sample.final_species,'NLI',sample.bio_nli_best_hit,'LLM',sample.bio_llm_best_hit,'API',sample.bio_mistral_best_hit)
 print('BIO',sample.bio_record)
 print('TSV',sample.tsv_record)
 print('LLM',sample.bio_llm_source_evidence,sample.bio_llm_source_llm_score)
 print('VERIFY',sample.bio_nli_nli_verification_score,sample.bio_llm_nli_verification_score,sample.bio_mistral_nli_verification_score)
for label in ['unknown','water','wildbird','goat','other_animal']:
 sub=rows[rows.final_species==label]
 print('ATTR_SOURCE',label,sub.bio_metadata_isolation_source.value_counts().head(12).to_dict())
for method in ['nli','llm','mistral']:
 wrong=rows['bio_'+method+'_best_hit']!=rows.final_species
 scores=pd.to_numeric(rows['bio_'+method+'_nli_verification_score'],errors='coerce')
 print('FALSE_CALLS_VERIFY',method,'wrong total',int(wrong.sum()),'wrong assigned scored',int((wrong&scores.notna()).sum()),'wrong with verify>=0.9',int((wrong&(scores>=.9)).sum()))
print('LLM_ABSTENTION',rows.bio_llm_source_evidence.str.startswith('llm_low_score=').sum())
rows.to_csv(out/'record_comparison.tsv',sep='\t',index=False)
