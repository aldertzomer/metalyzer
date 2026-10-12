import sys
sys.path.insert(0,'/home/zomer007/PycharmProjects/metalyzer')
from pathlib import Path
from collections import OrderedDict
import pandas as pd
from modules import input, nli
from modules.records import build_record
from modules.contracts import MetadataBatch
from transformers import AutoTokenizer

root=Path('/home/zomer007/PycharmProjects/metalyzer')
out=root/'benchmark_smoke/biosample_diagnostics'
bio=pd.read_pickle(out/'selected_metadata.pkl')
rows=input.read_tsv(out/'record_comparison.tsv')
sources=input.load_sources(root/'sources.tsv','accession')
model=nli.NLIModel(nli.NLIConfig(device=0,batch_size=16,min_score=.2))
tokenizer=AutoTokenizer.from_pretrained(nli.DEFAULT_NLI_MODEL,local_files_only=True)
hypotheses=[nli.HYPOTHESIS_TEMPLATE.format(label) for label in sources.labels]
lengths={}
for name,column in [('bio','bio_record'),('tsv','tsv_record')]:
 lengths[name]=[len(tokenizer(text,add_special_tokens=False)['input_ids']) for text in rows[column]]
 print('TOKENS',name,'median',pd.Series(lengths[name]).median(),'max',max(lengths[name]),flush=True)
 print('PAIR_NEEDS_TRUNCATION',name,'any_label',sum(any(len(tokenizer(text,hypothesis,truncation=False)['input_ids'])>tokenizer.model_max_length for hypothesis in hypotheses) for text in rows[column]),flush=True)
print('TOKENIZER_MAX_LENGTH',tokenizer.model_max_length,flush=True)
rows['bio_tokens']=lengths['bio']; rows['tsv_tokens']=lengths['tsv']
rows.to_csv(out/'record_comparison.tsv',sep='\t',index=False)
chosen=OrderedDict()
for title,mask,count in [('turkey_to_cat',(rows.final_species=='turkey')&(rows.bio_nli_best_hit=='cat'),5),
                         ('sheep_to_goat',(rows.final_species=='sheep')&(rows.bio_nli_best_hit=='goat'),3),
                         ('pig_to_goat',(rows.final_species=='pig')&(rows.bio_nli_best_hit=='goat'),2),
                         ('other_animal_to_environment',(rows.final_species=='other_animal')&(rows.bio_nli_best_hit=='environment'),3),
                         ('wildbird_to_environment',(rows.final_species=='wildbird')&(rows.bio_nli_best_hit=='environment'),3),
                         ('chicken_to_cat',(rows.final_species=='chicken')&(rows.bio_nli_best_hit=='cat'),1),
                         ('goat_metadata_conflict',(rows.final_species=='goat')&(rows.bio_metadata_host=='Gallus gallus'),1),
                         ('cattle_control',(rows.final_species=='cattle')&(rows.bio_nli_best_hit=='cattle'),1),
                         ('dog_control',(rows.final_species=='dog')&(rows.bio_nli_best_hit=='dog'),1)]:
 for idx in rows.index[mask][:count]: chosen[idx]=title
entries=[]
profile={'title','model','package','package_display_name'}
source_fields=['host','host_scientific_name','host_tax_id','isolation_source','sample_type','specimen','tissue','body_site','host_description','host_tissue_sampled','host_body_product','environmental_medium']
for idx,case in chosen.items():
 for variant in ['original','no_profile','source_fields_only']:
  if variant=='original': record=rows.loc[idx,'bio_record']
  elif variant=='no_profile': record=build_record(bio.iloc[idx].drop(labels=[c for c in profile if c in bio]),excluded_fields={'accession','tax_id','host_tax_id'})
  else: record=build_record(bio.iloc[idx][[c for c in source_fields if c in bio]],excluded_fields={'accession','tax_id','host_tax_id'})
  entries.append({'run_accession':rows.loc[idx,'run_accession'],'accession':rows.loc[idx,'accession'],
                  'true_source':rows.loc[idx,'final_species'],'case':case,'variant':variant,'record':record,
                  'saved_original_call':rows.loc[idx,'bio_nli_best_hit']})
frame=pd.DataFrame(entries)
batch=MetadataBatch(frame,pd.Series(frame.record.values,index=frame.index,name='record'))
result=nli.run(batch,sources,nli.NLIConfig(device=0,batch_size=16,min_score=.2),model=model)
frame['prediction']=result.table.best_hit.values
frame['top_score']=result.table[list(sources.names)].max(axis=1).values
frame.to_csv(out/'nli_ablation.tsv',sep='\t',index=False)
print(frame[['case','variant','true_source','prediction','top_score']].to_string(index=False),flush=True)
print('COUNTS',frame.groupby('variant').apply(lambda f:int((f.true_source==f.prediction).sum())).to_dict(),flush=True)
