import sys
sys.path.insert(0, '/home/zomer007/PycharmProjects/metalyzer')
from pathlib import Path
from collections import Counter
import pandas as pd
from modules import input, deterministic_source
from modules.biosample import load_biosamples
from modules.records import build_record, is_empty_like
from benchmark_smoke.benchmark_evaluation import align_predictions, read_tsv

root=Path('/home/zomer007/PycharmProjects/metalyzer')
output=root/'benchmark_smoke/biosample_diagnostics'
output.mkdir(exist_ok=True)
meta=input.read_tsv(root/'benchmark.tsv')
truth=input.read_tsv(root/'benchmark_true_labels.tsv')
mapping=input.read_tsv(root/'benchmark_smoke/six_method_evaluation/accession_mapping.tsv')
bio=load_biosamples(mapping.accession.tolist())
assert bio.accession.tolist()==mapping.accession.tolist()
bio.to_pickle(output/'selected_metadata.pkl')
bio_batch=input.prepare_batch(bio,id_col='accession')
tsv_batch=input.prepare_batch(meta,id_col='run_accession')
joined=mapping.merge(truth[['run_accession','final_species']],on='run_accession',validate='one_to_one')
for method,suffix in [('nli','nli'),('llm','llm'),('mistral','mistral_api')]:
 for mode,filename in [('tsv',f'benchmark_{suffix}.tsv'),('bio',f'benchmark_biosample_{suffix}.tsv')]:
  _,aligned=align_predictions(truth.to_dict('records'),read_tsv(root/filename),mapping.to_dict('records') if mode=='bio' else None)
  frame=pd.DataFrame(aligned.values())
  columns=['run_accession','best_hit','source_method','source_evidence','source_llm_score','nli_verification_score']
  frame=frame[columns].rename(columns={c:mode+'_'+method+'_'+c for c in columns if c!='run_accession'})
  joined=joined.merge(frame,on='run_accession',validate='one_to_one')
joined['bio_record']=bio_batch.records.values
joined['bio_full_record']=[build_record(row,max_value_chars=300,max_record_chars=100000,excluded_fields={'accession','tax_id','host_tax_id'}) for _,row in bio.iterrows()]
joined['bio_record_truncated']=joined.bio_full_record.str.len()>2000
joined['bio_record_chars']=joined.bio_record.str.len()
joined['bio_full_record_chars']=joined.bio_full_record.str.len()
ts=meta[['run_accession']].copy()
ts['tsv_record']=tsv_batch.records.values
ts['tsv_full_record']=[build_record(row,max_value_chars=300,max_record_chars=100000,excluded_fields={'run_accession','tax_id','host_tax_id'}) for _,row in meta.iterrows()]
joined=joined.merge(ts,on='run_accession',validate='one_to_one')
joined['tsv_record_truncated']=joined.tsv_full_record.str.len()>2000
for c in ['host','host_scientific_name','host_tax_id','host_taxid','isolation_source','sample_name','title','organism','tax_id','description','model','package','package_display_name','owner','source_type','ifsac_category']:
 joined['bio_metadata_'+c]=bio[c].values if c in bio else ''
fields=[c for c in bio if any(x in c.lower() for x in ['host','tax','source','sample','specimen','tissue','body','description','title'])]
print('Bio metadata relevant columns:',fields)
for c in ['host','host_scientific_name','host_tax_id','isolation_source','sample_type','title','organism','tax_id']:
 print('BIO_FIELD',c,int(bio[c].map(lambda v:not is_empty_like(v)).sum()) if c in bio else 0)
joined=joined.merge(meta[['run_accession','host_scientific_name','host_tax_id','isolation_source','sample_title','study_title']].rename(columns={c:'tsv_metadata_'+c for c in ['host_scientific_name','host_tax_id','isolation_source','sample_title','study_title']}),on='run_accession',validate='one_to_one')
joined.to_csv(output/'record_comparison.tsv',sep='\t',index=False)
print('TRUNCATED BIO/TSV',joined.bio_record_truncated.sum(),joined.tsv_record_truncated.sum())
print('LENGTH BIO median max',joined.bio_record_chars.median(),joined.bio_record_chars.max(),'TSV',joined.tsv_record.str.len().median(),joined.tsv_record.str.len().max())
for method in ['nli','llm','mistral']:
 b=joined['bio_'+method+'_best_hit']; t=joined['tsv_'+method+'_best_hit']; y=joined.final_species
 print('COUNTS',method,'both correct',int(((b==y)&(t==y)).sum()),'bio fixes',int(((b==y)&(t!=y)).sum()),'bio breaks',int(((b!=y)&(t==y)).sum()),'both wrong',int(((b!=y)&(t!=y)).sum()))
 print('ERROR PAIRS',method,Counter(zip(y[b!=y],b[b!=y])).most_common(12))
 for tax in [True,False]:
  ix=joined['tsv_'+method+'_source_method'].eq('host_tax_id')==tax
  print('TAX STRATUM',method,tax,'n',int(ix.sum()),'tsv correct',int((t[ix]==y[ix]).sum()),'bio correct',int((b[ix]==y[ix]).sum()))
 for trunc in [True,False]:
  ix=joined.bio_record_truncated==trunc
  print('TRUNC STRATUM',method,trunc,'n',int(ix.sum()),'bio correct',int((b[ix]==y[ix]).sum()))
 ix=(b!=y)&(t==y)
 print('EXAMPLES',method)
 print(joined.loc[ix,['run_accession','accession','final_species','tsv_'+method+'_best_hit','bio_'+method+'_best_hit','bio_metadata_host','bio_metadata_isolation_source','tsv_metadata_host_scientific_name','tsv_metadata_isolation_source','bio_metadata_title']].head(15).to_string(index=False))
print('UNKNOWN truth BIO hosts:',joined.loc[joined.final_species=='unknown','bio_metadata_host'].value_counts().to_dict())
print('GOAT BIO hosts:',joined.loc[joined.final_species=='goat','bio_metadata_host'].value_counts().to_dict())
print('TURKEY BIO hosts:',joined.loc[joined.final_species=='turkey','bio_metadata_host'].value_counts().to_dict())
