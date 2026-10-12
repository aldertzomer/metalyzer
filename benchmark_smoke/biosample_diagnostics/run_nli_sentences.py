import sys
sys.path.insert(0,'/home/zomer007/PycharmProjects/metalyzer')
from pathlib import Path
import pandas as pd
from modules import input,nli
from modules.contracts import MetadataBatch
root=Path('/home/zomer007/PycharmProjects/metalyzer')
out=root/'benchmark_smoke/biosample_diagnostics'
examples=[('turkey_latin','turkey','host: Meleagris gallopavo; isolation source: Animal-Turkey-Young Turkey'),
          ('turkey_english','turkey','host: turkey; isolation source: Animal-Turkey-Young Turkey'),
          ('turkey_no_host','turkey','isolation source: Animal-Turkey-Young Turkey'),
          ('turkey_sentence','turkey','The biological source of this sample is turkey.'),
          ('chicken_latin','chicken','host: Gallus gallus; isolation source: feces'),
          ('chicken_english','chicken','host: chicken; isolation source: feces'),
          ('chicken_sentence','chicken','The biological source of this sample is chicken.'),
          ('sheep_latin','sheep','host: Ovis aries; isolation source: feces'),
          ('sheep_english','sheep','host: sheep; isolation source: feces'),
          ('pig_latin','pig','host: Sus scrofa domesticus; isolation source: feces'),
          ('pig_english','pig','host: pig; isolation source: feces'),
          ('goose_fields','wildbird','isolation source: Goose faeces'),
          ('goose_sentence','wildbird','The sample is goose faeces from a wild bird.'),
          ('primate_fields','other_animal','host: Non-human primate; isolation source: feces'),
          ('primate_sentence','other_animal','The sample is feces from a non-human primate.'),
          ('cat_latin','cat','host: Felis catus; isolation source: feces'),
          ('cat_english','cat','host: cat; isolation source: feces')]
frame=pd.DataFrame(examples,columns=['case','expected','record'])
frame['accession']=frame.case
sources=input.load_sources(root/'sources.tsv','accession')
config=nli.NLIConfig(device=0,batch_size=16,min_score=.2)
batch=MetadataBatch(frame,frame.record)
result=nli.run(batch,sources,config)
frame['prediction']=result.table.best_hit.values
frame['top_score']=result.table[list(sources.names)].max(axis=1).values
frame.to_csv(out/'nli_sentinel_texts.tsv',sep='\t',index=False)
print(frame[['case','expected','prediction','top_score']].to_string(index=False))
