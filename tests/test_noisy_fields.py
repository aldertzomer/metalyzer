"""CPU-only Variant C selection, byte-identical legacy rendering and pipeline parity."""
import contextlib
from html import unescape
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import metalyzer
from modules import biosample, input, nli, noisy_fields, records, verification
from modules.contracts import SourceVocabulary
import test_taxonomy


def legacy(row, max_value_chars=300, max_record_chars=2000, excluded_fields=()):
    # Frozen pre-Variant-C implementation, not delegation to the new renderer.
    parts = []
    for col, val in row.items():
        if col in excluded_fields or records.is_empty_like(val):
            continue
        value = ' '.join(unescape(str(val)).split())
        if len(value) > max_value_chars:
            value = value[:max_value_chars] + '…'
        parts.append(f'{str(col).replace("_", " ")}: {value}')
    text = '; '.join(parts)
    if len(text) > max_record_chars:
        text = text[:max_record_chars] + '…'
    return text if text.strip() else 'metadata: (empty)'


class NoisyFieldTests(unittest.TestCase):
    def test_all_defined_exclusions_and_normalization(self):
        for name in noisy_fields.EXCLUDED:
            for variant in [name, name.upper().replace(' ', '_'), name.replace(' ', '-')]:
                with self.subTest(name=variant):
                    self.assertTrue(noisy_fields.is_noisy_field(variant))
        for name in ['INSDC_center_name', 'BarcodeSequence', 'LinkerPrimerSequence', 'id.SRA', 'LINK.ncbi',
                     'original_attributes.INSDC center name', 'diagnostic_pcr_protocol_1', 'diagnostic pcr protocol 123']:
            self.assertTrue(noisy_fields.is_noisy_field(name), name)
        for name in ['identity', 'linked habitat', 'id_source', 'diagnosis', 'diagnostic category',
                     'diagnostic_pcr_protocol_1_description', 'plate sampling description', 'well site',
                     'taxonomic host lineage', 'unknown biological trait']:
            self.assertFalse(noisy_fields.is_noisy_field(name), name)

    def test_protected_fields_unknown_fields_collection_locations(self):
        protected = '''host,host_scientific_name,host scientific name,host common name,host_description,specific_host,
host_taxid,host_tax_id,isolation_source,source_type,ifsac_category,IFSAC+ Category,sample_type,sample material,
sample derived from,tissue,tissue_type,body_site,host_body_site,specimen,body_product,host_disease,host health state,
clinical condition,diagnosis,disease,infection,pathogenicity,pathotype,cell type,cell line,culture_collection,
env_broad_scale,env_medium,env_local_scale,environmental_medium,environment_feature,environment_material,habitat,
sewage type,ww_sample_matrix,food_origin,food_source,food_product_type,title,sample_title,description,sample description,
study_title,study name,project name,purpose_of_sampling,study design,collection_date,collection method,latitude,
longitude,lat,lon,geo_loc_name,country,location,sample_name,isolate,strain,new_host_material_description'''
        row = pd.Series({name.strip(): 'source evidence' for name in protected.replace('\n','').split(',')})
        rendered = records.build_record(row, max_record_chars=20000)
        for name in row.index:
            self.assertFalse(noisy_fields.is_noisy_field(name), name)
            self.assertIn(str(name).replace('_',' ')+': source evidence', rendered)

    def test_equivalent_originals_and_unique_or_conflicting_values(self):
        row = pd.Series({'host': 'Gallus  gallus', 'isolation_source': 'feces',
                         'original_attributes.Host scientific name': 'gallus gallus',
                         'original_attributes.isolation-source': 'feces',
                         'original_attributes.sample derived from': 'feces',
                         'original_attributes.host': 'Capra hircus',
                         'original_attributes.unanticipated trait': 'unique'})
        rendered = records.build_record(row)
        self.assertNotIn('original attributes.Host scientific name', rendered)
        self.assertNotIn('original attributes.isolation-source', rendered)
        self.assertIn('original attributes.sample derived from: feces', rendered)
        self.assertIn('original attributes.host: Capra hircus', rendered)
        self.assertIn('unanticipated trait: unique', rendered)
        row = pd.Series({'host':'Gallus gallus','host_scientific_name':'Capra hircus',
                         'original_attributes.host':'Gallus gallus'})
        self.assertIn('original attributes.host: Gallus gallus', records.build_record(row))
        row = pd.Series({'host':'Gallus gallus', 'original_attributes.host':'Gallus gallus | Capra hircus'})
        self.assertIn('original attributes.host:', records.build_record(row))

    def test_disabled_matches_frozen_legacy_bytes(self):
        rows = [pd.Series({'accession':'SAMN1','owner':'lab','model':'Pathogen.env','host':'cat',
                           'original_attributes.host':'cat','text':' R&amp;D\n  value ', 'zero':0, 'false':False}),
                pd.Series({'title':'x'*500,'package_display_name':'environmental/food/other', 'host':'cattle'}),
                pd.Series({'missing':None,'blank':' ','na':'NA'}), pd.Series(dtype=object)]
        for row in rows:
            for value_limit, record_limit in [(300,2000),(5,12),(1,1)]:
                self.assertEqual(records.build_record(row,value_limit,record_limit,{'accession'},False),
                                 legacy(row,value_limit,record_limit,{'accession'}))

    def test_before_truncation_and_metadata_unchanged(self):
        raw = pd.DataFrame({'run_accession':['a','b'], 'owner':['x'*300,'lab'],
                            'host':['sheep','cat'], 'collection_date':['2024','2023'], 'lat':['51','52']})
        before = raw.copy(deep=True)
        filtered = input.prepare_batch(raw,id_col='run_accession',max_record_chars=30)
        original = input.prepare_batch(raw,id_col='run_accession',max_record_chars=30,exclude_noisy_fields=False)
        self.assertTrue(filtered.records.iloc[0].startswith('host: sheep'))
        self.assertTrue(original.records.iloc[0].startswith('owner:'))
        pd.testing.assert_frame_equal(filtered.metadata,before)
        pd.testing.assert_frame_equal(original.metadata,before)
        self.assertEqual(filtered.subset([1]).record_audit.index.tolist(),[1])
        self.assertEqual(filtered.record_audit.administrative_technical.tolist(),[1,1])

    def test_biosample_and_tsv_use_identical_field_rules(self):
        canonical = biosample.canonical_from_parquet(dict(accession='SAMN1', model='Pathogen.env',
                    attributes=[('host','cat'),('unknown trait','urine')],
                    original_attributes=[('Host','cat')], identifiers=[('id.SRA','SRS1')]))
        frame = pd.DataFrame([canonical])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metadata.tsv'
            frame.to_csv(path,sep='\t',index=False)
            loaded = input.read_tsv(path)
        for enabled in [True,False]:
            self.assertEqual(input.prepare_batch(frame,id_col='accession',exclude_noisy_fields=enabled).records.tolist(),
                             input.prepare_batch(loaded,id_col='accession',exclude_noisy_fields=enabled).records.tolist())

    def test_taxonomy_both_modes_and_verification_same_record(self):
        fixture = test_taxonomy.HostNameTaxonomyTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frame = pd.DataFrame({'accession':['a','b','c'], 'host':['Gallus gallus','Meleagris gallopavo','unknown'],
                              'host_tax_id':['9031','',''], 'owner':['lab']*3})
        sources = fixture.sources
        results = []
        for enabled in [True,False]:
            batch = input.prepare_batch(frame,id_col='accession',exclude_noisy_fields=enabled)
            seen = []
            def fake(texts,**kwargs):
                seen.extend(texts)
                return [dict(labels=kwargs['candidate_labels'],scores=[.9]*len(kwargs['candidate_labels'])) for _ in texts]
            with patch.object(nli,'pipeline') as factory, contextlib.redirect_stdout(io.StringIO()):
                factory.return_value.side_effect = fake
                model = nli.NLIModel(nli.NLIConfig(device=-1))
                result = metalyzer.classify_sources(batch,sources,taxonomy=fixture.taxonomy,nli_model=model)
                verification.run(batch,sources,result,model=model)
            self.assertTrue(all(text in batch.records.tolist() for text in seen))
            self.assertEqual(seen[0],batch.records.iloc[2])  # only unresolved source model row
            results.append(result.table.loc[[0,1]])
        pd.testing.assert_frame_equal(*results)

    def test_cli_provenance_audits_and_normal_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'metadata.tsv').write_text('run_accession\thost\tmodel\tcollection_date\na\tcat\tPathogen.env\t2024\n')
            (root/'sources.tsv').write_text('source\ncat\n')
            for disabled in [False,True]:
                command=['--metadata',str(root/'metadata.tsv'),'--sources',str(root/'sources.tsv'),
                         '--out',str(root/'out.tsv'),'--id-col','run_accession','--disable-verify-source','--device','-1']
                if disabled: command.append('--disable-noisy-field-exclusion')
                with patch.object(nli,'pipeline') as factory, contextlib.redirect_stdout(io.StringIO()):
                    factory.return_value.return_value=[dict(labels=['cat'],scores=[.99])]
                    metalyzer.main(command)
                output=input.read_tsv(root/'out.tsv')
                self.assertEqual(output.run_accession.tolist(),['a'])
                self.assertIn('nli_verification_score',output)
                self.assertNotIn('administrative_technical',output)
                stats=json.loads((root/'out.tsv.filtering.json').read_text())
                self.assertEqual(stats['enabled'],not disabled)
                self.assertEqual(stats['all_records']['administrative_technical'],int(not disabled))


if __name__ == '__main__': unittest.main()
