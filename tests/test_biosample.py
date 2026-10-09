"""Offline BioSample storage, builder and CLI integration tests."""
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import metalyzer
from modules import biosample as bio, biosample_build as build, input, deterministic_source
from modules.contracts import SourceResult, empty_source_table

XML = '''<BioSampleSet><BioSample accession="SAMN1" id="1"><Description><Title>cat sample</Title><Organism taxonomy_name="pathogen" taxonomy_id="197"/></Description><Attributes><Attribute attribute_name="Host" harmonized_name="host">cat</Attribute><Attribute attribute_name="host_tax_id">9685</Attribute><Attribute attribute_name="weird">0</Attribute><Attribute attribute_name="weird">false</Attribute><Attribute attribute_name="accession">wrong</Attribute></Attributes><Ids><Id db="SRA">SRS1</Id></Ids><Links><Link type="url">https://example.org</Link></Links></BioSample><BioSample accession="SAMEA2"><Attributes><Attribute attribute_name="isolation_source">urine</Attribute></Attributes></BioSample></BioSampleSet>'''


class BioSampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'biosample.parquet'
        maps = pa.map_(pa.string(), pa.string())
        schema = pa.schema([('accession', pa.string()), ('organism', pa.string()), ('tax_id', pa.string()),
                            ('title', pa.string()), *[(k, maps) for k in ('attributes', 'original_attributes', 'identifiers', 'links')],
                            ('arbitrary', pa.string()), ('empty', pa.string())])
        rows = [{'accession': a, 'organism': 'pathogen', 'tax_id': '197', 'title': 'cat sample',
                 'attributes': [('host', 'cat'), ('host_tax_id', '9685'), ('accession', 'wrong'),
                                ('repeated', 'a'), ('repeated', 'b'), ('literal_na', 'NA'), ('zero', '0'),
                                ('boolean', 'false'), ('blank', '  ')],
                 'original_attributes': [('Host', 'cat')], 'identifiers': [('id.SRA', 'SRS1')],
                 'links': [('link.url', 'https://example.org')], 'arbitrary': 'unexpected', 'empty': None}
                for a in ('SAMN1', 'SAMEA2', 'SAMD3')]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), self.path, row_group_size=1)

    def tearDown(self):
        self.temp.cleanup()

    def test_lookup_order_duplicates_maps_and_priority(self):
        with patch.object(pd, 'read_parquet', side_effect=AssertionError('full read')), patch.object(pq, 'read_table', side_effect=AssertionError('full read')):
            frame = bio.load_biosamples([' SAMD3 ', 'SAMN1', 'SAMD3'], self.path)
        self.assertEqual(frame.accession.tolist(), ['SAMD3', 'SAMN1', 'SAMD3'])
        row = frame.iloc[0]
        self.assertEqual(row['attributes.accession'], 'wrong')
        self.assertEqual(row['repeated'], 'a | b')
        for key, value in [('original_attributes.Host', 'cat'), ('id.SRA', 'SRS1'), ('link.url', 'https://example.org'),
                           ('arbitrary', 'unexpected'), ('literal_na', 'NA'), ('zero', '0'), ('boolean', 'false')]:
            self.assertEqual(row[key], value)
        self.assertNotIn('empty', frame)
        self.assertNotIn('blank', frame)
        self.assertLess(frame.columns.get_loc('host'), frame.columns.get_loc('organism'))
        self.assertLess(frame.columns.get_loc('title'), frame.columns.get_loc('tax_id'))
        self.assertEqual(bio.load_biosamples(['SAMN1'], self.path).accession.tolist(), ['SAMN1'])
        batch = input.prepare_batch(frame, id_col='accession')
        self.assertEqual(batch.metadata.host_tax_id.tolist(), ['9685'] * 3)
        self.assertNotIn('tax id:', batch.records[0])
        taxonomy = unittest.mock.Mock()
        taxonomy.classify_host_taxid.return_value = None
        sources = input.SourceVocabulary(('cat',), 'accession')
        deterministic_source.run(batch, sources, taxonomy, anchor_sources={})
        self.assertEqual(taxonomy.classify_host_taxid.call_args.args[0], '9685')

    def test_missing_and_index_reuse_invalidation(self):
        with self.assertRaisesRegex(ValueError, 'Missing BioSample accessions.*absent'):
            bio.load_biosamples(['SAMN1', 'absent'], self.path)
        with patch.object(pq.ParquetFile, 'iter_batches', side_effect=AssertionError('index reused')):
            bio.ensure_index(self.path)
        self.assertTrue(bio.index_path(self.path).exists())

    def test_nested_and_null_leaves(self):
        record = bio.canonical_from_parquet({'accession': 'a', 'attributes': {'novel': {'nested': [0, False, None, float('nan'), '']}}, 'links': [('link.x', ['a', 'b'])]})
        self.assertEqual(record['novel.nested'], '0 | False')
        self.assertEqual(record['link.x'], 'a | b')

    def test_cli_and_normal_tsv_pipeline(self):
        sources = self.root / 'sources.tsv'
        sources.write_text('source\ncat\n')
        out = self.root / 'result.tsv'
        common = ['--sources', str(sources), '--out', str(out)]
        args = metalyzer.parse_args(['--biosample', 'SAMN1', *common])
        self.assertEqual(args.id_col, 'accession')
        self.assertEqual(args.llm_min_score, .75)
        file = self.root / 'accessions.txt'
        file.write_text('SAMN1\n\nSAMEA2\n')
        metadata = self.root / 'metadata.tsv'
        metadata.write_text('id\thost\n1\tcat\n')
        self.assertEqual(metalyzer.parse_args(['--metadata', str(metadata), '--id-col', 'id', *common]).id_col, 'id')
        with self.assertRaises(SystemExit):
            metalyzer.parse_args(['--metadata', str(metadata), *common])
        with self.assertRaises(SystemExit):
            metalyzer.parse_args(['--metadata', str(metadata), '--biosample', 'a', *common])
        def fake(batch, vocab, **kwargs):
            table = empty_source_table(vocab, batch.metadata.index)
            table['best_hit'] = 'unknown'
            table['source_method'] = 'llm'
            table['source_evidence'] = ''
            return SourceResult(table)
        with patch.object(metalyzer, 'classify_sources', side_effect=fake), patch.object(metalyzer.nli, 'NLIModel'), patch.object(metalyzer.deterministic_source, 'load_taxonomy', return_value=None):
            metalyzer.main(['--biosample-file', str(file), '--biosample-path', str(self.path), *common,
                            '--method', 'llm', '--disable-verify-source'])
        frame = input.read_tsv(out)
        self.assertEqual(frame.accession.tolist(), ['SAMN1', 'SAMEA2'])
        for column in ('cat', 'best_hit', 'source_method', 'source_evidence', 'source_llm_score', 'nli_verification_score', 'year', 'country'):
            self.assertIn(column, frame)
        self.assertIn('Input mode: BioSample', Path(str(out) + '.log').read_text())

    def test_build_gzip_batches_and_roundtrip(self):
        archive = self.root / 'biosample_set.xml.gz'
        with gzip.open(archive, 'wt') as handle:
            handle.write(XML)
        build.convert_archive(archive, self.path, force=True, batch_size=1)
        parquet = pq.ParquetFile(self.path)
        self.assertEqual(parquet.metadata.num_rows, 2)
        self.assertEqual(parquet.num_row_groups, 2)
        frame = bio.load_biosamples(['SAMEA2', 'SAMN1'], self.path)
        self.assertEqual(frame.accession.tolist(), ['SAMEA2', 'SAMN1'])
        self.assertEqual(frame.iloc[1]['weird'], '0 | false')
        self.assertEqual(frame.iloc[1]['original_attributes.Host'], 'cat')
        self.assertEqual(frame.iloc[1]['attributes.accession'], 'wrong')
        self.assertTrue((self.root / 'attribute_catalogue.tsv').exists())
        self.assertTrue(build.DEFAULT_URL.startswith('https://'))
        with patch.object(build.subprocess, 'run', side_effect=AssertionError('download')):
            self.assertEqual(build.download_archive(archive, skip=True), archive)
            self.assertEqual(build.download_archive(archive), archive)

    def test_conversion_failure_preserves_existing(self):
        original = self.path.read_bytes()
        archive = self.root / 'bad.xml.gz'
        with gzip.open(archive, 'wt') as handle:
            handle.write('<BioSampleSet><BioSample accession="a"/></broken>')
        with self.assertRaises(Exception):
            build.convert_archive(archive, self.path, force=True, batch_size=1)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_directory_reuses_file_index_and_shards(self):
        bio.ensure_index(self.path)
        self.assertEqual(bio.index_path(self.root), bio.index_path(self.path))
        self.assertEqual(bio.load_biosamples(['SAMN1'], self.root).accession.tolist(), ['SAMN1'])
        shard = self.root / 'part2.parquet'
        pq.write_table(pa.table({'accession': ['novel'], 'attributes': [{'unexpected': 'retained'}]}), shard)
        frame = bio.load_biosamples(['novel', 'SAMN1'], self.root)
        self.assertEqual(frame.accession.tolist(), ['novel', 'SAMN1'])
        self.assertEqual(frame.iloc[0]['unexpected'], 'retained')

    def test_missing_fails_before_model_creation(self):
        with patch.object(metalyzer.nli, 'NLIModel') as model:
            with self.assertRaisesRegex(ValueError, 'missing-accession'):
                metalyzer.main(['--biosample', 'missing-accession', '--biosample-path', str(self.path),
                                '--sources', 'unused.tsv', '--out', str(self.root / 'failed.tsv')])
            model.assert_not_called()

    def test_streamed_download_resume_and_failure(self):
        archive = self.root / 'archive.gz'
        partial = self.root / 'archive.gz.part'
        partial.write_bytes(b'partial')
        def download(command, **kwargs):
            self.assertNotIn('shell', kwargs)
            self.assertIn('-C', command)
            self.assertEqual(command[-1], str(partial))
            partial.write_bytes(b'complete')
        with patch.object(build.subprocess, 'run', side_effect=download):
            build.download_archive(archive)
        self.assertEqual(archive.read_bytes(), b'complete')
        with patch.object(build.subprocess, 'run', side_effect=RuntimeError('download failed')):
            with self.assertRaises(RuntimeError):
                build.download_archive(archive, force=True)
        self.assertEqual(archive.read_bytes(), b'complete')
