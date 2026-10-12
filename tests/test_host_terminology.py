"""CPU-only operational aliases and conservative unlisted-animal coverage."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import metalyzer
from modules import combine, deterministic_source as ds, input, llm, mistral, nli, verification
from modules.contracts import SourceVocabulary
import test_taxonomy


class HostTerminologyTests(unittest.TestCase):
    def setUp(self):
        test_taxonomy.HostNameTaxonomyTests.setUp(self)
        with (self.directory / 'nodes.dmp').open('a') as nodes, (self.directory / 'names.dmp').open('a') as names:
            for taxid, parent, rank, name in [
                (30538, 33208, 'species', 'Vicugna pacos'),
                (9544, 33208, 'species', 'Macaca mulatta'),
                (9859, 33208, 'species', 'Odocoileus hemionus'),
                (40674, 33208, 'class', 'Mammalia'),
                (9612, 33554, 'family', 'Canidae'),
                (9627, 9612, 'species', 'Vulpes vulpes'),
                (9999, 33208, 'genus', 'Unlisted animal genus'),
                (9200, 8782, 'family', 'Columbidae'),
                (8843, 8782, 'species', 'Anser anser')]:
                nodes.write(f'{taxid} | {parent} | {rank} |\n')
                names.write(f'{taxid} | {name} | | scientific name |\n')
            names.write('30538 | alpaca | | common name |\n9913 | bovine | | common name |\n')
        self.taxonomy = ds.NCBITaxonomy(self.directory)
        self.sources = SourceVocabulary(
            ('chicken', 'turkey', 'pig', 'cattle', 'sheep', 'goat', 'human', 'dog', 'cat', 'other_animal'),
            'accession', ((9031,), (9103,), (9825,), (9913,), (9940,), (9925,), (9606,), (9615,), (9685,), ()),
            ((), (), ('porcine',), ('bovine',), ('ovine',), ('caprine',), (), ('canine',), ('feline',), ()))

    def rows(self, rows, **options):
        frame = pd.DataFrame(rows)
        before = frame.copy(deep=True)
        batch = input.prepare_batch(frame, id_col='accession')
        diagnostics = []
        with contextlib.redirect_stdout(io.StringIO()):
            result = ds.run(batch, self.sources, self.taxonomy, diagnostics=diagnostics, **options)
        pd.testing.assert_frame_equal(frame, before)
        return result.table, diagnostics

    def test_operational_aliases_exact_normalized_and_taxonomy_primary(self):
        table, _ = self.rows([{'host': value} for value in [' Canine ', 'PORCINE', 'ovine', 'Caprine', 'Feline', 'Bovine',
                                                          'porcine-associated wastewater', 'canines', 'unknown']])
        self.assertEqual(table.index.tolist(), list(range(6)))
        self.assertEqual(table.best_hit.tolist(), ['dog', 'pig', 'sheep', 'goat', 'cat', 'cattle'])
        self.assertEqual(table.source_method.tolist(), ['host_alias'] * 5 + ['host_name'])
        self.assertEqual(table.loc[1, 'source_evidence'], 'host=PORCINE; configured_host_alias=pig')
        self.assertNotIn('taxid=', table.loc[1, 'source_evidence'])
        table, _ = self.rows([{'host_scientific_name': 'canine'}, {'title': 'Canine', 'isolation_source': 'Porcine'},
                              {'organism': 'Canine', 'tax_id': '9615'}, {'host': 'feline'}], use_host_aliases=False)
        self.assertTrue(table.empty)

    def test_alias_contradictions_and_compatible_exact_evidence(self):
        rows = [dict(host='Canine', host_tax_id='9685'), dict(host='Canine', host_scientific_name='Vulpes vulpes'),
                dict(host='Porcine', host_taxid='9823'), dict(host='Porcine', host_scientific_name='Sus scrofa'),
                dict(host='Canine', host_tax_id='9615'), dict(host='Canine', host_scientific_name='Canis lupus familiaris'),
                dict(host='Canine', host_scientific_name='Feline')]
        table, diagnostics = self.rows(rows)
        self.assertEqual(table.index.tolist(), [4, 5])
        self.assertEqual(table.source_method.tolist(), ['host_tax_id', 'host_name'])
        self.assertEqual(sum(bool(item['resolution'].conflict) for item in diagnostics), 5)
        # Disabling operational assumptions restores independent exact taxonomy.
        table, _ = self.rows(rows[:1], use_host_aliases=False)
        self.assertEqual(table.best_hit.tolist(), ['cat'])

    def test_ambiguous_names_cannot_be_bypassed_by_an_alias(self):
        table, diagnostics = self.rows([dict(host='Canine', host_scientific_name='poultry')])
        self.assertTrue(table.empty)
        self.assertTrue(diagnostics[0]['resolution'].ambiguous)

    def test_multiple_aliases_and_optional_old_vocabulary(self):
        path = self.directory / 'sources.tsv'
        path.write_text('source\ttaxonomy_anchors\thost_aliases\ndog\t9615\tcanine;  pet DOG\ncat\t9685\tfeline\n')
        vocabulary = input.load_sources(path, 'accession')
        self.assertEqual(vocabulary.host_aliases, (('canine', 'pet dog'), ('feline',)))
        path.write_text('source\ttaxonomy_anchors\ndog\t9615\n')
        self.assertEqual(input.load_sources(path, 'accession').host_aliases, ((),))
        with self.assertRaisesRegex(ValueError, 'Conflicting host alias'):
            SourceVocabulary(('dog', 'cat'), 'accession', host_aliases=(('canine',), (' CANINE ',)))
        for value in ['canine;', ';canine', 'canine;;dog', 'na', '1234', 'dog | cat']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                path.write_text(f'source\thost_aliases\ndog\t{value}\n')
                input.load_sources(path, 'accession')

    def test_specific_unlisted_animals_opt_in_and_conservative_boundaries(self):
        rows = [{'host': value} for value in ['alpaca', 'Macaca mulatta', 'Odocoileus hemionus',
                'Metazoa', 'Mammalia', 'Canidae', 'Unlisted animal genus', 'Sus scrofa',
                'Anas platyrhynchos', 'Anser anser', 'Columbidae', 'Escherichia coli', 'Gallus gallus']]
        table, _ = self.rows(rows, other_animal=True)
        self.assertEqual(table.index.tolist(), [0, 1, 2, 12])
        self.assertEqual(table.source_method.tolist(), ['host_other_animal'] * 3 + ['host_name'])
        self.assertEqual(table.best_hit.tolist(), ['other_animal'] * 3 + ['chicken'])
        table, _ = self.rows(rows)
        self.assertEqual(table.index.tolist(), [12])
        table, _ = self.rows([{'host_tax_id': '30538'}], other_animal=True)
        self.assertTrue(table.empty)
        self.sources = SourceVocabulary(('chicken',), 'accession', ((9031,),))
        table, _ = self.rows([{'host': 'alpaca'}], other_animal=True)
        self.assertTrue(table.empty)

    def test_extra_methods_skip_classifiers_keep_normal_output_and_logging(self):
        batch = input.prepare_batch(pd.DataFrame({'accession': ['a', 'b'], 'host': ['Canine', 'alpaca']}), id_col='accession')
        for method in ['nli', 'llm', 'mistral']:
            with self.subTest(method=method), patch.object(nli, 'run') as nr, patch.object(llm, 'run') as lr, patch.object(mistral, 'run') as mr, contextlib.redirect_stdout(io.StringIO()) as log:
                result = metalyzer.classify_sources(batch, self.sources, method=method, taxonomy=self.taxonomy, other_animal=True)
            nr.assert_not_called(); lr.assert_not_called(); mr.assert_not_called()
            self.assertEqual(result.table.source_method.tolist(), ['host_alias', 'host_other_animal'])
            self.assertTrue(result.table.source_llm_score.isna().all())
            self.assertIn('Configured host aliases: 1', log.getvalue())
            self.assertIn('Specific unlisted animals: 1', log.getvalue())
            self.assertIn('taxonomy total: 2', log.getvalue())
            output = combine.run(batch, self.sources, result)
            self.assertEqual(output.columns.tolist(), ['accession', *self.sources.columns, 'nli_verification_score'])

    def test_cli_switches_and_unchanged_llm_threshold(self):
        args = metalyzer.parse_args(['--biosample', 'SAMN1', '--sources', 'sources.tsv', '--out', 'out.tsv',
                                    '--disable-host-aliases', '--other-animal-taxonomy'])
        self.assertTrue(args.disable_host_aliases)
        self.assertTrue(args.other_animal_taxonomy)
        self.assertEqual(args.llm_min_score, .75)

    def test_verification_still_scores_both_additional_routes(self):
        batch = input.prepare_batch(pd.DataFrame({'accession': ['a', 'b'], 'host': ['Canine', 'alpaca']}), id_col='accession')
        with contextlib.redirect_stdout(io.StringIO()):
            result = ds.run(batch, self.sources, self.taxonomy, other_animal=True)
            with patch.object(nli, 'pipeline') as factory:
                factory.return_value.side_effect = lambda texts, **kw: [dict(labels=kw['candidate_labels'], scores=[.96]) for _ in texts]
                verified = verification.run(batch, self.sources, result, nli.NLIConfig(device=-1))
        self.assertEqual(verified.values.tolist(), [.96, .96])

    def test_alias_validation_precedes_model_loading_for_tsv(self):
        metadata, sources, out = [self.directory / name for name in ['input.tsv', 'sources.tsv', 'output.tsv']]
        metadata.write_text('accession\thost\na\tCanine\n')
        sources.write_text('source\thost_aliases\ndog\tcanine\ncat\tCANINE\n')
        with patch.object(nli, 'pipeline') as model, contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Conflicting host alias'):
            metalyzer.main(['--metadata', str(metadata), '--sources', str(sources), '--out', str(out),
                            '--id-col', 'accession', '--taxonomy-dir', str(self.directory)])
        model.assert_not_called()
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
