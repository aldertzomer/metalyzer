"""Exact documented exclusions, preserved original scores, and both taxonomy routes."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from benchmark_smoke.metadata_version_evaluation import validated_conflict_runs
from benchmark_smoke.evaluate_local_backends import evaluate


class MetadataVersionTests(unittest.TestCase):
    def setUp(self):
        self.truth, self.metadata, self.mapping, self.current, self.documented, self.predictions = [], [], [], [], [], []
        for i in range(1320):
            run, accession = f'run{i}', f'SAMN{i}'
            label = 'goat' if i < 45 else ('unknown' if i == 45 else 'cat')
            self.truth.append(dict(run_accession=run, final_species=label))
            self.metadata.append(dict(run_accession=run, sample_accession=accession, host_scientific_name='Capra hircus' if i < 45 else 'Felis catus'))
            self.mapping.append(dict(run_accession=run, accession=accession))
            self.current.append(dict(accession=accession, host='Gallus gallus' if i < 45 else 'Felis catus',
                                     ifsac_category='chicken' if i < 45 else '', last_update='2025-06-18T08:40:35.278'))
            if i < 44:
                self.documented.append(dict(run_accession=run, accession=accession, final_species='goat',
                                            tsv_metadata_host_scientific_name='Capra hircus',
                                            bio_metadata_host='Gallus gallus', bio_metadata_ifsac_category='chicken'))
            self.predictions.append(dict(accession=accession, best_hit='chicken' if i < 45 else ('human' if i == 45 else 'cat'),
                                         source_method='host_tax_id' if i == 46 else 'host_name', source_evidence='evidence'))

    def validate(self):
        return validated_conflict_runs(self.truth, self.metadata, self.mapping, self.documented, biosample_rows=self.current)

    def test_exact_44_exclusions_and_untouched_original_evaluation(self):
        excluded = self.validate()
        self.assertEqual(len(excluded), 44)
        self.assertNotIn('run44', excluded)  # identical goat/chicken discrepancy, not documented
        self.assertNotIn('run45', excluded)  # reference-unknown / explicit-human is outside scope
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'prediction.tsv'
            pd.DataFrame(self.predictions).to_csv(path, sep='\t', index=False)
            before = path.read_bytes()
            truth = pd.DataFrame(self.truth)
            before_truth = truth.copy(deep=True)
            original_dir, adjusted_dir = root / 'original', root / 'adjusted'
            original_dir.mkdir(); adjusted_dir.mkdir()
            original = evaluate(truth, path, 'best_hit', original_dir, False, self.mapping)
            adjusted = evaluate(truth, path, 'best_hit', adjusted_dir, False, self.mapping, excluded)
            self.assertEqual((original['records'], adjusted['records']), (1320, 1276))
            self.assertEqual((original['correct'], adjusted['correct']), (1274, 1274))
            self.assertEqual(adjusted['excluded_records'], 44)
            self.assertEqual(adjusted['taxonomy_calls'], 1276)
            self.assertEqual(adjusted['numeric_taxonomy_calls'], 1)
            self.assertEqual(adjusted['host_name_calls'], 1275)
            self.assertEqual(path.read_bytes(), before)
            pd.testing.assert_frame_equal(truth, before_truth)
            again = evaluate(truth, path, 'best_hit', original_dir, False, self.mapping)
            self.assertEqual(original, again)

    def test_bad_pair_source_evidence_count_and_live_version_are_rejected(self):
        for alteration in ['pair', 'old_host', 'new_host', 'category', 'count', 'live_host', 'live_update']:
            with self.subTest(alteration=alteration):
                self.setUp()
                if alteration == 'pair': self.documented[0]['accession'] = 'unrelated'
                elif alteration == 'old_host': self.metadata[0]['host_scientific_name'] = 'Bos taurus'
                elif alteration == 'new_host': self.documented[0]['bio_metadata_host'] = 'Capra hircus'
                elif alteration == 'category': self.documented[0]['bio_metadata_ifsac_category'] = 'goat'
                elif alteration == 'count': self.documented.pop()
                elif alteration == 'live_host': self.current[0]['host'] = 'Capra hircus'
                elif alteration == 'live_update': self.current[0]['last_update'] = '2026-10-12'
                with self.assertRaises(ValueError): self.validate()
