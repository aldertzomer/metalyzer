"""CPU-only accession alignment checks for benchmark metrics and figures."""
import unittest

from benchmark_smoke.benchmark_evaluation import align_predictions


class BenchmarkAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.truth = [{'run_accession': 'run2', 'final_species': 'dog'},
                      {'run_accession': 'run1', 'final_species': 'cat'}]

    def test_truth_order_does_not_control_alignment(self):
        predictions = [{'run_accession': 'run1', 'best_hit': 'cat'},
                       {'run_accession': 'run2', 'best_hit': 'dog'}]
        truth, predicted = align_predictions(self.truth, predictions)
        self.assertTrue(all(truth[key]['final_species'] == row['best_hit'] for key, row in predicted.items()))

    def test_repeated_biosample_maps_to_distinct_runs(self):
        mapping = [{'run_accession': 'run1', 'accession': 'SAMN1'},
                   {'run_accession': 'run2', 'accession': 'SAMN1'}]
        predictions = [{'accession': 'SAMN1', 'best_hit': 'cat'},
                       {'accession': 'SAMN1', 'best_hit': 'dog'}]
        _, predicted = align_predictions(self.truth, predictions, mapping)
        self.assertEqual(predicted['run1']['best_hit'], 'cat')
        self.assertEqual(predicted['run2']['best_hit'], 'dog')
        self.assertNotIn('run_accession', predictions[0])

    def test_wrong_biosample_order_fails(self):
        mapping = [{'run_accession': 'run1', 'accession': 'a'},
                   {'run_accession': 'run2', 'accession': 'b'}]
        with self.assertRaisesRegex(ValueError, 'order or count'):
            align_predictions(self.truth, [{'accession': 'b'}, {'accession': 'a'}], mapping)

    def test_duplicate_truth_or_prediction_run_fails(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            align_predictions(self.truth + [self.truth[0]], [])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            align_predictions(self.truth, [{'run_accession': 'run1'}, {'run_accession': 'run1'}])

    def test_missing_predictions_fail_unless_subset_allowed(self):
        predictions = [{'run_accession': 'run1', 'best_hit': 'cat'}]
        with self.assertRaisesRegex(ValueError, 'different run accession sets'):
            align_predictions(self.truth, predictions)
        _, predicted = align_predictions(self.truth, predictions, allow_subset=True)
        self.assertEqual(list(predicted), ['run1'])

    def test_unmapped_biosamples_fail(self):
        with self.assertRaisesRegex(ValueError, 'require an accession mapping'):
            align_predictions(self.truth, [{'accession': 'SAMN1'}])


class BenchmarkFigureTests(unittest.TestCase):
    def test_biosample_matrix_counts_duplicates_and_handles_empty_classes(self):
        import tempfile
        from pathlib import Path
        import xml.etree.ElementTree as ET
        from render_benchmark_matrices import load_matrix, render_svg
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / 'truth.tsv'
            truth.write_text('run_accession\tfinal_species\nrun2\tdog\nrun1\tcat\n')
            predictions = root / 'predictions.tsv'
            predictions.write_text('accession\tbest_hit\nSAMN1\tcat\nSAMN1\tdog\n')
            mapping = root / 'mapping.tsv'
            mapping.write_text('run_accession\taccession\nrun1\tSAMN1\nrun2\tSAMN1\n')
            totals, matrix = load_matrix(predictions, mapping, truth)
            self.assertEqual(sum(totals.values()), 2)
            self.assertEqual(matrix['cat']['cat'], 1)
            self.assertEqual(matrix['dog']['dog'], 1)
            image = root / 'figure.svg'
            render_svg(image, 'Synthetic benchmark', totals, matrix, percentages=True)
            self.assertEqual(ET.parse(image).getroot().tag, '{http://www.w3.org/2000/svg}svg')
            self.assertIn('NA', image.read_text())
