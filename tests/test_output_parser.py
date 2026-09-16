import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from metalyzer import main, parse_source_scores


class OutputParserTests(unittest.TestCase):
    def test_names_scores_ties_and_index(self):
        scores = pd.DataFrame(
            {"wild bird (avian host)": [0.2, 0.5], "cat": [0.8, 0.5]},
            index=[5, 9],
        )
        result = parse_source_scores(scores, "sample_id")
        self.assertEqual(list(result.columns), ["wild bird", "cat", "best_hit"])
        self.assertEqual(result["best_hit"].tolist(), ["cat", "wild bird"])
        pd.testing.assert_frame_equal(result.iloc[:, :2], scores.set_axis(["wild bird", "cat"], axis=1))
        self.assertEqual(scores.columns[0], "wild bird (avian host)")

    def test_invalid_names(self):
        for columns in [[], ["(hint)"], ["cat (a)", "cat (b)"],
                        ["sample_id"], ["best_hit"], ["year"], ["country"]]:
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                parse_source_scores(pd.DataFrame(columns=columns), "sample_id")

    def test_empty_rows(self):
        result = parse_source_scores(pd.DataFrame(columns=["cat (host)"]), "sample_id")
        self.assertEqual(list(result.columns), ["cat", "best_hit"])
        self.assertTrue(result.empty)

    def test_cli_output_keeps_full_inference_labels(self):
        labels = ["wild bird (avian host)", "cat (feline host)"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata, sources, output = [root / name for name in ("input.tsv", "sources.tsv", "out.tsv")]
            pd.DataFrame({"sample_id": ["S1", "S2"], "country": ["USA", "Netherlands"],
                          "collection_date": ["2019", "2020"]}).to_csv(metadata, sep="\t", index=False)
            pd.DataFrame({"source": labels}).to_csv(sources, sep="\t", index=False)
            argv = ["metalyzer.py", "--metadata", str(metadata), "--sources", str(sources),
                    "--out", str(output), "--id-col", "sample_id", "--device", "-1", "--batch-size", "2"]
            with patch.object(sys, "argv", argv), patch("metalyzer.pipeline") as factory:
                factory.return_value.return_value = [
                    {"labels": labels[::-1], "scores": [0.8, 0.2]},
                    {"labels": labels, "scores": [0.9, 0.1]},
                ]
                main()
                self.assertEqual(factory.return_value.call_args.kwargs["candidate_labels"], labels)
            result = pd.read_csv(output, sep="\t", dtype={"year": str})
            self.assertEqual(list(result.columns),
                             ["sample_id", "wild bird", "cat", "best_hit", "year", "country"])
            self.assertEqual(result["sample_id"].tolist(), ["S1", "S2"])
            self.assertEqual(result["best_hit"].tolist(), ["cat", "wild bird"])
            self.assertEqual(result["wild bird"].tolist(), [0.2, 0.9])
            self.assertEqual(result["cat"].tolist(), [0.8, 0.1])
            self.assertEqual(result["year"].tolist(), ["2019", "2020"])
            self.assertEqual(result["country"].tolist(), ["United States", "Netherlands"])


if __name__ == "__main__":
    unittest.main()
