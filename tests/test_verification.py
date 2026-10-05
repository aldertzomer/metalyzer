"""Final-source verification without downloading an NLI checkpoint."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import metalyzer
from modules import combine, deterministic_source, input, nli, verification
from modules.contracts import SourceResult, SourceVocabulary, empty_source_table


class SourceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.batch = input.prepare_batch(pd.DataFrame({
            "id": ["a", "b", "c", "d", "e"],
            "host": ["Ovis aries", "turkey meat", "no host", "turkey", "unknown"],
        }))
        self.sources = SourceVocabulary((
            "sheep (host Ovis aries)", "turkey (poultry meat)",
            "unknown (insufficient evidence)",
        ), "id")
        table = empty_source_table(self.sources, self.batch.metadata.index)
        table["best_hit"] = ["sheep", "turkey", "unknown", "turkey", "unknown"]
        table["source_method"] = ["host_tax_id", "nli", "llm", "mistral", "llm"]
        self.result = SourceResult(table)

    def test_selected_full_label_and_unknown_rows(self):
        seen = []

        def classify(records, **kwargs):
            seen.append((records, kwargs))
            self.assertEqual(kwargs["multi_label"], True)
            self.assertEqual(len(kwargs["candidate_labels"]), 1)
            return [{"labels": kwargs["candidate_labels"], "scores": [0.76]}
                    for _ in records]

        with patch.object(nli, "pipeline", return_value=classify) as factory:
            scores = verification.run(self.batch, self.sources, self.result,
                                      nli.NLIConfig(device=-1, batch_size=2)).values
        factory.assert_called_once_with("zero-shot-classification", model=nli.DEFAULT_NLI_MODEL,
                                        device=-1, dtype="float32")
        self.assertEqual(scores.iloc[[0, 1, 3]].tolist(), [0.76] * 3)
        self.assertTrue(scores.iloc[[2, 4]].isna().all())
        self.assertEqual([call[1]["candidate_labels"] for call in seen],
                         [["sheep (host Ovis aries)"], ["turkey (poultry meat)"]])
        self.assertEqual(seen[0][0], [self.batch.records.iloc[0]])
        self.assertEqual(seen[1][0], self.batch.records.iloc[[1, 3]].tolist())
        self.assertTrue(all(call[1]["hypothesis_template"] == nli.HYPOTHESIS_TEMPLATE
                            for call in seen))

    def test_disabled_keeps_na_and_does_not_load_model(self):
        with patch.object(nli, "pipeline") as factory:
            verified = verification.run(self.batch, self.sources, self.result, disabled=True)
        factory.assert_not_called()
        self.assertTrue(verified.values.isna().all())
        output = combine.run(self.batch, self.sources, self.result, verification=verified)
        self.assertEqual(output.best_hit.tolist(), self.result.table.best_hit.tolist())
        self.assertEqual(output.source_method.tolist(), self.result.table.source_method.tolist())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.tsv"
            combine.write(output, path)
            saved = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(saved.nli_verification_score.tolist(), ["NA"] * 5)

    def test_all_unknown_never_loads_verifier(self):
        table = self.result.table.copy()
        table["best_hit"] = "unknown"
        with patch.object(nli, "pipeline") as factory:
            verified = verification.run(self.batch, self.sources, SourceResult(table))
        factory.assert_not_called()
        self.assertTrue(verified.values.isna().all())

    def test_nli_classification_reuses_same_model_and_default_definition(self):
        batch = self.batch.subset([0])
        sources = SourceVocabulary(("sheep (host Ovis aries)", "turkey (poultry meat)"), "id")
        classify = Mock(side_effect=[
            {"labels": list(sources.labels), "scores": [0.9, 0.1]},
            {"labels": [sources.labels[0]], "scores": [0.8]},
        ])
        config = nli.NLIConfig(device=-1, batch_size=1)
        model = nli.NLIModel(config)
        with patch.object(nli, "DEFAULT_NLI_MODEL", "test/shared-model"), \
             patch.object(nli, "pipeline", return_value=classify) as factory:
            result = nli.run(batch, sources, config, model=model)
            verified = verification.run(batch, sources, result, config, model=model)
        factory.assert_called_once_with("zero-shot-classification", model="test/shared-model",
                                        device=-1, dtype="float32")
        self.assertEqual(classify.call_count, 2)
        self.assertEqual(classify.call_args_list[0].kwargs["multi_label"], False)
        self.assertEqual(classify.call_args_list[1].kwargs["multi_label"], True)
        self.assertEqual(verified.values.tolist(), [0.8])

    def test_cli_disable_flag_and_reserved_name(self):
        args = metalyzer.parse_args(["--metadata", "in.tsv", "--sources", "src.tsv",
                                     "--out", "out.tsv", "--id-col", "id",
                                     "--disable-verify-source"])
        self.assertTrue(args.disable_verify_source)
        with self.assertRaises(ValueError):
            SourceVocabulary(("nli_verification_score",), "id")

    def test_cli_verifies_after_final_nli_decision_with_one_model(self):
        def classify(records, **kwargs):
            if kwargs["multi_label"]:
                self.assertEqual(kwargs["candidate_labels"], ["sheep (host Ovis aries)"])
                return {"labels": kwargs["candidate_labels"], "scores": [0.81]}
            return {"labels": ["sheep (host Ovis aries)", "turkey (poultry meat)"],
                    "scores": [0.9, 0.1]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["a"], "host": ["Ovis aries"]}).to_csv(
                root / "input.tsv", sep="\t", index=False)
            pd.DataFrame({"source": ["sheep (host Ovis aries)", "turkey (poultry meat)"]}).to_csv(
                root / "sources.tsv", sep="\t", index=False)
            with patch.object(deterministic_source, "load_taxonomy", return_value=None), \
                 patch.object(nli, "pipeline", return_value=classify) as factory:
                metalyzer.main(["--metadata", str(root / "input.tsv"),
                                "--sources", str(root / "sources.tsv"),
                                "--out", str(root / "out.tsv"), "--id-col", "id",
                                "--device", "-1"])
            output = pd.read_csv(root / "out.tsv", sep="\t", keep_default_na=False)
        factory.assert_called_once()
        self.assertEqual(output.best_hit.tolist(), ["sheep"])
        self.assertEqual(output.nli_verification_score.tolist(), [0.81])


if __name__ == "__main__":
    unittest.main()
