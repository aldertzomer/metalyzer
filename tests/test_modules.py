"""Independent stage contracts, row alignment, and CLI activation regressions."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import metalyzer
from modules import combine, country, date, deterministic_source, input, llm, nli
from modules.contracts import (
    FieldResult, MetadataBatch, SourceResult, SourceVocabulary, empty_source_table,
)


class ModuleTests(unittest.TestCase):
    def setUp(self):
        self.batch = input.prepare_batch(pd.DataFrame({
            "id": ["duplicate", "duplicate", "third"],
            "host": ["Ovis aries", "", "cat"],
            "collection_date": ["31-12-19", "missing", "1904"],
            "collection_date_end": ["2020", "2018-06", ""],
            "country": ["USA:WY", "Canada: Calgary, Alberta", ""],
        }, index=[100, 200, 300]))
        self.sources = SourceVocabulary(("sheep (Ovis aries)", "cat"), "id")

    def result(self, keys, method="example"):
        table = empty_source_table(self.sources, keys)
        table["best_hit"] = "sheep"
        table["source_method"] = method
        return SourceResult(table)

    def test_field_stages_in_isolation_and_missing_values(self):
        batch = self.batch.subset([2, 0, 1])
        years, countries = date.run(batch), country.run(batch)
        years.validate(batch)
        countries.validate(batch)
        self.assertEqual(years.values.tolist(), ["", "2019", "2018"])
        self.assertEqual(countries.values.tolist(), ["unknown", "United States", "Canada"])
        self.assertEqual(years.values.index.tolist(), [2, 0, 1])
        pd.testing.assert_frame_equal(self.batch.metadata, input.prepare_batch(self.batch.metadata).metadata)

    def test_date_boundaries_and_fallback(self):
        for raw, expected in [("1905", 1905), ("2026", 2026), ("2027", None),
                              ("15-06-18", 2018), ("15-06-99", 1999), ("NA", None)]:
            with self.subTest(raw=raw):
                self.assertEqual(date.year_from_value(raw), expected)
        batch = input.prepare_batch(pd.DataFrame({"sampling_year": ["collected in 2020"]}))
        self.assertEqual(date.run(batch).values.tolist(), ["2020"])
        self.assertEqual(date.run(batch, date.DateConfig(max_year=2019)).values.tolist(), [""])

    def test_country_does_not_fuzzy_match_unrelated_text(self):
        batch = input.prepare_batch(pd.DataFrame({"host": ["turkey"], "title": ["USA"]}))
        self.assertEqual(country.run(batch).values.tolist(), ["unknown"])
        for raw, expected in [("U.S.A;USA", "United States"), ("United Kingdom: Oxford", "United Kingdom")]:
            self.assertEqual(country.normalize_country(raw), expected)

    def test_combination_aligns_by_key_and_preserves_duplicate_external_ids(self):
        resolved = self.result([1])
        remaining = combine.unresolved(self.batch, self.sources, resolved)
        self.assertEqual(remaining.metadata.index.tolist(), [0, 2])
        fallback = self.result([2, 0], "fallback")
        source = combine.sources(self.batch, self.sources, resolved, fallback)
        years = date.run(self.batch.subset([2, 1, 0]))
        output = combine.run(self.batch, self.sources, source, years, country.run(self.batch))
        self.assertEqual(output.id.tolist(), ["duplicate", "duplicate", "third"])
        self.assertEqual(output.source_method.tolist(), ["fallback", "example", "fallback"])
        self.assertEqual(output.year.tolist(), ["2019", "2018", ""])
        self.assertEqual(output.columns.tolist(), ["id", *self.sources.columns,
                                                   "source_verification_score", "year", "country"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.tsv"
            combine.write(output, path)
            saved = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(saved.sheep.tolist(), ["NA"] * 3)
        self.assertEqual(saved.year.tolist(), ["2019", "2018", ""])

    def test_explicit_unknown_is_completed_not_unresolved(self):
        result = self.result([0])
        result.table.loc[0, "best_hit"] = "unknown"
        self.assertEqual(combine.unresolved(self.batch, self.sources, result).metadata.index.tolist(), [1, 2])

    def test_combiner_rejects_source_gaps_overlaps_and_foreign_keys(self):
        cases = [(self.result([0]),), (self.result([0, 1]), self.result([1, 2])),
                 (self.result([0, 1, 3]),), (self.result([0, 0, 2]),)]
        for results in cases:
            with self.subTest(keys=[r.table.index.tolist() for r in results]):
                with self.assertRaises(ValueError):
                    combine.sources(self.batch, self.sources, *results)

    def test_source_contract_rejects_bad_schema_values_and_scores(self):
        for column, value in [("best_hit", "unlisted"), ("source_method", ""),
                              ("source_evidence", None), ("sheep", 1.1),
                              ("sheep", "0.5"), ("sheep", 0.5)]:
            with self.subTest(column=column, value=value):
                table = self.result([0]).table
                if isinstance(value, str) and column == "sheep":
                    table[column] = table[column].astype(object)
                table.loc[0, column] = value
                with self.assertRaises(ValueError):
                    SourceResult(table).validate(self.batch, self.sources)
        with self.assertRaises(ValueError):
            SourceResult(self.result([0]).table.drop(columns="cat")).validate(self.batch, self.sources)

    def test_field_contract_rejects_gaps_duplicates_and_column_collisions(self):
        source = self.result([0, 1, 2])
        fields = [pd.Series(["x"], index=[0], name="year"),
                  pd.Series(["x"] * 3, index=[0, 0, 2], name="year"),
                  pd.Series(["x"] * 3, index=[0, 1, 3], name="year"),
                  pd.Series(["x"] * 3, name="id"),
                  pd.Series([None] * 3, name="year")]
        for values in fields:
            with self.subTest(values=values.tolist()):
                with self.assertRaises(ValueError):
                    combine.run(self.batch, self.sources, source, FieldResult(values))
        with self.assertRaises(ValueError):
            combine.run(self.batch, self.sources, source, date.run(self.batch), date.run(self.batch))

    def test_batch_rejects_misaligned_records(self):
        with self.assertRaises(ValueError):
            MetadataBatch(self.batch.metadata, self.batch.records.iloc[::-1])
        with self.assertRaises(ValueError):
            self.batch.subset([0, 0])

    def test_nli_preserves_sparse_keys_and_rejects_wrong_response_count(self):
        batch = self.batch.subset([2, 0])
        with patch.object(nli, "pipeline") as factory:
            factory.return_value.return_value = [
                {"labels": list(self.sources.labels), "scores": [0.2, 0.2]},
                {"labels": list(self.sources.labels), "scores": [0.1, 0.1]},
            ]
            result = nli.run(batch, self.sources, nli.NLIConfig(device=-1))
            self.assertEqual(result.table.index.tolist(), [2, 0])
            self.assertEqual(result.table.best_hit.tolist(), ["sheep", "unknown"])
            factory.return_value.return_value = []
            with self.assertRaisesRegex(RuntimeError, "number of responses"):
                nli.run(batch, self.sources)

    def test_every_stage_accepts_empty_batches_without_models(self):
        batch = self.batch.subset([])
        with patch.object(nli, "pipeline") as nli_loader, patch.object(llm, "load_local_llm") as llm_loader:
            for result in [nli.run(batch, self.sources), llm.run(batch, self.sources),
                           deterministic_source.run(batch, self.sources)]:
                result.validate(batch, self.sources)
                self.assertEqual(len(result.table), 0)
            nli_loader.assert_not_called()
            llm_loader.assert_not_called()
        date.run(batch).validate(batch)
        country.run(batch).validate(batch)

    def test_imports_do_not_load_model_libraries(self):
        code = ("import metalyzer; from modules import date, country, deterministic_source, nli, llm, mistral, combine; "
                "import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
                "assert 'mistralai' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True,
                       cwd=Path(__file__).resolve().parents[1], timeout=30)

    def test_cli_can_disable_field_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.batch.metadata.to_csv(root / "input.tsv", sep="\t", index=False)
            pd.DataFrame({"source": self.sources.labels}).to_csv(root / "sources.tsv", sep="\t", index=False)
            for flags, expected in [([], ["year", "country"]), (["--skip-date"], ["country"]),
                                    (["--skip-country"], ["year"]),
                                    (["--skip-date", "--skip-country"], [])]:
                with self.subTest(flags=flags), patch.object(deterministic_source, "load_taxonomy", return_value=None), \
                     patch.object(nli, "run", return_value=self.result([0], "nli")), \
                     patch.object(date, "run", wraps=date.run) as date_stage, \
                     patch.object(country, "run", wraps=country.run) as country_stage, \
                     contextlib.redirect_stdout(io.StringIO()):
                    metalyzer.main(["--metadata", str(root / "input.tsv"), "--sources", str(root / "sources.tsv"),
                                    "--out", str(root / "out.tsv"), "--id-col", "id", "--limit", "1",
                                    "--disable-verify-source", *flags])
                    self.assertEqual(date_stage.call_count, int("year" in expected))
                    self.assertEqual(country_stage.call_count, int("country" in expected))
                    output = pd.read_csv(root / "out.tsv", sep="\t")
                    self.assertEqual(output.columns.tolist(), ["id", *self.sources.columns,
                                                               "source_verification_score", *expected])
                    self.assertTrue(output.source_verification_score.isna().all())

    def test_input_vocabulary_and_id_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.batch.metadata.to_csv(root / "input.tsv", sep="\t", index=False)
            pd.DataFrame({"labels": [" sheep (Ovis aries) ", "", "cat"]}).to_csv(root / "sources.tsv", sep="\t", index=False)
            batch, sources = input.load(root / "input.tsv", root / "sources.tsv", "id", limit=1)
            self.assertEqual(len(batch), 1)
            self.assertEqual(sources, self.sources)
            with self.assertRaisesRegex(ValueError, "ID column"):
                input.load(root / "input.tsv", root / "sources.tsv", "absent")
        with self.assertRaises(ValueError):
            SourceVocabulary(("cat",), "year")

    def test_input_ignores_invalid_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.tsv").write_bytes(b"id\thost\nrow-1\twestern\x96european\n")
            (root / "sources.tsv").write_bytes(b"source\nother\x96animal\n")
            batch, sources = input.load(root / "input.tsv", root / "sources.tsv", "id")
        self.assertEqual(batch.metadata.host.tolist(), ["westerneuropean"])
        self.assertEqual(sources.labels, ("otheranimal",))


if __name__ == "__main__":
    unittest.main()
