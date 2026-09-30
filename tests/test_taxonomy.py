import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import metalyzer
from helpers import classify
from modules import country, deterministic_source, generative, input as input_module, nli, records
from modules.contracts import SourceVocabulary
from modules.deterministic_source import NCBITaxonomy, configured_anchors, download_taxonomy, load_taxonomy


class TaxonomyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        # A small rank-independent lineage fixture with the relevant NCBI IDs.
        taxa = {
            1: (1, "root"), 33208: (1, "Metazoa"), 8782: (33208, "Aves"),
            33554: (33208, "Carnivora"), 9031: (8782, "Gallus gallus"),
            9103: (8782, "Meleagris gallopavo"), 9940: (33208, "Ovis aries"),
            11: (9940, "Sheep subspecies"), 9606: (33208, "Homo sapiens"),
            9685: (33554, "Felis catus"), 9615: (33554, "Canis lupus familiaris"),
            1936016: (1, "environmental samples"), 41000: (1936016, "soil metagenome"),
            41001: (1936016, "marine metagenome"), 41002: (1936016, "air metagenome"),
            81077: (1, "artificial sequences"), 32630: (81077, "synthetic construct"),
            2: (1, "Bacteria"), 562: (2, "Escherichia coli"),
            13: (14, "Broken lineage"), 15: (16, "Cycle one"), 16: (15, "Cycle two"),
        }
        (self.directory / "nodes.dmp").write_text("".join(
            f"{taxid}\t|\t{parent}\t|\tspecies\t|\n" for taxid, (parent, _) in taxa.items()))
        (self.directory / "names.dmp").write_text("".join(
            f"{taxid}\t|\t{name}\t|\t\t|\tscientific name\t|\n"
            for taxid, (_, name) in taxa.items()))
        (self.directory / "merged.dmp").write_text(
            "9000 | 9001 |\n9001 | 9940 |\n8000 | 8001 |\n8001 | 8000 |\n")
        self.taxonomy = NCBITaxonomy(self.directory)
        self.sources = SourceVocabulary(
            ("sheep (Ovis aries)", "chicken (Gallus gallus)", "human (Homo sapiens)",
             "environment (soil and sediment)", "laboratory (artificial sequences)",
             "cat (Felis catus)", "other_animal (other hosts)"),
            "run_accession", ((9940,), (9031,), (9606,), (1936016,),
                              (81077, 32630), (9685,), ()),
        )
        self.args = SimpleNamespace(id_col="run_accession", min_score=0.2, batch_size=2, device=-1)

    def match(self, taxid, sources=None):
        sources = sources or self.sources
        return self.taxonomy.classify_host_taxid(taxid, configured_anchors(sources, self.taxonomy))

    def test_exact_descendant_and_merged_ids(self):
        for value in (9940, 9940.0, "9940.0", '"9940"', " 9940 ", 9000, 11):
            with self.subTest(value=value):
                self.assertEqual(self.match(value).source, "sheep")
        self.assertEqual(self.match(9000).evidence, "host_tax_id=9940; Ovis aries")
        before = self.taxonomy._lineage.cache_info().hits
        self.taxonomy.lineage(9940)
        self.taxonomy.lineage(9940)
        self.assertGreater(self.taxonomy._lineage.cache_info().hits, before)

    def test_specific_anchor_overrides_broad_rank_independent_ancestors(self):
        sources = SourceVocabulary(
            ("other_animal (broad)", "bird (Aves)", "chicken (Gallus gallus)",
             "carnivora (order)", "cat (Felis catus)"), "run_accession",
            ((33208,), (8782,), (9031,), (33554,), (9685,)),
        )
        for taxid, expected in [(9031, "chicken"), (9103, "bird"), (9615, "carnivora"),
                                (9685, "cat"), (9940, "other_animal"), (8782, "bird")]:
            with self.subTest(taxid=taxid):
                self.assertEqual(self.match(taxid, sources).source, expected)

    def test_environment_and_artificial_descendants_not_ordinary_organisms(self):
        for taxid in (1936016, 41000, 41001, 41002):
            self.assertEqual(self.match(taxid).source, "environment")
        for taxid in (81077, 32630):
            self.assertEqual(self.match(taxid).source, "laboratory")
        self.assertIsNone(self.match(562))

    def test_duplicate_anchor_fails_before_classification(self):
        sources = SourceVocabulary(("broad", "first", "second"), "run_accession",
                                   ((33208,), (9940,), (9940,)))
        with self.assertRaisesRegex(ValueError, "conflicting taxonomy anchors: 9940"):
            configured_anchors(sources, self.taxonomy)
        # Merged IDs in the configuration normalize to the same anchor too.
        merged = SourceVocabulary(("first", "second"), "run_accession", ((9000,), (9940,)))
        with self.assertRaisesRegex(ValueError, "conflicting taxonomy anchors: 9940"):
            configured_anchors(merged, self.taxonomy)

    def test_invalid_and_merged_anchor_configuration_is_reported(self):
        invalid = SourceVocabulary(("missing", "broken"), "run_accession", ((999999,), (13,)))
        with contextlib.redirect_stdout(io.StringIO()) as log:
            with self.assertRaisesRegex(ValueError, "unresolvable taxonomy anchors"):
                configured_anchors(invalid, self.taxonomy, report=True)
        self.assertIn("invalid anchors:    2", log.getvalue())
        merged = SourceVocabulary(("sheep",), "run_accession", ((9000,),))
        with contextlib.redirect_stdout(io.StringIO()) as log:
            anchors = configured_anchors(merged, self.taxonomy, report=True)
        self.assertEqual(anchors, {9940: {"sheep"}})
        self.assertIn("taxonomy anchor 9000 -> 9940 (merged)", log.getvalue())
        self.assertIn("merged IDs:         1", log.getvalue())

    def test_cli_rejects_conflicting_anchors_before_model_loading(self):
        pd.DataFrame({"run_accession": ["a"], "host_tax_id": ["9940"]}).to_csv(
            self.directory / "input.tsv", sep="\t", index=False)
        pd.DataFrame({"source": ["first", "second"],
                      "taxonomy_anchors": ["9000", "9940"]}).to_csv(
            self.directory / "sources.tsv", sep="\t", index=False)
        with patch.object(nli, "pipeline") as model, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "conflicting taxonomy anchors"):
                metalyzer.main(["--metadata", str(self.directory / "input.tsv"),
                                "--sources", str(self.directory / "sources.tsv"),
                                "--out", str(self.directory / "out.tsv"),
                                "--id-col", "run_accession", "--taxonomy-dir", str(self.directory)])
        model.assert_not_called()
        self.assertFalse((self.directory / "out.tsv").exists())
        log = (self.directory / "out.tsv.log").read_text(encoding="utf-8")
        self.assertIn("taxonomy anchor 9000 -> 9940 (merged)", log)
        self.assertIn("conflicts:          1", log)
        self.assertIn("Traceback (most recent call last)", log)

    def test_blank_missing_anchors_and_invalid_taxids(self):
        empty = SourceVocabulary(("sheep",), "run_accession")
        self.assertEqual(empty.taxonomy_anchors, ((),))
        self.assertIsNone(self.match(9940, empty))
        for value in (None, pd.NA, "NA", "", "nonsense", 0, -1, 9940.5,
                      999999, 8000, 13, 15):
            with self.subTest(value=value):
                self.assertIsNone(self.match(value))

    def test_sources_tsv_parsing_keeps_text_models_free_of_anchor_ids(self):
        pd.DataFrame({"run_accession": ["a"], "host": ["Ovis aries"]}).to_csv(
            self.directory / "input.tsv", sep="\t", index=False)
        pd.DataFrame({"source": ["sheep (Ovis aries)", "", "laboratory (synthetic)"],
                      "taxonomy_anchors": ["9940", "", " 81077, 32630 "]}).to_csv(
            self.directory / "sources.tsv", sep="\t", index=False)
        batch, sources = input_module.load(self.directory / "input.tsv",
                                           self.directory / "sources.tsv", "run_accession")
        self.assertEqual(sources.taxonomy_anchors, ((9940,), (81077, 32630)))
        self.assertEqual(sources.labels, ("sheep (Ovis aries)", "laboratory (synthetic)"))
        self.assertEqual(sources.names, ("sheep", "laboratory"))
        prompt = generative.system_prompt(sources.labels, sources.names)
        self.assertIn("sheep (Ovis aries)", prompt)
        self.assertNotIn("9940", prompt)
        self.assertNotIn("81077", prompt)
        self.assertNotIn("taxonomy_anchors", batch.records.iloc[0])
        with patch.object(nli, "pipeline") as factory:
            factory.return_value.return_value = [{"labels": list(sources.labels), "scores": [0.8, 0.1]}]
            nli.run(batch, sources, nli.NLIConfig(device=-1))
            self.assertEqual(factory.return_value.call_args.kwargs["candidate_labels"], list(sources.labels))
        (self.directory / "sources.tsv").write_text("source\nsheep (Ovis aries)\n")
        _, no_anchors = input_module.load(self.directory / "input.tsv",
                                          self.directory / "sources.tsv", "run_accession")
        self.assertIsNone(self.match(9940, no_anchors))
        for invalid in ("9940,", "abc", "0", "9940;9031"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "taxonomy_anchors"):
                    input_module.parse_taxonomy_anchors(invalid)

    def test_repository_vocabulary_has_requested_anchors(self):
        root = Path(__file__).resolve().parents[1]
        _, sources = input_module.load(root / "benchmark.tsv", root / "sources.tsv", "run_accession", limit=1)
        configured = dict(zip(sources.names, sources.taxonomy_anchors))
        expected = {"chicken": (9031,), "turkey": (9103,), "pig": (9825,),
                    "cattle": (9913,), "sheep": (9940,), "goat": (9925,),
                    "human": (9606,), "dog": (9615,), "cat": (9685,),
                    "environment": (1936016,), "laboratory": (81077, 32630)}
        for source, anchors in expected.items():
            self.assertEqual(configured[source], anchors)
        self.assertEqual(configured["other_animal"], ())

    def test_build_record_uses_natural_field_and_value_boundaries(self):
        row = pd.Series({"host_scientific_name": "Ovis aries", "isolation_source": "stool",
                         "geo_loc_name": "USA: WY", "missing_value": "NA"})
        record = records.build_record(row)
        self.assertEqual(record,
                         "host scientific name: Ovis aries; isolation source: stool; geo loc name: USA: WY")
        self.assertNotIn('="', record)
        self.assertEqual(country.extract_country_from_row(row, record), "United States")
        self.assertEqual(records.build_record(pd.Series(dtype=object)), "metadata: (empty)")

    def fake_pipeline(self, *args, **kwargs):
        self.assertEqual(kwargs["dtype"], "float32" if self.args.device < 0 else "auto")

        def classifier(batch, **options):
            self.seen.extend(batch)
            self.batches.append(len(batch))
            self.assertEqual(options["batch_size"], 2)
            return [{"labels": list(self.sources.labels),
                     "scores": [0.1] * len(self.sources.labels)} for _ in batch]
        return classifier

    def test_mixed_integration_fallback_order_and_generic_taxid_ignored(self):
        df = pd.DataFrame({"run_accession": list("abcdef"),
                           "host_tax_id": [9940, "NA", 9031, 562, 9606, "NA"],
                           "tax_id": [562, 9940, 562, 81077, 562, 1936016],
                           "host_scientific_name": ["Ovis aries", "missing", "chicken", "bad", "human", "unknown"],
                           "collection_date": ["2019"] * 6, "country": ["USA"] * 6})
        df.to_csv(self.directory / "input.tsv", sep="\t", index=False)
        pd.DataFrame({"source": self.sources.labels,
                      "taxonomy_anchors": [",".join(map(str, anchors)) for anchors in self.sources.taxonomy_anchors]}).to_csv(
            self.directory / "sources.tsv", sep="\t", index=False)
        self.seen, self.batches = [], []
        with patch.object(nli, "pipeline", side_effect=self.fake_pipeline), contextlib.redirect_stdout(io.StringIO()) as log:
            metalyzer.main(["--metadata", str(self.directory / "input.tsv"),
                            "--sources", str(self.directory / "sources.tsv"),
                            "--out", str(self.directory / "out.tsv"), "--id-col", "run_accession",
                            "--device", "-1", "--batch-size", "2", "--taxonomy-dir", str(self.directory),
                            "--disable-verify-source"])
        out = pd.read_csv(self.directory / "out.tsv", sep="\t", keep_default_na=False)
        self.assertEqual(out.run_accession.tolist(), list("abcdef"))
        self.assertEqual(out.best_hit.tolist(), ["sheep", "unknown", "chicken", "unknown", "human", "unknown"])
        self.assertEqual(out.source_method.tolist(),
                         ["host_tax_id", "nli", "host_tax_id", "nli", "host_tax_id", "nli"])
        self.assertTrue((out.loc[[0, 2, 4], list(self.sources.names)] == "NA").all().all())
        self.assertEqual(out.loc[0, "source_evidence"], "host_tax_id=9940; Ovis aries")
        self.assertTrue((out.year == 2019).all())
        self.assertTrue((out.country == "United States").all())
        self.assertEqual(self.batches, [2, 1])
        self.assertIn("taxonomy host_tax_id: 3", log.getvalue())

    def test_all_taxonomy_never_loads_model_and_no_match_falls_back(self):
        df = pd.DataFrame({"host_tax_id": [9940, 9031, 9606]})
        with patch.object(nli, "pipeline") as model:
            out = classify(df, ["host"] * 3, self.sources.labels, self.args,
                           self.taxonomy, self.sources.taxonomy_anchors)
        model.assert_not_called()
        self.assertTrue(out.iloc[:, :len(self.sources.labels)].isna().all().all())
        with patch.object(nli, "pipeline") as model:
            model.return_value.return_value = [{"labels": list(self.sources.labels),
                                                "scores": [0.7] * len(self.sources.labels)}]
            out = classify(pd.DataFrame({"tax_id": [9940]}), ["Ovis aries"],
                           self.sources.labels, self.args, self.taxonomy, self.sources.taxonomy_anchors)
            self.assertEqual(out.source_method.tolist(), ["nli"])
            model.return_value.assert_called_once()

    def test_nli_score_logic_unchanged(self):
        df = pd.read_csv(Path(__file__).resolve().parents[1] / "benchmark.tsv",
                         sep="\t", dtype=str, keep_default_na=False)
        rendered = [records.build_record(row) for _, row in df.iterrows()]
        self.seen, self.batches = [], []
        self.args.device = 0
        with patch.object(nli, "pipeline", side_effect=self.fake_pipeline), contextlib.redirect_stdout(io.StringIO()):
            out = classify(df, rendered, self.sources.labels, self.args)
        expected = nli.parse_source_scores(pd.DataFrame(
            [[0.1] * len(self.sources.labels)] * len(df), columns=self.sources.labels),
            self.args.id_col, 0.2)
        pd.testing.assert_frame_equal(out[expected.columns], expected)
        self.assertEqual(self.seen, rendered)

    def test_load_failures_and_empty_input(self):
        with self.assertWarns(UserWarning):
            self.assertIsNone(load_taxonomy(None))
        with self.assertWarns(UserWarning):
            self.assertIsNone(load_taxonomy(self.directory / "absent"))
        with patch.object(nli, "pipeline") as model:
            result = classify(pd.DataFrame(), [], self.sources.labels, self.args, self.taxonomy,
                              self.sources.taxonomy_anchors)
        self.assertEqual(len(result), 0)
        model.assert_not_called()
        (self.directory / "names.dmp").write_text("malformed\n")
        with self.assertWarns(UserWarning):
            self.assertIsNone(load_taxonomy(self.directory))

    def test_explicit_download_and_safe_members(self):
        import tarfile
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as handle:
            for name in ("nodes.dmp", "names.dmp", "merged.dmp"):
                handle.add(self.directory / name, arcname=name)
            info = tarfile.TarInfo("../unwanted.txt")
            info.size = 3
            handle.addfile(info, io.BytesIO(b"bad"))
        archive.seek(0)
        with patch("modules.deterministic_source.urllib.request.urlopen", return_value=archive):
            download_taxonomy(self.directory / "download")
        self.assertFalse((self.directory / "unwanted.txt").exists())
        taxonomy = NCBITaxonomy(self.directory / "download")
        self.assertEqual(taxonomy.classify_host_taxid(
            9940, configured_anchors(self.sources, taxonomy)).source, "sheep")
        with patch.object(deterministic_source, "download_taxonomy") as download, patch.object(nli, "pipeline") as model:
            metalyzer.main(["--download-taxonomy", "taxonomy"])
            download.assert_called_once_with("taxonomy")
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
