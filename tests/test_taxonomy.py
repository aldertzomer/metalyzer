import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import metalyzer
from taxonomy import ANCHORS, NCBITaxonomy, load_taxonomy, download_taxonomy


class TaxonomyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        taxa = {
            1: (1, "root"), 2: (1, "Metazoa"), 3: (2, "Aves"),
            4: (2, "Mammalia"), 5: (4, "Sus scrofa"),
            9940: (4, "Ovis aries"), 9031: (3, "Gallus gallus"),
            9606: (4, "Homo sapiens"), 9796: (4, "Equus caballus"),
            10: (3, "Anas platyrhynchos"), 11: (9940, "Sheep subspecies"),
            12: (1, "Bacteria"), 13: (14, "Broken lineage"),
            15: (16, "Cycle one"), 16: (15, "Cycle two"),
            17: (5, "Sus scrofa scrofa"),
        }
        for i, name in enumerate(ANCHORS.values(), 100):
            if name not in {name for _, name in taxa.values()}:
                taxa[i] = (5 if name == "Sus scrofa domesticus" else 3 if name == "Meleagris gallopavo" else 4, name)
        (self.directory / "nodes.dmp").write_text("".join(f"{i}\t|\t{p}\t|\tspecies\t|\n" for i, (p, _) in taxa.items()))
        (self.directory / "names.dmp").write_text("".join(f"{i}\t|\t{name}\t|\t\t|\tscientific name\t|\n" for i, (_, name) in taxa.items()))
        (self.directory / "merged.dmp").write_text("9000 | 9001 |\n9001 | 9940 |\n8000 | 8001 |\n8001 | 8000 |\n")
        self.taxonomy = NCBITaxonomy(self.directory)
        self.labels = [f"{source} (hint)" for source in ANCHORS] + ["other_animal", "wildbird", "waterbird"]
        self.args = SimpleNamespace(id_col="run_accession", min_score=0.2, batch_size=2, device=-1)

    def test_parser_and_specific_hosts(self):
        for value in (9940, 9940.0, "9940.0", '"9940"', " 9940 ", 9000, 11):
            with self.subTest(value=value):
                self.assertEqual(self.taxonomy.classify_host_taxid(value).source, "sheep")
        for source, name in ANCHORS.items():
            result = self.taxonomy.classify_host_taxid(self.taxonomy.taxid_by_scientific_name[name])
            self.assertEqual(result.source, source)
        self.assertEqual(self.taxonomy.classify_host_taxid(9796).source, "other_animal")
        self.assertEqual(self.taxonomy.classify_host_taxid(9000).evidence, "host_tax_id=9940; Ovis aries")
        before = self.taxonomy._lineage.cache_info().hits
        self.taxonomy.lineage(9940)
        self.taxonomy.lineage(9940)
        self.assertGreater(self.taxonomy._lineage.cache_info().hits, before)

    def test_invalid_ambiguous_and_birds(self):
        for value in (None, pd.NA, "NA", "missing", "", "nonsense", 0, -1, 9940.5, 999999, 8000, 10, 3, 2, 4, 5, 17, 12, 13, 15):
            with self.subTest(value=value):
                self.assertIsNone(self.taxonomy.classify_host_taxid(value))

    def fake_pipeline(self, *args, **kwargs):
        self.assertEqual(kwargs["dtype"], "float32" if self.args.device < 0 else "auto")
        def classifier(batch, **options):
            self.seen.extend(batch)
            self.batches.append(len(batch))
            self.assertEqual(options["batch_size"], 2)
            return [{"labels": self.labels, "scores": [0.1] * len(self.labels)} for _ in batch]
        return classifier

    def test_mixed_integration_and_serialization(self):
        df = pd.DataFrame({"run_accession": list("abcdefg"), "host_tax_id": [9940, "NA", 9031, "nonsense", 9606, 9796, 10],
                           "host_scientific_name": ["Ovis aries", "missing-sheep", "chicken", "bad", "human", "horse", "duck"],
                           "collection_date": ["2019"] * 7, "country": ["USA"] * 7})
        df.to_csv(self.directory / "input.tsv", sep="\t", index=False)
        pd.DataFrame({"source": self.labels}).to_csv(self.directory / "sources.tsv", sep="\t", index=False)
        self.seen, self.batches = [], []
        with patch.object(metalyzer, "pipeline", side_effect=self.fake_pipeline), contextlib.redirect_stdout(io.StringIO()) as log:
            metalyzer.main(["--metadata", str(self.directory / "input.tsv"), "--sources", str(self.directory / "sources.tsv"),
                            "--out", str(self.directory / "out.tsv"), "--id-col", "run_accession", "--device", "-1",
                            "--batch-size", "2", "--taxonomy-dir", str(self.directory)])
        out = pd.read_csv(self.directory / "out.tsv", sep="\t", keep_default_na=False)
        self.assertEqual(out.run_accession.tolist(), list("abcdefg"))
        self.assertEqual(out.best_hit.tolist(), ["sheep", "unknown", "chicken", "unknown", "human", "other_animal", "unknown"])
        score_names = [label.split("(")[0].strip() for label in self.labels]
        self.assertTrue((out.loc[[0, 2, 4, 5], score_names] == "NA").all().all())
        self.assertEqual(out.source_method.tolist(), ["host_tax_id", "nli", "host_tax_id", "nli", "host_tax_id", "host_tax_id", "nli"])
        self.assertEqual(out.loc[0, "source_evidence"], "host_tax_id=9940; Ovis aries")
        self.assertTrue((out.year == 2019).all())
        self.assertTrue((out.country == "United States").all())
        self.assertFalse(any("Ovis aries" in record for record in self.seen))
        self.assertEqual(self.batches, [2, 1])
        self.assertIn("taxonomy host_tax_id: 4", log.getvalue())
        self.assertIn("NLI -> unknown:        3", log.getvalue())

    def test_all_taxonomy_never_loads_model(self):
        df = pd.DataFrame({"host_tax_id": [9940, 9031, 9606]})
        with patch.object(metalyzer, "pipeline") as model:
            out = metalyzer.classify_sources(df, ["Ovis aries"] * 3, self.labels, self.args, self.taxonomy)
        model.assert_not_called()
        self.assertTrue(out.iloc[:, :len(self.labels)].isna().all().all())

    def test_missing_host_column_and_unavailable_source_fall_back(self):
        for df, labels in [(pd.DataFrame({"tax_id": [9940]}), self.labels),
                           (pd.DataFrame({"host_tax_id": [9940]}), ["human", "other_animal"])]:
            with patch.object(metalyzer, "pipeline") as model:
                model.return_value.return_value = [{"labels": labels, "scores": [0.7] * len(labels)}]
                out = metalyzer.classify_sources(df, ["Ovis aries"], labels, self.args, self.taxonomy)
                self.assertEqual(out.source_method.tolist(), ["nli"])
                model.return_value.assert_called_once()

    def test_disabled_matches_original_score_logic_on_benchmark(self):
        df = pd.read_csv(Path(__file__).resolve().parents[1] / "benchmark.tsv", sep="\t", dtype=str, keep_default_na=False)
        records = [metalyzer.build_record(row) for _, row in df.iterrows()]
        self.seen, self.batches = [], []
        self.args.device = 0
        with patch.object(metalyzer, "pipeline", side_effect=self.fake_pipeline), contextlib.redirect_stdout(io.StringIO()):
            out = metalyzer.classify_sources(df, records, self.labels, self.args)
        expected = metalyzer.parse_source_scores(pd.DataFrame([[0.1] * len(self.labels)] * len(df), columns=self.labels), self.args.id_col, 0.2)
        pd.testing.assert_frame_equal(out[expected.columns], expected)
        self.assertEqual(self.seen, records)
        self.assertTrue((out.source_method == "nli").all())

    def test_load_failures_and_missing_anchor(self):
        with self.assertWarns(UserWarning):
            self.assertIsNone(load_taxonomy(None))
        with self.assertWarns(UserWarning):
            self.assertIsNone(load_taxonomy(self.directory / "absent"))
        path = self.directory / "names.dmp"
        path.write_text(path.read_text().replace("Ovis aries", "Unresolved sheep name"))
        with self.assertWarns(UserWarning):
            taxonomy = NCBITaxonomy(self.directory)
        self.assertIsNone(taxonomy.classify_host_taxid(9940))
        path.write_text("malformed\n")
        with self.assertWarns(UserWarning):
            self.assertIsNone(load_taxonomy(self.directory))

    def test_provenance_collisions_and_threshold(self):
        for label in ("source_method", "source_evidence"):
            with self.assertRaises(ValueError):
                metalyzer.parse_source_scores(pd.DataFrame(columns=[label]), "id")
        result = metalyzer.parse_source_scores(pd.DataFrame([[0.2, 0.2], [0.19, 0.1]], columns=["sheep", "cat"]), "id", 0.2)
        self.assertEqual(result.best_hit.tolist(), ["sheep", "unknown"])

    def test_ambiguous_anchor_and_empty_input(self):
        path = self.directory / "names.dmp"
        with path.open("a") as handle:
            handle.write("9796 | Ovis aries | | scientific name |\n")
        with self.assertWarns(UserWarning):
            taxonomy = NCBITaxonomy(self.directory)
        self.assertIsNone(taxonomy.classify_host_taxid(9940))
        with patch.object(metalyzer, "pipeline") as model:
            result = metalyzer.classify_sources(pd.DataFrame(), [], self.labels, self.args, taxonomy)
        self.assertEqual(len(result), 0)
        model.assert_not_called()

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
        with patch("taxonomy.urllib.request.urlopen", return_value=archive):
            download_taxonomy(self.directory / "download")
        self.assertFalse((self.directory / "unwanted.txt").exists())
        self.assertEqual(NCBITaxonomy(self.directory / "download").classify_host_taxid(9940).source, "sheep")
        with patch.object(metalyzer, "download_taxonomy") as download, patch.object(metalyzer, "pipeline") as model:
            metalyzer.main(["--download-taxonomy", "taxonomy"])
            download.assert_called_once_with("taxonomy")
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
