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
            9925: (33208, "Capra hircus"), 9913: (33208, "Bos taurus"),
            9823: (33208, "Sus scrofa"), 9825: (9823, "Sus scrofa domesticus"),
            8839: (8782, "Anas platyrhynchos"),
            501: (33208, "Shared scientific name"), 502: (33208, "Shared scientific name"),
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
        self.assertEqual(country.extract_country_from_row(row, record), "US")
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
        self.assertTrue((out.country == "US").all())
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


class HostNameTaxonomyTests(unittest.TestCase):
    def setUp(self):
        TaxonomyTests.setUp(self)
        with (self.directory / 'names.dmp').open('a') as handle:
            for taxid, name, kind in [(9031, 'chicken', 'common name'), (9103, 'turkey', 'genbank common name'),
                                      (9606, 'human', 'common name'), (9031, 'poultry', 'synonym'),
                                      (9103, 'poultry', 'synonym'), (9031, 'Felis catus', 'synonym'),
                                      (9000, 'old sheep', 'synonym'), (9940, 'old sheep', 'synonym')]:
                handle.write(f'{taxid} | {name} | | {kind} |\n')
        self.taxonomy = NCBITaxonomy(self.directory)
        names = ('chicken', 'turkey', 'goat', 'sheep', 'cattle', 'human', 'cat', 'pig', 'dog', 'wildbird')
        self.sources = SourceVocabulary(names, 'accession',
                                        ((9031,), (9103,), (9925,), (9940,), (9913,), (9606,),
                                         (9685,), (9825,), (9615,), ()))
        self.anchors = configured_anchors(self.sources, self.taxonomy)

    def run_rows(self, rows, sources=None):
        batch = input_module.prepare_batch(pd.DataFrame(rows), id_col='accession')
        return deterministic_source.run(batch, sources or self.sources, self.taxonomy), batch

    def test_exact_names_and_case_whitespace_normalization(self):
        scientific = [('Gallus gallus', 9031, 'chicken'), ('Meleagris gallopavo', 9103, 'turkey'),
                      ('Capra hircus', 9925, 'goat'), ('Ovis aries', 9940, 'sheep'),
                      ('Bos taurus', 9913, 'cattle'), ('Homo sapiens', 9606, 'human'), ('Felis catus', 9685, 'cat')]
        rows = [{'accession': str(i), 'host': f'  {name.upper().replace(" ", "   ")}  '}
                for i, (name, _, _) in enumerate(scientific)]
        result, batch = self.run_rows(rows)
        self.assertEqual(result.table.best_hit.tolist(), [source for _, _, source in scientific])
        self.assertEqual(result.table.source_method.tolist(), ['host_name'] * 7)
        self.assertTrue(result.table.source_llm_score.isna().all())
        self.assertNotIn('host_tax_id', batch.metadata)
        for i, (name, taxid, _) in enumerate(scientific):
            self.assertEqual(self.taxonomy.resolve_host_name(name).taxid, taxid)
            self.assertIn(f'taxid={taxid}; {name}', result.table.loc[i, 'source_evidence'])

    def test_unique_aliases_scientific_precedence_and_merged_alias(self):
        for name, expected in [('chicken', 9031), ('turkey', 9103), ('human', 9606), ('old sheep', 9940)]:
            self.assertEqual(self.taxonomy.resolve_host_name(name).taxid, expected)
        self.assertEqual(self.taxonomy.resolve_host_name('Felis catus').taxid, 9685)
        self.assertEqual(self.taxonomy.resolve_host_name('Felis catus').match_type, 'scientific name')

    def test_ambiguous_missing_invalid_and_nonexact_names(self):
        for name in ['poultry', 'Shared scientific name']:
            match = self.taxonomy.resolve_host_name(name)
            self.assertIsNone(match.taxid)
            self.assertEqual(match.match_type, 'ambiguous')
        for name in [None, pd.NA, float('nan'), '', ' NA ', 'unknown', 'not collected', 'not applicable',
                     'tur', 'Gallus gallus sample', 'Gallus gallus | Homo sapiens', 'nonsense']:
            self.assertIsNone(self.taxonomy.resolve_host_name(name).taxid)
        result, _ = self.run_rows([{'host': 'poultry'}, {'host': 'Shared scientific name'}, {'host': 'missing'}])
        self.assertTrue(result.table.empty)

    def test_numeric_aliases_agreement_missing_invalid_and_conflicts(self):
        rows = [{'host_tax_id': '9940'}, {'host_taxid': '9000'},
                {'host_tax_id': 'NA', 'host': 'Ovis aries'},
                {'host_tax_id': '9940', 'host': 'Ovis aries'},
                {'host_tax_id': '9940', 'host_scientific_name': 'Sheep subspecies'},
                {'host_tax_id': 'invalid', 'host_scientific_name': 'Bos taurus'},
                {'host_tax_id': '9913', 'host': 'unrecognized host'},
                {'host_tax_id': '9913', 'host': 'Capra hircus'},
                {'host_tax_id': '9913', 'host_taxid': '9925'},
                {'host': 'Felis catus', 'host_scientific_name': 'Homo sapiens'}]
        before = pd.DataFrame(rows)
        with contextlib.redirect_stdout(io.StringIO()) as screen:
            result, batch = self.run_rows(rows)
        self.assertEqual(result.table.index.tolist(), list(range(7)))
        self.assertEqual(result.table.source_method.tolist(), ['host_tax_id', 'host_tax_id', 'host_name',
                                                              'host_tax_id', 'host_tax_id', 'host_name', 'host_tax_id'])
        self.assertIn('host_taxid=9940', result.table.loc[1, 'source_evidence'])
        pd.testing.assert_frame_equal(batch.metadata, before)
        self.assertIn('Host taxonomy conflict', screen.getvalue())
        self.assertIn('Ambiguous/conflicting hosts: 3', screen.getvalue())
        self.assertIn('Total deterministic assignments: 7', screen.getvalue())

    def test_original_host_scientific_alias_and_conflict(self):
        result, _ = self.run_rows([{'original_attributes.host scientific name': 'Meleagris gallopavo'},
                                   {'host': 'Gallus gallus', 'original_attributes.host scientific name': 'Homo sapiens'},
                                   {'host': 'Gallus gallus', 'original_attributes.host_scientific_name': 'Gallus gallus'}])
        self.assertEqual(result.table.index.tolist(), [0, 2])
        self.assertEqual(result.table.best_hit.tolist(), ['turkey', 'chicken'])
        self.assertIn('original_attributes.host scientific name=', result.table.loc[0, 'source_evidence'])

    def test_incompatible_hosts_are_not_hidden_by_a_broad_anchor(self):
        broad = SourceVocabulary(('animal',), 'accession', ((33208,),))
        result, _ = self.run_rows([{'host_tax_id': '9031', 'host': 'Homo sapiens'},
                                   {'host': 'Felis catus', 'original_attributes.host scientific name': 'Homo sapiens'},
                                   {'host_tax_id': '33208', 'host': 'Felis catus'}], broad)
        self.assertEqual(result.table.index.tolist(), [2])
        result, _ = self.run_rows([{'host_tax_id': '8839', 'host': 'Gallus gallus'}])
        self.assertTrue(result.table.empty)

    def test_pathogen_free_text_and_unconfigured_wildlife_never_assign(self):
        result, _ = self.run_rows([{'tax_id': '9031', 'organism': 'Gallus gallus'},
                                   {'title': 'Gallus gallus', 'description': 'host Homo sapiens', 'isolation_source': 'chicken'},
                                   {'host': 'Sus scrofa'}, {'host': 'Anas platyrhynchos'}])
        self.assertTrue(result.table.empty)
        result, _ = self.run_rows([{'host': 'Sus scrofa domesticus'}])
        self.assertEqual(result.table.best_hit.tolist(), ['pig'])
        result, _ = self.run_rows([{'host': 'Gallus gallus'}], SourceVocabulary(('chicken',), 'accession'))
        self.assertTrue(result.table.empty)

    def test_one_names_pass_per_batch_and_cached_results(self):
        from modules.deterministic_source import _rows
        with patch.object(deterministic_source, '_rows', wraps=_rows) as reading:
            self.taxonomy.resolve_host_names(['Gallus gallus', 'turkey', 'poultry', 'missing'])
            for _ in range(10):
                for name in ['Gallus gallus', 'TURKEY', 'poultry']:
                    self.taxonomy.resolve_host_name(name)
            self.assertEqual(reading.call_count, 1)

    def test_all_methods_skip_fallback_and_verification_still_runs(self):
        from modules import llm, mistral, verification
        batch = input_module.prepare_batch(pd.DataFrame({'accession': ['a', 'b'], 'host': ['Gallus gallus', 'Meleagris gallopavo']}), id_col='accession')
        for method in ['nli', 'llm', 'mistral']:
            with self.subTest(method=method), patch.object(nli, 'run') as nr, patch.object(llm, 'run') as lr, patch.object(mistral, 'run') as mr:
                result = metalyzer.classify_sources(batch, self.sources, method=method, taxonomy=self.taxonomy)
                nr.assert_not_called(); lr.assert_not_called(); mr.assert_not_called()
                self.assertEqual(result.table.source_method.tolist(), ['host_name', 'host_name'])
        with patch.object(nli, 'pipeline') as factory:
            factory.return_value.side_effect = lambda texts, **kw: [{'labels': kw['candidate_labels'], 'scores': [.97]} for _ in texts]
            verified = verification.run(batch, self.sources, result, nli.NLIConfig(device=-1))
        self.assertEqual(verified.values.tolist(), [.97, .97])

    def test_cli_log_reports_both_routes_with_regular_output(self):
        pd.DataFrame({'accession': ['a', 'b'], 'host_taxid': ['9031', ''], 'host': ['Gallus gallus', 'Meleagris gallopavo']}).to_csv(self.directory / 'host.tsv', sep='\t', index=False)
        pd.DataFrame({'source': self.sources.labels, 'taxonomy_anchors': [','.join(map(str,a)) for a in self.sources.taxonomy_anchors]}).to_csv(self.directory / 'labels.tsv', sep='\t', index=False)
        with patch.object(nli, 'pipeline') as model, contextlib.redirect_stdout(io.StringIO()):
            metalyzer.main(['--metadata', str(self.directory / 'host.tsv'), '--sources', str(self.directory / 'labels.tsv'),
                            '--out', str(self.directory / 'out.tsv'), '--id-col', 'accession',
                            '--taxonomy-dir', str(self.directory), '--disable-verify-source'])
        model.assert_not_called()
        log = (self.directory / 'out.tsv.log').read_text()
        for text in ['Explicit host taxonomy ID: 1', 'Exact host-name resolution: 1',
                     'deterministic taxonomy assignments: 2', 'exact host-name assignments: 1', 'Taxonomy file names.dmp:']:
            self.assertIn(text, log)
        output = input_module.read_tsv(self.directory / 'out.tsv')
        self.assertEqual(output.source_method.tolist(), ['host_tax_id', 'host_name'])
        self.assertIn('nli_verification_score', output)
        self.assertNotIn('host_taxid', output)  # existing regular output schema


if __name__ == "__main__":
    unittest.main()
