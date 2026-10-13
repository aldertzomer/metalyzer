"""CPU-only checks for the adjudicated current benchmark population."""
from pathlib import Path
import re
import tempfile
import unittest

import pandas as pd

from benchmark_smoke.current_dataset import ROOT, build_dataset
from benchmark_smoke.run_current_benchmark import checked_cache_hash, options
from modules.input import read_tsv


class CurrentDatasetTests(unittest.TestCase):
    def test_real_reference_selection_and_documented_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            manifest = build_dataset(destination)
            historical = read_tsv(ROOT / "benchmark.tsv")
            truth = read_tsv(ROOT / "benchmark_true_labels.tsv")
            selected = read_tsv(destination / "benchmark_atb_1275.tsv")
            requests = read_tsv(destination / "benchmark_accessions_1275.tsv")
            mapping = read_tsv(destination / "run_to_biosample.tsv")
            excluded = read_tsv(destination / "excluded_accessions.tsv")
            conflicts = read_tsv(ROOT / "benchmark_smoke/biosample_diagnostics/goat_host_conflicts.tsv")
            self.assertEqual(len(selected), 1275)
            self.assertEqual(len(truth), 1275)
            self.assertEqual(selected.run_accession.tolist(),
                             historical[historical.run_accession.isin(truth.run_accession)].run_accession.tolist())
            self.assertEqual(requests.run_accession.tolist(), selected.run_accession.tolist())
            self.assertEqual(mapping.run_accession.tolist(), selected.run_accession.tolist())
            self.assertEqual(set(selected.run_accession), set(truth.run_accession))
            self.assertEqual(set(excluded.run_accession), set(conflicts.run_accession) | {"ERR2197811"})
            self.assertEqual(len(excluded), 45)
            self.assertEqual(mapping.accession.value_counts().loc["SAMN08169257"], 2)
            self.assertEqual(mapping.run_accession.nunique(), 1275)
            self.assertEqual(manifest["included_runs"], 1275)
            self.assertFalse(manifest["labels_from_predictions"])
            self.assertEqual(read_tsv(destination / "benchmark_true_labels_1275.tsv").to_dict("records"),
                             truth.to_dict("records"))
            self.assertEqual((destination / "biosample_accessions.txt").read_text().splitlines(),
                             mapping.accession.tolist())

    def test_missing_reference_run_and_extra_exclusion_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = pd.DataFrame({"run_accession": ["r1", "r2", "r3", "r4"],
                                     "sample_accession": ["s1", "s2", "s2", "s4"],
                                     "host": ["cat"] * 4})
            requests = pd.DataFrame({"run_accession": ["r1", "r2", "r3", "r4"]})
            truth = pd.DataFrame({"run_accession": ["r1", "r2", "r3"],
                                  "final_species": ["unknown", "cat", "dog"]})
            for name, frame in [("metadata", metadata), ("requests", requests), ("truth", truth)]:
                frame.to_csv(root / f"{name}.tsv", sep="\t", index=False)
            arguments = dict(metadata_path=root / "metadata.tsv", accessions_path=root / "requests.tsv",
                             truth_path=root / "truth.tsv", expected_rows=3,
                             expected_excluded={"r4"})
            result = build_dataset(root / "derived", **arguments)
            self.assertEqual(result["included_runs"], 3)
            self.assertEqual(read_tsv(root / "derived/benchmark_true_labels_1275.tsv").final_species.tolist(),
                             ["unknown", "cat", "dog"])
            with self.assertRaisesRegex(ValueError, "Derived benchmark input changed"):
                (root / "derived/benchmark_atb_1275.tsv").write_text("tampered\n")
                build_dataset(root / "derived", **arguments)
            (root / "derived/benchmark_atb_1275.tsv").unlink()
            with self.assertRaisesRegex(ValueError, "Excluded run set differs"):
                build_dataset(root / "other", **(arguments | {"expected_excluded": {"r3"}}))
            truth.loc[2, "run_accession"] = "missing"
            truth.to_csv(root / "truth.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "missing from ATB metadata"):
                build_dataset(root / "missing", **arguments)

    def test_duplicate_run_and_mismatched_request_sets_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = pd.DataFrame({"run_accession": ["r1", "r1"], "sample_accession": ["s1", "s2"]})
            requests = pd.DataFrame({"run_accession": ["r1", "r2"]})
            truth = pd.DataFrame({"run_accession": ["r1"], "final_species": ["cat"]})
            for name, frame in [("metadata", metadata), ("requests", requests), ("truth", truth)]:
                frame.to_csv(root / f"{name}.tsv", sep="\t", index=False)
            arguments = dict(metadata_path=root / "metadata.tsv", accessions_path=root / "requests.tsv",
                             truth_path=root / "truth.tsv", expected_rows=1, expected_excluded={"r2"})
            with self.assertRaisesRegex(ValueError, "duplicate run_accession"):
                build_dataset(root / "out", **arguments)
            metadata.loc[1, "run_accession"] = "r3"
            metadata.to_csv(root / "metadata.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "run sets differ"):
                build_dataset(root / "out", **arguments)


class CurrentRunnerTests(unittest.TestCase):
    def test_changed_biosample_cache_hash_rejects_result_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "selected.pkl"
            cache.write_bytes(b"original")
            original = checked_cache_hash(cache, {})
            self.assertEqual(checked_cache_hash(cache, {"selected_biosample_cache_sha256": original}), original)
            cache.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "BioSample input hash changed"):
                checked_cache_hash(cache, {"selected_biosample_cache_sha256": original})

    def test_identical_taxonomy_settings_and_explicit_legacy_switch(self):
        root = Path("/tmp/current-benchmark-test")
        filtered = options("biosample", "mistral", "filtered", root, Path("/taxonomy"), Path("/key"))
        original = options("biosample", "mistral", "unfiltered", root, Path("/taxonomy"), Path("/key"))
        comparable = [arg for arg in original if arg != "--disable-noisy-field-exclusion"]
        comparable[comparable.index("--out") + 1] = filtered[filtered.index("--out") + 1]
        self.assertEqual(comparable, filtered)
        self.assertIn("--disable-noisy-field-exclusion", original)
        self.assertIn("--other-animal-taxonomy", filtered)
        self.assertIn("mistral-small-latest", filtered)
        self.assertIn("12345", filtered)
        self.assertEqual(options("tsv", "nli", "filtered", root, Path("/taxonomy"), Path("/key"))[0],
                         "--metadata")

    def test_readme_relative_links_exist(self):
        for markdown in [ROOT / "README.md", ROOT / "benchmark_smoke/current_benchmark/README.md",
                         ROOT / "benchmark_smoke/archive/README_before_1275.md"]:
            if not markdown.exists():
                continue
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown.read_text()):
                if target.startswith(("#", "http:", "https:")):
                    continue
                self.assertTrue((markdown.parent / target.split("#", 1)[0]).exists(),
                                f"{markdown}: {target}")


if __name__ == "__main__":
    unittest.main()
