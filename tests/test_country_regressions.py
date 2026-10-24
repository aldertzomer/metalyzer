"""Regression tests for ISO country parsing and BioSample field selection."""
import csv
from pathlib import Path
import unittest

import pandas as pd
import pycountry

from modules.country import extract_country_from_row, normalize_country


class CountryNormalizationTests(unittest.TestCase):
    def test_country_names_and_aliases(self):
        cases = {
            "Turkey": "TR",
            "Türkiye": "TR",
            "Turkey: Ankara": "TR",
            "Korea": "KR",
            "Republic of Korea": "KR",
            "Korea, Republic of": "KR",
            "South Korea": "KR",
            "North Korea": "KP",
            "DPRK": "KP",
            "Korea, Democratic People's Republic of": "KP",
            "Democratic People's Republic of Korea": "KP",
            "Congo": "CG",
            "Republic of the Congo": "CG",
            "Congo, The Democratic Republic of the": "CD",
            "Democratic Republic of the Congo": "CD",
            "DR Congo": "CD",
            "Congo-Kinshasa": "CD",
            "Virgin Islands, U.S.": "VI",
            "US Virgin Islands": "VI",
            "Virgin Islands, British": "VG",
            "British Virgin Islands": "VG",
            "Virgin Islands": "unknown",
            "United States": "US",
            "USA:SC": "US",
            "United Kingdom: Scotland": "GB",
            "Netherlands": "NL",
            "Czech Republic": "CZ",
            "Viet Nam": "VN",
            "NA": "unknown",
            "N/A": "unknown",
            "unknown": "unknown",
            "US": "US",
            "GB": "GB",
            "TR": "TR",
            "KR": "KR",
            "KP": "KP",
            "CG": "CG",
            "CD": "CD",
            "VI": "VI",
            "VG": "VG",
            "TW": "TW",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), expected)

    def test_namibia_na_ambiguity(self):
        # "NA" is commonly a missing marker as well as Namibia's ISO2
        # code.  Do not reinterpret absent metadata as Namibia.
        self.assertEqual(normalize_country("NA"), "unknown")
        self.assertEqual(normalize_country("N/A"), "unknown")
        self.assertEqual(normalize_country("Namibia"), "NA")
        self.assertEqual(normalize_country("Namibia: Windhoek"), "NA")
        self.assertEqual(
            extract_country_from_row(pd.Series({"geo_loc_name": "Namibia"}), ""),
            "NA",
        )
        self.assertEqual(
            extract_country_from_row(pd.Series({"geo_loc_name": "NA"}), ""),
            "unknown",
        )

    def test_priority_and_original_attributes(self):
        cases = [
            ({"original_attributes.sample_identifier": "AL",
              "geo_loc_name": "Netherlands"}, "NL"),
            ({"original_attributes.country": "Albania",
              "geo_loc_name": "Netherlands"}, "NL"),
            ({"original_attributes.sample_type": "TH",
              "country": "United Kingdom"}, "GB"),
            ({"country": "USA:SC",
              "original_attributes.specimen_name": "VG"}, "US"),
            ({"country": "NA", "location": "France"}, "FR"),
            ({"original_attributes.sample_identifier": "AL"}, "unknown"),
        ]
        for metadata, expected in cases:
            with self.subTest(metadata=metadata):
                self.assertEqual(extract_country_from_row(pd.Series(metadata), ""), expected)

    def test_record_fallback_and_multiple_candidates(self):
        self.assertEqual(
            extract_country_from_row(pd.Series({"sample_title": "unrelated"}),
                                     "country: N/A; geo loc name: Turkey"),
            "TR",
        )
        self.assertEqual(
            extract_country_from_row(pd.Series({"sample_title": "unrelated"}),
                                     "original attributes: AL; sample title: TH"),
            "unknown",
        )

    def test_reference_countries_use_iso2(self):
        root = Path(__file__).resolve().parents[1]
        canonical = root / "benchmark_true_labels.tsv"
        benchmark = root / "benchmark_smoke/current_benchmark/inputs/benchmark_true_labels_1275.tsv"
        self.assertEqual(canonical.read_bytes(), benchmark.read_bytes())

        with canonical.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 1275)
        self.assertEqual(len({row["run_accession"] for row in rows}), 1275)
        for row in rows:
            value = row["final_country"]
            with self.subTest(run_accession=row["run_accession"], country=value):
                self.assertTrue(
                    value == "unknown" or (
                        len(value) == 2
                        and value.isupper()
                        and pycountry.countries.get(alpha_2=value) is not None
                    )
                )


if __name__ == "__main__":
    unittest.main()
