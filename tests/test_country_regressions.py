"""Regression tests for country parsing and BioSample field selection."""
import unittest

import pandas as pd

from modules.country import extract_country_from_row, normalize_country


class CountryNormalizationTests(unittest.TestCase):
    def test_country_names_and_aliases(self):
        cases = {
            "Turkey": "Turkey",
            "Türkiye": "Turkey",
            "Turkey: Ankara": "Turkey",
            "Korea": "South Korea",
            "Republic of Korea": "South Korea",
            "Korea, Republic of": "South Korea",
            "South Korea": "South Korea",
            "North Korea": "North Korea",
            "DPRK": "North Korea",
            "Korea, Democratic People's Republic of": "North Korea",
            "Democratic People's Republic of Korea": "North Korea",
            "Congo": "Republic of the Congo",
            "Republic of the Congo": "Republic of the Congo",
            "Congo, The Democratic Republic of the": "Democratic Republic of the Congo",
            "Democratic Republic of the Congo": "Democratic Republic of the Congo",
            "DR Congo": "Democratic Republic of the Congo",
            "Congo-Kinshasa": "Democratic Republic of the Congo",
            "Virgin Islands, U.S.": "US Virgin Islands",
            "US Virgin Islands": "US Virgin Islands",
            "Virgin Islands, British": "British Virgin Islands",
            "British Virgin Islands": "British Virgin Islands",
            "Virgin Islands": "unknown",
            "United States": "United States",
            "USA:SC": "United States",
            "United Kingdom: Scotland": "United Kingdom",
            "Netherlands": "Netherlands",
            "Czech Republic": "Czechia",
            "Viet Nam": "Vietnam",
            "NA": "unknown",
            "N/A": "unknown",
            "unknown": "unknown",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), expected)

    def test_priority_and_original_attributes(self):
        cases = [
            ({"original_attributes.sample_identifier": "AL",
              "geo_loc_name": "Netherlands"}, "Netherlands"),
            ({"original_attributes.country": "Albania",
              "geo_loc_name": "Netherlands"}, "Netherlands"),
            ({"original_attributes.sample_type": "TH",
              "country": "United Kingdom"}, "United Kingdom"),
            ({"country": "USA:SC",
              "original_attributes.specimen_name": "VG"}, "United States"),
            ({"country": "NA", "location": "France"}, "France"),
            ({"original_attributes.sample_identifier": "AL"}, "unknown"),
        ]
        for metadata, expected in cases:
            with self.subTest(metadata=metadata):
                self.assertEqual(extract_country_from_row(pd.Series(metadata), ""), expected)

    def test_record_fallback_and_multiple_candidates(self):
        self.assertEqual(
            extract_country_from_row(pd.Series({"sample_title": "unrelated"}),
                                     "country: N/A; geo loc name: Turkey"),
            "Turkey",
        )
        self.assertEqual(
            extract_country_from_row(pd.Series({"sample_title": "unrelated"}),
                                     "original attributes: AL; sample title: TH"),
            "unknown",
        )


if __name__ == "__main__":
    unittest.main()
