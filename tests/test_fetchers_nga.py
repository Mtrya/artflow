"""NGA filtering and metadata extraction checked against annotated input records."""

import unittest

import pandas as pd

from src.dataset.fetchers.nga import filter_rows, row_to_record


def sample_joined_df():
    return pd.DataFrame([
        # open access painting by Monet in range
        {"uuid": "u1", "objectid": 100, "title": "Water Lilies", "attribution": "Claude Monet",
         "displaydate": "1907", "beginyear": 1907, "endyear": 1907,
         "medium": "oil on canvas", "classification": "Painting", "openaccess": 1,
         "width": 2000, "height": 1500},
        # open access painting out of year range
        {"uuid": "u2", "objectid": 101, "title": "Madonna", "attribution": "Raphael",
         "displaydate": "c. 1510", "beginyear": 1508, "endyear": 1512,
         "medium": "oil on panel", "classification": "Painting", "openaccess": 1,
         "width": 1000, "height": 1200},
        # print, excluded by classification
        {"uuid": "u3", "objectid": 102, "title": "Etching", "attribution": "Rembrandt",
         "displaydate": "1648", "beginyear": 1648, "endyear": 1648,
         "medium": "etching", "classification": "Print", "openaccess": 1,
         "width": 500, "height": 600},
        # not open access
        {"uuid": "u4", "objectid": 103, "title": "Sketch", "attribution": "Claude Monet",
         "displaydate": "1872", "beginyear": 1872, "endyear": 1872,
         "medium": "graphite", "classification": "Painting", "openaccess": 0,
         "width": 100, "height": 100},
        # unknown year (beginyear=0), painting, open access
        {"uuid": "u5", "objectid": 104, "title": "Untitled", "attribution": "Anonymous",
         "displaydate": "", "beginyear": 0, "endyear": 0,
         "medium": "oil", "classification": "Painting", "openaccess": 1,
         "width": 800, "height": 600},
    ])


class TestNgaFilters(unittest.TestCase):
    def test_default_only_open_access_paintings(self):
        df = filter_rows(sample_joined_df())
        self.assertEqual(set(df["uuid"]), {"u1", "u2", "u5"})

    def test_all_filters_combined(self):
        df = filter_rows(sample_joined_df(), classification="Painting",
                         year_begin=1900, year_end=1910, attribution="Monet")
        self.assertEqual(set(df["uuid"]), {"u1"})


class TestNgaRecord(unittest.TestCase):

    def test_row_to_record(self):
        row = {"uuid": "u1", "objectid": 100, "title": "Water Lilies",
               "attribution": "Claude Monet", "displaydate": "1907",
               "beginyear": 1907, "endyear": 1907, "medium": "oil on canvas",
               "classification": "Painting", "width": 2000, "height": 1500}
        rec = row_to_record(row)
        self.assertEqual(rec["image_id"], "nga-u1")
        self.assertEqual(rec["source"], "nga")
        self.assertEqual(rec["object_id"], "100")
        self.assertEqual(rec["artist"], "Claude Monet")
        self.assertEqual(rec["date"], "1907")
        self.assertEqual(rec["medium"], "oil on canvas")
        self.assertEqual(rec["beginyear"], 1907)
        self.assertEqual(rec["width"], 2000)
        self.assertEqual(rec["license"], "CC0 (NGA open data / PD images)")
