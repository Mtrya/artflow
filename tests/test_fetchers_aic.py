"""Offline tests for the Art Institute of Chicago (AIC) fetcher (no network)."""

import unittest

from src.dataset.fetchers.aic import build_image_url, build_query, item_to_record


class TestAicQuery(unittest.TestCase):
    def test_default_query(self):
        q = build_query(1860, 1910, "Painting", None)
        filters = q["bool"]["filter"]
        self.assertIn({"term": {"is_public_domain": True}}, filters)
        self.assertIn({"range": {"date_start": {"gte": 1860, "lte": 1910}}}, filters)
        self.assertIn({"term": {"artwork_type_title.keyword": "Painting"}}, filters)
        self.assertNotIn("must", q["bool"])

    def test_keyword_adds_query_string(self):
        q = build_query(1860, 1910, "Painting", "impressionism")
        self.assertEqual(
            q["bool"]["must"],
            [{"query_string": {"query": "impressionism", "default_operator": "AND"}}],
        )

    def test_no_artwork_type(self):
        q = build_query(1860, 1910, None, None)
        filters = q["bool"]["filter"]
        self.assertEqual(len(filters), 2)
        self.assertTrue(all("artwork_type_title" not in str(f) for f in filters))

    def test_match_fallback(self):
        q = build_query(1860, 1910, "Painting", None, use_match=True)
        self.assertIn({"match": {"artwork_type_title": "Painting"}}, q["bool"]["filter"])
        self.assertNotIn({"term": {"artwork_type_title.keyword": "Painting"}}, q["bool"]["filter"])


class TestAicRecord(unittest.TestCase):
    def test_build_image_url(self):
        self.assertEqual(
            build_image_url("abc-123", 843),
            "https://www.artic.edu/iiif/2/abc-123/full/843,/0/default.jpg",
        )

    def test_item_to_record(self):
        item = {
            "id": 28560,
            "title": "The Bedroom",
            "artist_display": "Vincent van Gogh (Dutch, 1853\u20131890)",
            "date_display": "1889",
            "medium_display": "Oil on canvas",
            "classification_title": "oil on canvas",
            "style_title": "Post-Impressionism",
            "image_id": "6644829f-f292-c5c4-a73c-0356a6fdbf0d",
            "is_public_domain": True,
        }
        rec = item_to_record(item)
        self.assertEqual(rec["image_id"], "aic-28560")
        self.assertEqual(rec["source"], "aic")
        self.assertEqual(rec["object_id"], "28560")
        self.assertEqual(rec["artist"], "Vincent van Gogh (Dutch, 1853\u20131890)")
        self.assertEqual(rec["period"], "Post-Impressionism")
        self.assertEqual(rec["style_title"], "Post-Impressionism")
        self.assertEqual(rec["classification"], "oil on canvas")
        self.assertEqual(rec["license"], "CC0 (AIC public domain)")
        self.assertEqual(rec["object_url"], "https://www.artic.edu/artworks/28560")
        self.assertIn("/full/843,/0/default.jpg", rec["iiif_url"])
        self.assertNotIn("local_path", rec)

    def test_item_to_record_no_style(self):
        item = {"id": 1, "title": "x", "image_id": "img1"}
        rec = item_to_record(item)
        self.assertNotIn("period", rec)
        self.assertNotIn("style_title", rec)
        self.assertEqual(rec["title"], "x")
        self.assertEqual(rec["artist"], "")


if __name__ == "__main__":
    unittest.main()
