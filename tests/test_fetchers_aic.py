"""AIC metadata extraction checked against an API-response fixture."""

import unittest

from src.dataset.fetchers.aic import item_to_record


class TestAicRecord(unittest.TestCase):

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
