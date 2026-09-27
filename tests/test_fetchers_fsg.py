"""FSG metadata extraction checked against an API-response fixture."""

import unittest

from src.dataset.fetchers.fsg import parse_record, pick_largest_resource


def make_record():
    rec = {
        "id": "ld1-0", "version": "", "unitCode": "FSG", "type": "edanmdm",
        "content": {
            "descriptiveNonRepeating": {
                "title": {"label": "Title", "content": "Two magpies under an orchid cliff"},
                "record_ID": "fsg_F1911.309",
                "record_link": "https://asia.si.edu/object/F1911.309/",
                "online_media": {"media": [{
                    "id": "media:FS-5907_12", "type": "Images", "idsId": "FS-5907_12",
                    "usage": {"access": "CC0"},
                    "resources": [
                        {"label": "High-resolution JPEG", "url": "https://ids.si.edu/ids/download?id=FS-5907_12.jpg",
                         "width": 1106, "height": 2537},
                        {"label": "Thumbnail Image", "url": "https://ids.si.edu/ids/download?id=FS-5907_12_thumb"},
                    ],
                }]},
            },
            "freetext": {
                "date": [{"label": "Date", "content": "15th century"},
                         {"label": "Period", "content": "Ming dynasty"}],
                "name": [{"label": "Artist", "content": "Possibly by Lü Ji (ca. 1420-ca. 1505)"},
                         {"label": "Previous custodian or owner", "content": "Charles Lang Freer"}],
                "topic": [{"label": "Topic", "content": "bird"}, {"label": "Topic", "content": "Chinese Art"},
                          {"label": "Topic", "content": "Ming dynasty (1368 - 1644)"}],
                "objectType": [{"label": "Type", "content": "Painting"}],
                "physicalDescription": [
                    {"label": "Medium", "content": "Ink and color on silk"},
                    {"label": "Dimensions", "content": "H x W (image): 133.2 x 52.2 cm"}],
                "identifier": [{"label": "Accession Number", "content": "F1911.309"}],
                "creditLine": [{"label": "Credit Line", "content": "Gift of Charles Lang Freer"}],
                "objectRights": [{"label": "Restrictions & Rights", "content": "CC0"}],
            },
            "indexedStructured": {
                "culture": ["Chinese"],
                "object_type": ["Paintings"],
                "date": ["1500s", "1400s"],
            },
        },
    }
    return rec


class TestFsgParsing(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_record(make_record())

    def test_parse_record_basic_fields(self):
        self.assertEqual(self.parsed["record_id"], "fsg_F1911.309")
        self.assertEqual(self.parsed["title"], "Two magpies under an orchid cliff")
        self.assertEqual(self.parsed["object_url"], "https://asia.si.edu/object/F1911.309/")
        self.assertEqual(self.parsed["date"], "15th century")
        self.assertEqual(self.parsed["period"], "Ming dynasty")
        self.assertEqual(self.parsed["culture"], "Chinese")
        self.assertEqual(self.parsed["medium"], "Ink and color on silk")
        self.assertEqual(self.parsed["classification"], "Paintings")
        self.assertEqual(self.parsed["object_type"], "Painting")
        self.assertEqual(self.parsed["accession"], "F1911.309")
        self.assertEqual(self.parsed["credit_line"], "Gift of Charles Lang Freer")
        self.assertEqual(self.parsed["dimensions"], "H x W (image): 133.2 x 52.2 cm")

    def test_parse_record_artist_only_artist_label(self):
        # only the "Artist" labelled entries are joined, not previous owners
        self.assertEqual(self.parsed["artist"], "Possibly by Lü Ji (ca. 1420-ca. 1505)")

    def test_pick_largest_resource(self):
        media = {"resources": [
            {"label": "Thumbnail Image", "url": "t.jpg"},
            {"label": "High-resolution JPEG", "url": "h.jpg", "width": 1106, "height": 2537},
            {"label": "Screen Image", "url": "s.jpg", "width": 500, "height": 1000},
        ]}
        self.assertEqual(pick_largest_resource(media)["url"], "h.jpg")
