"""Offline tests for the Princeton University Art Museum fetcher parsing (no network)."""

import unittest

from src.dataset.fetchers.princeton import (
    build_image_url,
    is_web_usable,
    matches_filters,
    parse_object,
    sanitize_objectnumber,
)


def make_object(**overrides):
    obj = {
        "objectnumber": "1998-111 d",
        "displaytitle": "In Wind and Snow",
        "department": "Asian Art",
        "classification": "Paintings",
        "displaydate": "the twelfth lunar month of 1737",
        "medium": "Ink and color on paper",
        "makers": [{"id": 4240, "displayname": "Gao Fenghan 高鳳翰", "role": "Artist"},
                   {"id": 99, "displayname": "Some Donor", "role": "Donor"}],
        "primaryimage": ["https://media.artmuseum.princeton.edu/iiif/3/collection/1998-111D"],
        "restrictions": None,
        "nowebuse": "False",
    }
    obj.update(overrides)
    return obj


class TestPrincetonParsing(unittest.TestCase):
    def test_parse_object_basic_fields(self):
        parsed = parse_object(make_object())
        self.assertEqual(parsed["objectnumber"], "1998-111 d")
        self.assertEqual(parsed["title"], "In Wind and Snow")
        self.assertEqual(parsed["department"], "Asian Art")
        self.assertEqual(parsed["classification"], "Paintings")
        self.assertEqual(parsed["displaydate"], "the twelfth lunar month of 1737")
        self.assertEqual(parsed["medium"], "Ink and color on paper")
        self.assertEqual(parsed["primaryimage"],
                         "https://media.artmuseum.princeton.edu/iiif/3/collection/1998-111D")
        self.assertEqual(parsed["restrictions"], "")
        self.assertEqual(parsed["nowebuse"], "False")

    def test_parse_object_null_displaydate_becomes_empty(self):
        self.assertEqual(parse_object(make_object(displaydate=None))["displaydate"], "")

    def test_sanitize_objectnumber(self):
        self.assertEqual(sanitize_objectnumber("1998-111 d"), "1998-111_d")
        self.assertEqual(sanitize_objectnumber("1998-91 ee"), "1998-91_ee")
        self.assertEqual(sanitize_objectnumber("2003-139.7"), "2003-139.7")
        self.assertEqual(sanitize_objectnumber("y1947-138"), "y1947-138")
        self.assertEqual(sanitize_objectnumber("x(1)/2"), "x_1_2")

    def test_parse_object_artist_only_artist_role(self):
        parsed = parse_object(make_object())
        self.assertEqual(parsed["artist"], "Gao Fenghan 高鳳翰")

    def test_parse_object_primaryimage_string_fallback(self):
        parsed = parse_object(make_object(primaryimage="https://x/iiif/coll/1"))
        self.assertEqual(parsed["primaryimage"], "https://x/iiif/coll/1")
        parsed = parse_object(make_object(primaryimage=[]))
        self.assertEqual(parsed["primaryimage"], "")

    def test_parse_object_title_fallback_to_titles(self):
        obj = make_object(displaytitle="", titles=[{"title": "Fallback Title"}])
        self.assertEqual(parse_object(obj)["title"], "Fallback Title")

    def test_is_web_usable(self):
        self.assertTrue(is_web_usable(parse_object(make_object())))
        # non-empty restrictions -> not usable
        self.assertFalse(is_web_usable(parse_object(make_object(restrictions="Restricted"))))
        self.assertFalse(is_web_usable(parse_object(make_object(restrictions="Copyright"))))
        # nowebuse string forms
        self.assertFalse(is_web_usable(parse_object(make_object(nowebuse="True"))))
        self.assertFalse(is_web_usable(parse_object(make_object(nowebuse="true"))))
        self.assertFalse(is_web_usable(parse_object(make_object(nowebuse="1"))))
        self.assertTrue(is_web_usable(parse_object(make_object(nowebuse="False"))))
        # nowebuse boolean/int forms
        self.assertFalse(is_web_usable(parse_object(make_object(nowebuse=True))))
        self.assertFalse(is_web_usable(parse_object(make_object(nowebuse=1))))
        # no primary image -> not usable
        self.assertFalse(is_web_usable(parse_object(make_object(primaryimage=[]))))

    def test_matches_filters(self):
        parsed = parse_object(make_object())
        self.assertTrue(matches_filters(parsed, "Asian Art", ("Paintings", "Calligraphy")))
        self.assertFalse(matches_filters(parsed, "European Art", None))
        self.assertFalse(matches_filters(parsed, "Asian Art", ("Calligraphy",)))
        self.assertTrue(matches_filters(parsed, None, None))
        # restricted never matches, even with matching department/class
        restricted = parse_object(make_object(restrictions="Restricted"))
        self.assertFalse(matches_filters(restricted, "Asian Art", ("Paintings",)))

    def test_build_image_url(self):
        self.assertEqual(
            build_image_url("https://media.artmuseum.princeton.edu/iiif/3/collection/1998-111D"),
            "https://media.artmuseum.princeton.edu/iiif/3/collection/1998-111D/full/max/0/default.jpg",
        )
        self.assertEqual(
            build_image_url("https://media.artmuseum.princeton.edu/iiif/3/collection/1998-111D",
                            "!1600,1600"),
            "https://media.artmuseum.princeton.edu/iiif/3/collection/1998-111D/full/!1600,1600/0/default.jpg",
        )


if __name__ == "__main__":
    unittest.main()
