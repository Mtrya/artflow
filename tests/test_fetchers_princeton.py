"""Princeton parsing and image eligibility checked against API-response fixtures."""

import unittest

from src.dataset.fetchers.princeton import is_web_usable, parse_object


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

    def test_parse_object_artist_only_artist_role(self):
        parsed = parse_object(make_object())
        self.assertEqual(parsed["artist"], "Gao Fenghan 高鳳翰")

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
