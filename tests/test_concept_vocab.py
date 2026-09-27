"""Unit tests for src.dataset.concept_vocab — pure functions and merging."""

import json

from src.dataset.concept_vocab import (Concept, aat_style_ok, merge,
                                       normalize_en, singularize_last,
                                       split_qualifier, wordnet_form_ok)


def test_normalize_en_strips_accents_and_case():
    assert normalize_en("Aide-Mémoires") == "aide-memoires"
    assert normalize_en("  Red-Figure  ") == "red-figure"


def test_split_qualifier():
    assert split_qualifier("ibriks (coffee makers)") == ("ibriks", "coffee makers")
    assert split_qualifier("Hellenistic") == ("Hellenistic", None)


def test_wordnet_form_ok():
    assert wordnet_form_ok("zebra")
    assert wordnet_form_ok("air hammer")
    assert not wordnet_form_ok("Aberdeen")        # proper noun
    assert not wordnet_form_ok(".22")             # leading punctuation
    assert not wordnet_form_ok("3d radar")        # leading digit
    assert not wordnet_form_ok("a battery")       # leading article
    assert not wordnet_form_ok("x")               # too short


def test_aat_style_ok():
    assert aat_style_ok("Jun ware", "Chinese ceramics style")
    assert aat_style_ok("Edo", "Japanese period")
    assert aat_style_ok("Red-figure", None)
    assert aat_style_ok("Collegiate Gothic", None)
    assert not aat_style_ok("Arapaho", "culture or style")
    assert not aat_style_ok("Nevadan", "of modern U.S. state")
    assert not aat_style_ok("Bruttii", None)


def test_merge_deduplicates_and_keeps_curated_zh():
    a = Concept(en="hanfu", axis="entity", sub="garment_ea",
                sources=["curated"], zh="汉服", curated=True, freq_en=1.2)
    b = Concept(en="hanfu", axis="entity", sub="artifact",
                sources=["wordnet"], freq_en=2.9)
    (m,) = merge([b, a])
    assert m.curated and m.zh == "汉服"
    assert m.sources == ["curated", "wordnet"]
    assert m.freq_en == 2.9


def test_merge_drops_blocklist():
    out = merge([Concept(en="cunt", axis="entity", sub="person_role",
                         sources=["wordnet"], freq_en=3.0)])
    assert out == []


def test_singularize_last():
    assert singularize_last("churches") == "church"
    assert singularize_last("fireflies") == "firefly"
    assert singularize_last("kimonos") == "kimono"
    assert singularize_last("boxes") == "box"
    assert singularize_last("glass") == "glass"          # -ss untouched
    assert singularize_last("walrus") == "walrus"        # -us untouched
    assert singularize_last("iris") == "iris"            # -is untouched
    assert singularize_last("kimono sleeve") == "kimono sleeve"  # singular kept


def test_merge_merges_plural_across_sources():
    a = Concept(en="church", axis="scene", sub="location",
                sources=["wordnet"], freq_en=4.0, aliases=["kirk"])
    b = Concept(en="churches", axis="scene", sub="settlement_landscape",
                sources=["aat"], freq_en=4.5, qualifier="buildings")
    (m,) = merge([b, a])
    assert m.en == "churches"          # first writer keeps the label
    assert m.sources == ["aat", "wordnet"]
    assert "church" in m.aliases and "kirk" in m.aliases
    assert m.freq_en == 4.5 and m.qualifier == "buildings"


def test_merge_keeps_axes_separate():
    a = Concept(en="brush", axis="entity", sub="artifact", sources=["wordnet"])
    b = Concept(en="brush", axis="technique", sub="process", sources=["aat"])
    assert len(merge([a, b])) == 2
