"""Offline tests for domain_filter and dedup."""

import unittest

from src.dataset.dedup import find_cross_canvas_dupes, norm_text
from src.dataset.domain_filter import assign_domain


class TestDomainFilter(unittest.TestCase):
    def test_npm_tw_always_guohua(self):
        self.assertEqual(assign_domain("npm_tw", {}), "guo_hua")

    def test_met_china_painting(self):
        rec = {"culture": "China", "classification": "Paintings", "department": "Asian Art"}
        self.assertEqual(assign_domain("met", rec), "guo_hua")

    def test_met_european(self):
        rec = {"culture": "", "classification": "Paintings", "department": "European Paintings"}
        self.assertEqual(assign_domain("met", rec), "western")

    def test_met_japan_print(self):
        rec = {"culture": "Japan", "classification": "Prints", "department": "Asian Art"}
        self.assertEqual(assign_domain("met", rec), "japanese_print")

    def test_met_ceramics_object(self):
        rec = {"culture": "China", "classification": "Ceramics", "department": "Asian Art"}
        self.assertEqual(assign_domain("met", rec), "object")

    def test_aic_china_painting(self):
        rec = {"style_title": "Chinese (culture or style)", "classification": "hanging scroll"}
        self.assertEqual(assign_domain("aic", rec), "guo_hua")

    def test_aic_chinese_object(self):
        rec = {"style_title": "Chinese (culture or style)", "classification": "drinking vessel"}
        self.assertEqual(assign_domain("aic", rec), "object")

    def test_aic_japanese(self):
        rec = {"style_title": "Japanese (culture or style)", "classification": "painting"}
        self.assertEqual(assign_domain("aic", rec), "japanese_print")

    def test_aic_unknown_culture_painting(self):
        rec = {"style_title": "", "period": "", "classification": "album leaf"}
        self.assertEqual(assign_domain("aic", rec), "other")

    def test_aic_none_classification(self):
        rec = {"style_title": "Chinese (culture or style)", "classification": None}
        self.assertEqual(assign_domain("aic", rec), "guo_hua")

    def test_princeton_asian_painting(self):
        rec = {"classification": "Paintings", "department": "Asian Art"}
        self.assertEqual(assign_domain("princeton", rec), "guo_hua")

    def test_princeton_calligraphy(self):
        rec = {"classification": "Calligraphy", "department": "Asian Art"}
        self.assertEqual(assign_domain("princeton", rec), "guo_hua")

    def test_nga_painting_western(self):
        self.assertEqual(assign_domain("nga", {"classification": "Painting"}), "western")
        self.assertEqual(assign_domain("nga", {"classification": "Index of American Design"}), "other")

    def test_fsg_chinese_painting(self):
        rec = {"culture": "Chinese", "classification": "Painting"}
        self.assertEqual(assign_domain("fsg", rec), "guo_hua")


class TestCrossCanvasDedup(unittest.TestCase):
    def test_groups_same_title_artist(self):
        recs = [
            {"image_id": "a", "canvas_label": "K2A000001N000000000PAA",
             "title": "山水　軸", "artist": "佚名"},
            {"image_id": "b", "canvas_label": "K2A000099N000000000PAA",
             "title": "山水 軸", "artist": "佚名"},  # whitespace-normalized dupe
            {"image_id": "c", "canvas_label": "K2A000001N000000000PAB",
             "title": "山水　軸", "artist": "佚名"},  # same canvas, not cross-canvas
            {"image_id": "d", "canvas_label": "K2A000002N000000000PAA",
             "title": "花鸟", "artist": "佚名"},
        ]
        groups = find_cross_canvas_dupes(recs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[("山水軸", "佚名")],
                         ["K2A000001N000000000", "K2A000099N000000000"])

    def test_norm_text(self):
        self.assertEqual(norm_text("a　 b  c"), "abc")
        self.assertEqual(norm_text(None), "")


if __name__ == "__main__":
    unittest.main()
