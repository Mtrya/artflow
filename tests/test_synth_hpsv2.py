"""Offline tests for the synthetic batch's data-side helpers (no API calls)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.data.build_synth_grid import frames_for, pick_aspect, stable_hash
from scripts.data.expand_hpsv2_prompts import (
    AXES, BAND_SHARES, assign_contract, near_duplicate, normalize_caption,
    parse_variants, plan_for, validate,
)
from scripts.data.qa_synth_batch import needs_caption


class FramesTest(unittest.TestCase):
    def test_every_frame_is_a_multiple_of_32(self):
        # The pipeline silently resizes anything else, which would leave the
        # row's recorded size describing a file that does not exist.
        for area in (1048576, 1536 * 1024):
            for name, (width, height) in frames_for(area).items():
                self.assertEqual(width % 32, 0, name)
                self.assertEqual(height % 32, 0, name)

    def test_every_ratio_keeps_its_shape_and_budget(self):
        frames = frames_for(1048576)
        for name, (width, height) in frames.items():
            self.assertLess(abs(width * height - 1048576), 0.1 * 1048576, name)
            self.assertLess(abs(width / height - frames[name][0] / frames[name][1]), 0.1, name)
        self.assertEqual(frames["1:1"], (1024, 1024))

    def test_aspect_draw_is_deterministic_and_covers_every_ratio(self):
        self.assertEqual(pick_aspect("syn21-000001"), pick_aspect("syn21-000001"))
        drawn = {pick_aspect(f"syn21-{index:06d}") for index in range(500)}
        self.assertEqual(drawn, set(frames_for(1048576)))

    def test_stable_hash_is_uniform_enough(self):
        draws = [stable_hash(f"key-{index}", 42) for index in range(2000)]
        self.assertTrue(all(0.0 <= value < 1.0 for value in draws))
        low = sum(1 for value in draws if value < 0.5)
        self.assertLess(abs(low / len(draws) - 0.5), 0.05)


class ContractTest(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        first, second = assign_contract("hps-0000-0"), assign_contract("hps-0000-0")
        self.assertEqual((first.language, first.format, first.band),
                         (second.language, second.format, second.band))

    def test_assignment_matches_the_caption_contract_cells(self):
        cells = [assign_contract(f"hps-{index:04d}-{axis}") for index in range(400)
                 for axis in range(len(AXES))]
        bands = {name: sum(1 for cell in cells if cell.band == name) for name, _ in BAND_SHARES}
        for name, share in BAND_SHARES:
            self.assertLess(abs(bands[name] / len(cells) - share), 0.06, name)
        chinese = sum(1 for cell in cells if cell.language == "zh")
        self.assertLess(abs(chinese / len(cells) - 0.5), 0.06)
        structured = sum(1 for cell in cells if cell.format == "structured")
        self.assertLess(abs(structured / len(cells) - 0.5), 0.06)

    def test_plan_has_one_request_per_axis(self):
        plan = plan_for("hps-0001")
        self.assertEqual([job["axis"] for job in plan], list(AXES))
        self.assertEqual(len({job["request"].band for job in plan}), len(set(
            job["request"].band for job in plan)))


class ValidationTest(unittest.TestCase):
    def request(self, language="en"):
        request = assign_contract("validation-0")
        request.language = language
        return request

    def test_plain_description_is_accepted(self):
        text = ("A photograph of a young woman in a red silk hanfu standing on "
                "the stone steps of a temple gate, morning side light, film grain.")
        self.assertEqual(validate(text, self.request(), None), [])

    def test_hedging_evaluation_and_attribution_are_rejected(self):
        request = self.request()
        for text in ("A photograph of a woman who appears to be in a red dress.",
                     "An oil painting, beautifully composed, of a harbour at dusk.",
                     "A pencil drawing of a fox by Alice, on cartridge paper.",
                     "A photograph of an anime scene posted on r/streetwear."):
            self.assertTrue(validate(text, request, None), text)

    def test_language_is_enforced(self):
        self.assertTrue(validate("画面中央是一只动物。" * 4, self.request("en"), None))
        self.assertTrue(validate("A photograph of a stone bridge over a river.",
                                 self.request("zh"), None))

    def test_a_strippable_opening_is_stripped_rather_than_rejected(self):
        request = self.request()
        text = "This image shows a photograph of a stone bridge over a river."
        self.assertEqual(validate(text, request, None), [])
        self.assertTrue(normalize_caption(text).startswith("A photograph"))

    def test_an_opening_that_cannot_be_stripped_is_rejected(self):
        self.assertTrue(validate("This image shows a cat.", self.request(), None))

    def test_near_duplicate_detection(self):
        first = ("A photograph of a young woman in a red silk hanfu standing on "
                 "the stone steps of a temple gate in morning light")
        self.assertTrue(near_duplicate(first, [first]))
        self.assertFalse(near_duplicate(first, ["An oil painting of a harbour at dusk"]))


class ParseTest(unittest.TestCase):
    def test_raw_newline_inside_a_string_is_tolerated(self):
        text = ('{"variants": [{"axis": "setting", "text": "a line\nand another"}]}')
        parsed = parse_variants(text)
        self.assertEqual(parsed[0]["axis"], "setting")
        self.assertIn("and another", parsed[0]["text"])

    def test_missing_json_raises(self):
        with self.assertRaises(ValueError):
            parse_variants("no object here")


class CaptionDecisionTest(unittest.TestCase):
    def test_only_mismatches_get_a_caption(self):
        self.assertFalse(needs_caption({"verdict": "full"}))
        self.assertFalse(needs_caption({"verdict": "mostly"}))
        self.assertTrue(needs_caption({"verdict": "partial"}))
        self.assertTrue(needs_caption({"verdict": "none"}))
        self.assertTrue(needs_caption({"verdict": "error"}))
        self.assertTrue(needs_caption({"verdict": "mostly", "contradicted": ["a red dress"]}))


class NormalizeTest(unittest.TestCase):
    def test_catalogue_opening_is_stripped(self):
        self.assertEqual(normalize_caption("This image shows a stone bridge."),
                         "A stone bridge.")


class StageWiringTest(unittest.TestCase):
    def test_async_stages_are_awaited(self):
        """The check stage is a coroutine; a bare call would silently do nothing."""
        from scripts.data import qa_synth_batch as module

        called = []

        async def fake_check(args, rows):
            called.append(len(rows))
            return []

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["qa_synth_batch", "--stage", "check", "--generated", "gen",
                    "--out-dir", tmp]
            with mock.patch.object(module, "stage_check", fake_check), \
                    mock.patch.object(module, "load_rows", lambda grid, gen: [{"prompt_id": "x"}]), \
                    mock.patch.object(sys, "argv", argv):
                module.main()
            self.assertEqual(called, [1])

            # A later stage without its input is a mistake, not an empty run.
            argv = ["qa_synth_batch", "--stage", "assemble", "--generated", "gen",
                    "--out-dir", tmp]
            with mock.patch.object(module, "load_rows", lambda grid, gen: [{"prompt_id": "x"}]), \
                    mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    module.main()


if __name__ == "__main__":
    unittest.main()
