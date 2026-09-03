"""Offline tests for the museum scan cleaning pipeline (synthetic arrays)."""

import unittest

import numpy as np

from src.dataset.clean import (clean_image, cut_chart_side, detect_chart,
                               detect_mount_border, view_tag)


def make_painting(w=200, h=300, border=20, border_color=(235, 228, 210), seed=0):
    """Low-saturation 'painting' rect surrounded by a uniform mounting border."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = border_color
    arr[border:h - border, border:w - border] = muted_noise(rng, (h - 2 * border, w - 2 * border))
    return arr


def muted_noise(rng, shape):
    """Grayish noise with small channel jitter — mimics low-saturation ink painting."""
    gray = rng.integers(60, 180, (*shape, 1), dtype=np.int16)
    jitter = rng.integers(-15, 15, (*shape, 3), dtype=np.int16)
    return np.clip(gray + jitter, 0, 255).astype(np.uint8)


def make_doc_photo(w=800, h=600):
    """Black background, painting rect, and a vivid color-chart strip on the left.

    Proportions mimic real NPM archival shots: chart ≈1% of the frame."""
    rng = np.random.default_rng(1)
    arr = np.zeros((h, w, 3), dtype=np.uint8) + 15  # near-black bg
    arr[80:h - 80, 240:w - 120] = muted_noise(rng, (h - 160, w - 360))
    # color chart: grid of saturated patches on the left
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, c in enumerate(colors):
        arr[60 + i * 40:88 + i * 40, 30:70] = c
    return arr


def make_mounted_scroll(w=960, h=720):
    """Full-mount archival shot: black bg, scroll with mount+painting, chart at right."""
    rng = np.random.default_rng(4)
    arr = np.zeros((h, w, 3), dtype=np.uint8) + 15
    arr[20:700, 60:460] = (235, 228, 210)  # cream mount
    arr[80:640, 120:400] = muted_noise(rng, (560, 280))  # painting
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, c in enumerate(colors):  # chart strip at the right edge
        arr[80 + i * 50:110 + i * 50, 880:925] = c
    return arr


class TestDetectChart(unittest.TestCase):
    def test_chart_found_in_doc_photo(self):
        arr = make_doc_photo()
        chart = detect_chart(arr)
        self.assertIsNotNone(chart)
        l, t, r, b = chart
        self.assertLess(l, 60)  # chart is at the left edge

    def test_no_chart_without_black_bg(self):
        self.assertIsNone(detect_chart(make_painting()))

    def test_dark_painting_no_chart(self):
        dark = np.random.default_rng(2).integers(45, 80, (300, 200, 3), dtype=np.uint8)
        self.assertIsNone(detect_chart(dark))

    def test_cut_chart_side(self):
        arr = make_doc_photo()
        chart = detect_chart(arr)
        cut = cut_chart_side(arr, chart)
        self.assertIsNotNone(cut)
        self.assertGreater(cut.shape[1], 300)  # keeps the painting side
        self.assertEqual(cut.shape[0], arr.shape[0])

    def test_central_chart_cannot_be_cut(self):
        arr = make_doc_photo()
        h, w = arr.shape[:2]
        self.assertIsNone(cut_chart_side(arr, (int(w * 0.4), int(h * 0.4), int(w * 0.6), int(h * 0.6))))


class TestDetectMountBorder(unittest.TestCase):
    def test_uniform_border_cropped(self):
        arr = make_painting(w=200, h=300, border=20)
        l, t, r, b = detect_mount_border(arr)
        self.assertEqual((l, t, r, b), (20, 20, 180, 280))

    def test_asymmetric_border(self):
        rng = np.random.default_rng(3)
        arr = np.zeros((300, 200, 3), dtype=np.uint8)
        arr[50:280, 30:170] = muted_noise(rng, (230, 140))
        l, t, r, b = detect_mount_border(arr)
        self.assertEqual((l, t, r, b), (30, 50, 170, 280))

    def test_degenerate_refuses_crop(self):
        # fully uniform image: content would be empty, refuse to crop
        arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        self.assertEqual(detect_mount_border(arr), (0, 0, 100, 100))


class TestCleanImage(unittest.TestCase):
    def test_mounted_scroll_recovered_not_rejected(self, tmp_path=None):
        import os
        import tempfile
        from PIL import Image
        arr = make_mounted_scroll()
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.jpg")
            out = os.path.join(d, "out.jpg")
            Image.fromarray(arr).save(src)
            rec = clean_image(src, out)
            self.assertFalse(rec["rejected"])
            self.assertIn("chart_bbox", rec)
            # result should be roughly the scroll region (mount + painting)
            self.assertGreater(rec["width"], 150)
            self.assertLess(rec["width"], 300)

    def test_mostly_black_rejected(self):
        import os
        import tempfile
        from PIL import Image
        arr = np.zeros((300, 400, 3), dtype=np.uint8) + 10
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.jpg")
            Image.fromarray(arr).save(src)
            rec = clean_image(src, os.path.join(d, "out.jpg"))
            self.assertTrue(rec["rejected"])
            self.assertEqual(rec["reject_reason"], "mostly_black")


class TestViewTag(unittest.TestCase):
    def test_suffix_parsing(self):
        self.assertEqual(view_tag("K2A000001N000000000PAA"), 0)
        self.assertEqual(view_tag("K2A000001N000000000PAZ"), 25)
        self.assertEqual(view_tag(""), -1)
        self.assertEqual(view_tag("random"), -1)


if __name__ == "__main__":
    unittest.main()
