"""Image cleaning checked against synthetic scenes with known content bounds."""

import unittest

import numpy as np

from src.dataset.clean import clean_image, detect_mount_border


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
