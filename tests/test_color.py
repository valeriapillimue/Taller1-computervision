
import unittest
import numpy as np
import cv2
from cvtools.color import rgb_to_hsv, rgb_to_lab, color_histogram, quantize_image, reduce_image_size_by_color


class TestColor(unittest.TestCase):

    def setUp(self):
        # Imagen RGB aleatoria de prueba (64x64)
        self.img = (np.random.rand(64,64,3) * 255).astype(np.uint8)

    def test_rgb_to_hsv(self):
        hsv = rgb_to_hsv(self.img)
        self.assertEqual(hsv.shape, self.img.shape)

    def test_rgb_to_lab(self):
        lab = rgb_to_lab(self.img)
        self.assertEqual(lab.shape, self.img.shape)

    def test_color_histogram(self):
        hist = color_histogram(self.img, bins=32, show=False)
        self.assertIn("r", hist)
        self.assertIn("g", hist)
        self.assertIn("b", hist)
        self.assertEqual(len(hist["r"]), 32)

    def test_quantize_image(self):
        qimg = quantize_image(self.img, 64)
        self.assertEqual(qimg.shape, self.img.shape)
        self.assertEqual(qimg.dtype, np.uint8)

    def test_reduce_image_size_by_color(self):
        qimg, size_kb = reduce_image_size_by_color(self.img, 32)
        self.assertEqual(qimg.shape, self.img.shape)
        self.assertTrue(size_kb > 0)


if __name__ == "__main__":
    unittest.main()
