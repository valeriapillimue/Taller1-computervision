import unittest
import numpy as np
import cv2
from cvtools.filters import convolve2d, sobel_x, sobel_y, canny, laplacian


class TestFilters(unittest.TestCase):

    def setUp(self):
        # Imagen gris aleatoria de prueba (64x64)
        self.img_gray = (np.random.rand(64,64) * 255).astype(np.uint8)

    def test_convolve2d(self):
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        out = convolve2d(self.img_gray, kernel)
        self.assertEqual(out.shape, self.img_gray.shape)

    def test_sobel_x(self):
        out = sobel_x(self.img_gray)
        self.assertEqual(out.shape, self.img_gray.shape)

    def test_sobel_y(self):
        out = sobel_y(self.img_gray)
        self.assertEqual(out.shape, self.img_gray.shape)

    def test_canny(self):
        edges = canny(self.img_gray, 50, 150)
        self.assertEqual(edges.shape, self.img_gray.shape)
        # Debe ser binaria (0 o 255 en OpenCV)
        self.assertTrue(np.all(np.isin(np.unique(edges), [0, 255])))

    def test_laplacian(self):
        out = laplacian(self.img_gray)
        self.assertEqual(out.shape, self.img_gray.shape)


if __name__ == "__main__":
    unittest.main()
