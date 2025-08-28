import unittest
import numpy as np

from cvtools.camera import apply_radial_distortion, project_points_pinhole 


class TestCamera(unittest.TestCase):

    def test_apply_radial_distortion_center_point(self):
        """Un punto en el centro (0,0) debe permanecer igual tras la distorsión."""
        pts = np.array([[0.0, 0.0]])
        result = apply_radial_distortion(pts, k1=0.1, k2=0.01)
        np.testing.assert_array_almost_equal(result, pts)

    def test_apply_radial_distortion_nonzero_point(self):
        """Un punto fuera del centro debe escalarse con el factor de distorsión."""
        pts = np.array([[0.5, 0.0]])  # punto en X
        result = apply_radial_distortion(pts, k1=0.1, k2=0.01)
        # factor esperado: 1 + k1*r^2 + k2*r^4
        r2 = 0.5**2
        factor = 1 + 0.1*r2 + 0.01*(r2**2)
        expected = np.array([[0.5 * factor, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_project_points_pinhole_simple(self):
        """Probar proyección pinhole con un punto sencillo."""
        pts3d = np.array([[2.0, 4.0, 2.0]])  # X=2,Y=4,Z=2
        f = 100
        result = project_points_pinhole(pts3d, f)
        expected = np.array([[f * (2.0/2.0), f * (4.0/2.0)]])  # (100, 200)
        np.testing.assert_array_almost_equal(result, expected)

    def test_project_points_pinhole_invalid_Z(self):
        """Si Z <= 0, debe lanzar error."""
        pts3d = np.array([[1.0, 2.0, -1.0]])
        with self.assertRaises(ValueError):
            project_points_pinhole(pts3d, f=100)


if __name__ == "__main__":
    unittest.main()
