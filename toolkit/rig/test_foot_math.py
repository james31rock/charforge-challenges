import unittest
import numpy as np
from toolkit.rig.foot_math import pivoted_toe, shoe_features, bind_shoes

class FootMathTests(unittest.TestCase):

    def test_ball_pivot_is_fixed_under_independent_toe_rotation(self):
        a = np.array([0.1, 0, 0.12])
        t = np.array([0.1, -0.15, 0.03])
        ball = np.array([0.1, -0.12, 0])
        angle = 0.6
        r = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        identity = np.eye(3)
        for scale in [0.7, 1, 1.85]:
            toe = pivoted_toe(a * scale, a * scale, t * scale, ball * scale, r, identity)
            fixed = toe + identity @ (ball * scale - t * scale)
            np.testing.assert_allclose(fixed, a * scale + r @ (ball * scale - a * scale), atol=1e-12)
            broken = a * scale + r @ (t * scale - a * scale) + identity @ (ball * scale - t * scale)
            self.assertGreater(np.linalg.norm(broken - fixed), 0.005 * scale)

    def test_shoe_feature_is_scale_equivariant(self):
        v = np.array([[x, y, z] for x in [-0.04, 0.04] for y in np.linspace(-0.2, 0.05, 31) for z in [0, 0.04]])
        a = np.array([0, 0, 0.12])
        t = np.array([0, -0.15, 0.03])
        first = shoe_features(v, a, t)
        for scale in [0.7, 1.85]:
            f = shoe_features(v * scale, a * scale, t * scale)
            np.testing.assert_allclose(f['ball'] / scale, first['ball'])
            self.assertAlmostEqual(f['length'] / scale, first['length'])
if __name__ == '__main__':
    unittest.main()
