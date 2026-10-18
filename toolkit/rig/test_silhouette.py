import unittest
import numpy as np
from toolkit.rig.silhouette import envelope, branch_distance

class SilhouetteTests(unittest.TestCase):

    def test_visible_gap_is_not_crotch(self):
        a = np.zeros((1000, 600), np.uint8)
        a[50:700, 200:400] = 255
        a[650:950, 210:270] = 255
        a[650:950, 330:390] = 255
        a[400:650, 110:170] = 255
        a[400:650, 430:490] = 255
        e = envelope(a)
        self.assertAlmostEqual(e['visible_leg_gap_top'], 250 / 900, delta=0.02)
        self.assertIsNotNone(e['visible_arm_separation_z'])
        self.assertEqual(branch_distance(e, 0.5, 0.2, 'arm'), 0)
        self.assertGreater(branch_distance(e, 0.5, 0.02, 'arm'), 0.1)
        self.assertIsNone(branch_distance(e, 0.1, 0.1, 'arm'))

    def test_scale_invariant(self):
        a = np.zeros((100, 60), np.uint8)
        a[5:65, 20:40] = 255
        a[60:95, 21:27] = 255
        a[60:95, 33:39] = 255
        x = envelope(a)
        y = envelope(np.repeat(np.repeat(a, 3, 0), 3, 1))
        self.assertAlmostEqual(x['visible_leg_gap_top'], y['visible_leg_gap_top'], delta=0.012)
if __name__ == '__main__':
    unittest.main()
