import unittest
import numpy as np
from toolkit.rig.gait_planning import swing_clearance_offsets

class SwingClearanceTests(unittest.TestCase):

    def test_scale_and_fixed_support(self):
        n = 120
        phase = np.arange(n) / n
        needed = np.zeros(n)
        needed[(phase > 0.23) & (phase < 0.29)] = 0.03
        needed[(phase > 0.74) & (phase < 0.79)] = 0.04
        a, report = swing_clearance_offsets(needed, 0.5)
        b, _ = swing_clearance_offsets(needed * 3, 1.5)
        np.testing.assert_allclose(b, a * 3, atol=1e-14)
        self.assertLessEqual(report['bound_violation_m'], 1e-14)
        for side in range(2):
            stance = (phase - side * 0.5) % 1 < 0.6
            np.testing.assert_array_equal(a[stance, side], np.zeros(stance.sum()))

    def test_double_support_bound_cannot_be_hidden_in_swing(self):
        needed = np.zeros(120)
        needed[0] = 0.02
        needed[30] = 0.04
        offset, report = swing_clearance_offsets(needed, 0.5)
        self.assertAlmostEqual(report['stance_extra_separation_m'], 0.02)
        self.assertTrue(np.all(offset.sum(1) >= needed - 1e-12))
        np.testing.assert_allclose(offset[0], [0.01, 0.01])
if __name__ == '__main__':
    unittest.main()
