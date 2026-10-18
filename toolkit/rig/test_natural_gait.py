import unittest
import copy
import numpy as np
from toolkit.rig.natural_gait import circular_phase_error, swing_floor

class NaturalGaitTests(unittest.TestCase):

    def test_phase_ignores_amplitude_and_dc_not_timing(self):
        x = 2 * np.pi * np.arange(48) / 48
        a = np.sin(x)
        self.assertAlmostEqual(circular_phase_error(8 * a + 40, a), 0.0)
        self.assertAlmostEqual(circular_phase_error(np.sin(x + 0.3), a), 0.3)
        self.assertAlmostEqual(circular_phase_error(np.roll(a, 7), np.roll(3 * a + 1, 7)), 0.0)
        with self.assertRaises(ValueError):
            circular_phase_error(np.ones(48), a)

    def test_clearance_scales_and_closes_without_a_last_frame_step(self):
        rest = {s + p: [sign * 0.1, 0, z] for s, sign in [('left', 1), ('right', -1)] for p, z in [('UpperLeg', 1.0), ('LowerLeg', 0.6), ('Foot', 0.2)]}
        ref = dict(rest=rest, samples=[dict(phase=i / 24, feet={s: dict(min_z_m=0.04 * np.sin(np.pi * ((i / 24 - (0 if s == 'left' else 0.5)) % 1 - 0.6) / 0.4) ** 2 if (i / 24 - (0 if s == 'left' else 0.5)) % 1 > 0.6 else 0.0) for s in ['left', 'right']}) for i in range(24)])
        for phase in np.linspace(0.6, 1.0, 101):
            a = swing_floor(ref, 'left', phase, 0.4) - 0.0005
            b = swing_floor(ref, 'left', phase, 1.2) - 0.0005
            self.assertAlmostEqual(b, 3 * a)
            self.assertGreaterEqual(a, 0.0)
        self.assertEqual(swing_floor(ref, 'left', 0.4, 0.4), 0.0005)
        self.assertLess(swing_floor(ref, 'left', 1 - 1e-05, 0.4) - 0.0005, 1e-08)
        self.assertLess(swing_floor(ref, 'left', 0.6 + 1e-05, 0.4) - 0.0005, 1e-08)
        translated = copy.deepcopy(ref)
        translated['floor_m'] = 0.7
        for point in translated['rest'].values():
            point[2] += 0.7
        for row in translated['samples']:
            for foot in row['feet'].values():
                foot['min_z_m'] += 0.7
        for phase in np.linspace(0.6, 1.0, 101):
            self.assertAlmostEqual(swing_floor(ref, 'left', phase, 0.4), swing_floor(translated, 'left', phase, 0.4))
if __name__ == '__main__':
    unittest.main()
