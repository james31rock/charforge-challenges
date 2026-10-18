import unittest
import numpy as np
from toolkit.rig.math_core import two_bone, knee_plane, cyclic_lowpass, shoulder_counterrotation, arm_swing, lock_contact, corrective_profile, frame

class MotionMath(unittest.TestCase):

    def test_nonfinite_physical_parameters_fail_before_output(self):
        for invalid in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError):
                shoulder_counterrotation([0, 0.1], 1, 0.2, invalid)
            with self.assertRaises(ValueError):
                arm_swing(0, 1, 0.5, 0.4, invalid)
            with self.assertRaises(ValueError):
                corrective_profile(invalid, 0.4, 0.3)

    def test_sphere_intersection_lengths_and_scale(self):
        for scale in [0.6, 1.05, 1.4, 1.85, 3.0]:
            result = two_bone(np.array([0, 0, 0.8]) * scale, np.zeros(3), 0.45 * scale, 0.44 * scale, [0, -1, 0])
            self.assertTrue(result.reachable)
            self.assertAlmostEqual(np.linalg.norm(result.knee - [0, 0, 0.8 * scale]), 0.45 * scale)
            self.assertAlmostEqual(np.linalg.norm(result.knee - result.ankle), 0.44 * scale)

    def test_straight_knee_no_noisy_pole(self):
        for x in [-1e-10, 0, 1e-10]:
            r = two_bone([0, 0, 1], [x, 0, 0], 0.5, 0.5, [0, -1, 0])
            self.assertLess(abs(r.knee[0]), 1e-08)
            self.assertLessEqual(r.knee[1], 0)

    def test_unreachable_contact_is_failure(self):
        r, error = lock_contact([0, 0, 2], [0, 0, 0], [0, -0.1, -0.1], np.eye(3), [0, 0, 0], 0.4, 0.4, [0, -1, 0])
        self.assertFalse(r.reachable)
        self.assertGreater(error, 1.0)

    def test_contact_and_lengths_preserved(self):
        r, error = lock_contact([0, 0, 0.7], [0, 0, 0.1], [0, -0.1, -0.1], np.eye(3), [0, -0.1, 0], 0.4, 0.4, [0, -1, 0])
        self.assertTrue(r.reachable)
        self.assertLess(error, 1e-09)

    def test_no_height_specific_motion(self):
        yaw = np.sin(np.arange(32) * 2 * np.pi / 32) * 0.1
        base = shoulder_counterrotation(yaw, 0.7, 0.2, 0.4)
        for s in [0.75, 1, 2.6]:
            np.testing.assert_allclose(base, shoulder_counterrotation(yaw, 0.7 * s, 0.2 * s, 0.4 * s))
            for key, value in arm_swing(1, 0.7, 0.5, 0.4, 0.2).items():
                self.assertAlmostEqual(value, arm_swing(1, 0.7 * s, 0.5 * s, 0.4 * s, 0.2 * s)[key])
            a = corrective_profile(0.05 * s, 0.4 * s, 0.3 * s)
            self.assertAlmostEqual(a['start_angle'], corrective_profile(0.05, 0.4, 0.3)['start_angle'])

    def test_variable_cadence(self):
        for n in [16, 24, 37]:
            t = np.arange(n) * 2 * np.pi / n
            x = np.tile(np.sin(t) + 0.3 * np.sin(4 * t), 2)
            np.testing.assert_allclose(cyclic_lowpass(x, n), np.tile(np.sin(t), 2), atol=1e-12)

    def test_frame_and_ik_rotation_equivariance(self):
        g = frame([0, 1, 0], [0, 0, 1])
        self.assertAlmostEqual(np.linalg.det(g), 1.0)
        h = np.array([0, 0, 0.7])
        a = np.zeros(3)
        f = np.array([0, -1, 0])
        l = np.array([1, 0, 0])
        r = two_bone(h, a, 0.4, 0.4, f, 0.02, l)
        q = two_bone(g @ h, g @ a, 0.4, 0.4, g @ f, 0.02, g @ l)
        np.testing.assert_allclose(g @ r.knee, q.knee)

    def test_attenuation_preserves_lengths(self):
        h = np.array([0.1, 0, 0.8])
        a = np.array([0.1, 0, 0])
        k = np.array([0.13, -0.15, 0.4])
        r = knee_plane(h, k, a, [0, -1, 0], [1, 0, 0])
        self.assertAlmostEqual(r.knee[0] - 0.1, 0.25 * (k[0] - 0.1))
        self.assertAlmostEqual(np.linalg.norm(r.knee - h), np.linalg.norm(k - h))

    def test_bad_inputs_rejected(self):
        for args in [([0, 0, 0], [0, 0, 0], 1, 1, [0, 1, 0]), ([0, 0, 1], [0, 0, 0], -1, 1, [0, 1, 0])]:
            with self.assertRaises(ValueError):
                two_bone(*args)
        with self.assertRaises(ValueError):
            cyclic_lowpass([1, 2, 3, 4], 3)
        with self.assertRaises(ValueError):
            frame([0, 0, 1], [0, 0, 1])

    def test_distinct_body_proportions(self):
        for upper, lower, radius, hipwidth, armlength in [(0.22, 0.21, 0.035, 0.14, 0.3), (0.31, 0.28, 0.04, 0.18, 0.4), (0.46, 0.43, 0.13, 0.42, 0.67), (0.1, 0.09, 0.06, 0.22, 0.16)]:
            length = upper + lower
            r = two_bone([hipwidth / 2, 0, 0.9 * length], [hipwidth / 2, 0, 0], upper, lower, [0, -1, 0])
            self.assertTrue(r.reachable)
            self.assertAlmostEqual(np.linalg.norm(r.knee - [hipwidth / 2, 0, 0.9 * length]), upper)
            profile = corrective_profile(radius, upper, lower)
            self.assertGreater(profile['support'], 0)
            self.assertLessEqual(profile['support'], 0.4 * min(upper, lower))
            self.assertTrue(np.isfinite(list(arm_swing(0.5, length, armlength, 0.5 * length, hipwidth).values())).all())
if __name__ == '__main__':
    unittest.main()
