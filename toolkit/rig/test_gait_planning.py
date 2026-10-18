import unittest
import numpy as np
from toolkit.rig.gait_planning import constrained_quadratic, plan_root, shoe_spacing, clipped_x_extents

class PlannedGaitTests(unittest.TestCase):

    def test_triangle_support_includes_edges_crossing_empty_vertex_slice(self):
        tri = np.array([[[0.0, -1, 0], [2, 1, 0], [0, -1, 2]]])
        np.testing.assert_allclose(clipped_x_extents(tri, [-0.1, 0.2], [0.1, 0.4]), [0.9, 1.1], atol=1e-09)

    def test_qp_respects_active_reach_bound(self):
        x, _ = constrained_quadratic(np.eye(2), np.array([2.0, 3.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 1.0]), np.zeros(2))
        np.testing.assert_allclose(x, [1, 1], atol=1e-08)

    def test_root_is_scale_equivariant_periodic_and_bounded(self):
        n = 24
        p = np.arange(n) / n
        hips = np.zeros((n, 2, 3))
        hips[:, :, 0] = [0.1, -0.1]
        ankles = np.zeros_like(hips)
        ankles[:, :, 0] = [0.08, -0.08]
        ankles[:, :, 1] = np.column_stack([0.12 * np.cos(2 * np.pi * p), -0.12 * np.cos(2 * np.pi * p)])
        lengths = np.ones((2, 2)) * 0.5
        flex = np.ones((n, 2)) * 0.25
        stance = np.column_stack([p < 0.6, (p + 0.5) % 1 < 0.6])
        source = 0.02 * np.cos(4 * np.pi * p)
        z, report = plan_root(source, hips, ankles, lengths, flex, stance, 1.0)
        z2, _ = plan_root(source * 7, hips * 7, ankles * 7, lengths * 7, flex, stance, 7.0)
        np.testing.assert_allclose(z, z2 / 7, atol=1e-08)
        self.assertLessEqual(report['maximum_reach_bound_violation_m'], 1e-08)
        self.assertLessEqual(np.ptp(z), np.ptp(source) + 1e-08)
        self.assertLess(np.linalg.norm(np.fft.rfft(z)[4:]), 1e-08)

    def test_shoe_bound_uses_vertical_separation(self):
        box = np.array([[x, y, z] for x in [-0.05, 0.05] for y in [-0.1, 0.1] for z in [0.0, 0.05]])
        shoes = [[box + np.array([0.04, 0, 0]), box - np.array([0.04, 0, 0])]]
        ankles = np.array([[[0.04, 0, 0], [-0.04, 0, 0]]])
        centre = np.zeros((1, 3))
        rel = ankles[:, :, 0]
        x, report = shoe_spacing(shoes, ankles, centre, rel, 1.0)
        self.assertAlmostEqual(x[0, 0] - x[0, 1], 0.11)
        shoes[0][0][:, 2] += 0.2
        _, report = shoe_spacing(shoes, ankles, centre, rel, 1.0)
        self.assertEqual(report['extra_separation_m'], 0)
