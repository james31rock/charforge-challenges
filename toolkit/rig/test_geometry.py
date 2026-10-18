import unittest
import numpy as np
from toolkit.rig.geometry import section, section_loops, arm_section_tracks, depth_interval, arm_region_masks, interior_arm_seed
from toolkit.rig.weights import diffuse

def cylinder(cx, cy, r, z0=0, z1=1, n=32):
    phi = np.arange(n) * 2 * np.pi / n
    v = np.array([[cx + r * np.cos(p), cy + r * np.sin(p), z] for z in [z0, z1] for p in phi])
    t = []
    for i in range(n):
        j = (i + 1) % n
        t.extend([[i, j, n + j], [i, n + j, n + i]])
    return (v, np.array(t))

class GeometryTests(unittest.TestCase):

    def test_extrapolated_shoulder_is_inside_the_surface_at_any_scale(self):
        v, t = cylinder(0, 0, 0.25)
        for scale in [0.7, 1.05, 1.85]:
            seed = np.array([0.38, 0.17, 0.75]) * scale
            self.assertIsNone(depth_interval(v * scale, t, seed))
            fixed, changed = interior_arm_seed(v * scale, t, seed, 1, 0.08 * scale)
            self.assertTrue(changed)
            self.assertIsNotNone(depth_interval(v * scale, t, fixed))
            self.assertLess(fixed[0] / scale, 0.25)

    def test_visible_arm_tubes_exclude_belly_weights(self):
        v, t = cylinder(0, 0, 0.22, 0.25, 0.75)
        a, b = cylinder(0.32, 0, 0.04, 0.25, 0.75)
        c, d = cylinder(-0.32, 0, 0.04, 0.25, 0.75)
        vertices = np.concatenate([v, a, c, [[0.18, 0, 0.5], [0.32, 0, 0.5], [-0.32, 0, 0.5]]])
        triangles = np.concatenate([t, b + len(v), d + len(v) + len(a)])
        points = {}
        for side, sign in [('left', 1), ('right', -1)]:
            points[side + 'Hip'] = np.array([sign * 0.08, 0, 0.4])
            points[side + 'Hand'] = np.array([sign * 0.32, 0, 0.27])
            points[side + 'Shoulder'] = np.array([sign * 0.3, 0, 0.8])
        for scale in [0.7, 1.85]:
            mask, report = arm_region_masks(vertices * scale, triangles, scale, {k: v * scale for k, v in points.items()})
            self.assertFalse(mask['left'][-3])
            self.assertTrue(mask['left'][-2])
            self.assertFalse(mask['left'][-1])
            self.assertGreater(report['sections'], 0)

    def test_arm_ownership_excludes_nearby_thigh_and_is_scale_invariant(self):
        v = np.array([[0.28, 0, 0.5], [0.281, 0, 0.501], [0.1, 0, 0.5]])
        edges = np.array([[0, 1]])
        bones = [dict(name='leftUpperLeg', head=[0.27, 0, 0.8], tail=[0.27, 0, 0.2], role='leg'), dict(name='leftLowerArm', head=[0.4, 0, 0.8], tail=[0.4, 0, 0.2], role='arm'), dict(name='hips', head=[0, 0, 0.8], tail=[0, 0, 1], role='core')]
        regions = {'left': np.array([True, True, False]), 'right': np.zeros(3, bool), 'ownership': {'left': dict(mass=np.array([1.0, 1.0, 0.0]), observed=np.ones(3, bool))}}
        results = []
        for scale in [0.7, 1.85]:
            bs = [{**b, 'head': np.array(b['head']) * scale, 'tail': np.array(b['tail']) * scale} for b in bones]
            w, _ = diffuse(v * scale, edges, bs, scale, arm_regions=regions)
            np.testing.assert_allclose(w[:2, 1], 1)
            self.assertEqual(w[2, 1], 0)
            np.testing.assert_allclose(w.sum(axis=1), 1)
            results.append(w)
        np.testing.assert_allclose(*results, atol=1e-12)

    def test_dirichlet_arm_domain_does_not_leak_into_head(self):
        parts = [cylinder(0, 0, 0.22, 0.25, 1.0), cylinder(0.32, 0, 0.04, 0.25, 0.75), cylinder(-0.32, 0, 0.04, 0.25, 0.75)]
        vertices = np.concatenate([x[0] for x in parts])
        tri = []
        offset = 0
        for v, t in parts:
            tri.extend(t + offset)
            offset += len(v)
        tri.extend([[32, 64 + 32, 65 + 32], [33, 128 + 32, 129 + 32]])
        points = {'neck': np.array([0, 0, 0.8])}
        for side, sign in [('left', 1), ('right', -1)]:
            points[side + 'Hip'] = np.array([sign * 0.08, 0, 0.4])
            points[side + 'Hand'] = np.array([sign * 0.32, 0, 0.27])
            points[side + 'Shoulder'] = np.array([sign * 0.3, 0, 0.8])
        for scale in [0.7, 1.85]:
            mask, _ = arm_region_masks(vertices * scale, np.array(tri), scale, {k: v * scale for k, v in points.items()})
            for field in mask['ownership'].values():
                np.testing.assert_array_equal(field['mass'][vertices[:, 2] >= 0.8], 0)

    def test_depth_correction_preserves_lateral_and_height_station(self):
        v, t = cylinder(0.1, 0.02, 0.04)
        for scale in [0.7, 1.05, 1.85]:
            p = np.array([0.11, -0.016, 0.5]) * scale
            r = depth_interval(v * scale, t, p)
            np.testing.assert_allclose(r['centre'], np.array([0.11, 0.02, 0.5]) * scale, atol=1e-10)
            self.assertIsNone(depth_interval(v * scale, t, np.array([0.3, 0, 0.5]) * scale))

    def test_arm_tracks_ignore_longer_central_garment_and_legs(self):
        parts = [cylinder(0, 0, 0.22, 0.25, 0.75), cylinder(0.3, 0, 0.035, 0.4, 0.7), cylinder(-0.3, 0, 0.035, 0.4, 0.7), cylinder(0.09, 0, 0.04, 0, 0.5), cylinder(-0.09, 0, 0.04, 0, 0.5)]
        v = []
        t = []
        offset = 0
        for a, b in parts:
            v.append(a)
            t.append(b + offset)
            offset += len(a)
        v = np.concatenate(v)
        t = np.concatenate(t)
        for scale in [0.7, 1.05, 1.85]:
            tracks = arm_section_tracks(v * scale, t, scale, 0.25 * scale, 0.78 * scale, 0.16 * scale)
            for side, sign in [('left', 1), ('right', -1)]:
                np.testing.assert_allclose(tracks[side][:, 0] / scale, sign * 0.3, atol=1e-10)
                self.assertGreaterEqual(tracks[side][0, 2] / scale, 0.4)
                self.assertLessEqual(tracks[side][-1, 2] / scale, 0.7)

    def test_pivot_is_ring_centre_not_front_surface(self):
        for scale in [0.65, 1.05, 1.4, 1.85]:
            v, t = cylinder(0.1, 0.02, 0.04)
            v *= scale
            r = section(v, t, np.array([0.1, -0.016, 0.5]) * scale, [0, 0, 1], 0.15 * scale)
            np.testing.assert_allclose(r['centre'], np.array([0.1, 0.02, 0.5]) * scale, atol=1e-10)

    def test_disconnected_arm_does_not_expand_torso_width(self):
        v, t = cylinder(0, 0, 0.15)
        a, b = cylinder(0.3, 0, 0.04)
        c, d = cylinder(-0.3, 0, 0.04)
        vs = np.concatenate([v, a, c])
        ts = np.concatenate([t, b + len(v), d + len(v) + len(a)])
        r = section(vs, ts, [0, 0, 0.5], [0, 0, 1], 0.5)
        self.assertAlmostEqual(r['widths'][0], 0.3)
        self.assertEqual(len(section_loops(vs, ts, [0, 0, 0.5], [0, 0, 1])), 3)

    def test_inner_leg_selected_under_garment(self):
        v, t = cylinder(0.1, 0, 0.04)
        a, b = cylinder(0, 0, 0.25)
        r = section(np.concatenate([v, a]), np.concatenate([t, b + len(v)]), [0.1, 0, 0.5], [0, 0, 1], 0.3)
        self.assertLess(r['radius'], 0.041)

    def test_heat_masks_no_cross_limb_bleed_and_normalized(self):
        v, t = cylinder(0.1, 0, 0.04)
        a, b = cylinder(-0.1, 0, 0.04)
        vertices = np.concatenate([v, a])
        tri = np.concatenate([t, b + len(v)])
        edges = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
        bones = [dict(name=s + 'UpperLeg', head=[x, 0, 1], tail=[x, 0, 0], role='leg') for s, x in [('left', 0.1), ('right', -0.1)]]
        w, report = diffuse(vertices, edges, bones, 1.85)
        self.assertEqual(report['forbidden_weight'], 0)
        np.testing.assert_allclose(w.sum(axis=1), 1)
        np.testing.assert_allclose(w[:len(v), 1], 0)
        np.testing.assert_allclose(w[len(v):, 0], 0)
        w2, _ = diffuse(vertices * 2, edges, [{**r, 'head': (np.array(r['head']) * 2).tolist(), 'tail': (np.array(r['tail']) * 2).tolist()} for r in bones], 3.7)
        np.testing.assert_allclose(w, w2, atol=1e-12)

    def test_hand_below_hip_stays_with_arm_and_does_not_capture_leg(self):
        leg, lt = cylinder(0.1, 0, 0.04, 0, 1)
        arm, at = cylinder(0.4, 0, 0.025, 0.4, 1.2)
        v = np.concatenate([leg, arm])
        t = np.concatenate([lt, at + len(leg)])
        edges = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1), axis=0)
        bones = [dict(name='leftUpperLeg', head=[0.1, 0, 1], tail=[0.1, 0, 0], role='leg'), dict(name='leftLowerArm', head=[0.4, 0, 1.2], tail=[0.4, 0, 0.4], role='arm'), dict(name='hips', head=[0, 0, 1], tail=[0, 0, 1.3], role='core')]
        w, _ = diffuse(v, edges, bones, 1.85)
        self.assertGreater(w[len(leg):, 1].min(), 0.99)
        self.assertLess(w[:len(leg), 1].max(), 0.01)
if __name__ == '__main__':
    unittest.main()
