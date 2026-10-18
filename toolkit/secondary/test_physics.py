import sys, unittest, math, json
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from toolkit.secondary.physics import resolve_preset, Lattice, Capsule, assess, PRESETS
from toolkit.secondary.topology import from_roots_tips

def trial(height, mass, name='hair_long', speed=1, flip=False):
    p = resolve_preset(name, dict(height_cm=height, mass_proxy_kg=mass))
    L = p['length_m']
    n = p['segments']
    rest = np.array([[0, 0, 1 - i * L / n] for i in range(n + 1)])
    sim = Lattice(rest, [-1] + list(range(n)), [True] + [False] * n, p, iterations=4)
    out = []
    targets = []
    for f in range(97):
        for sub in range(10):
            t = max(0, (f - 1 + (sub + 1) / 10) / 24)
            active = min(t, 2)
            a = (math.pi * 2 * active / 2 if flip else 0.18 * math.sin(2 * math.pi * active)) * speed
            c, s = (math.cos(a), math.sin(a))
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            target = (rest - rest[0]) @ R.T + rest[0] + [0.05 * L * math.sin(math.pi * active) * speed, 0, 0]
            sim.step(target, 1 / 240)
        out.append(sim.pos.copy())
        targets.append(target.copy())
    return (np.array(out), np.array(targets), assess(out, targets, 24, p, sim.parents, stop_frame=48))

class PhysicsTests(unittest.TestCase):

    def test_constant_translation_does_not_create_permanent_drag(self):
        p = resolve_preset('skirt', dict(height_cm=105, mass_proxy_kg=20), 0.2)
        p['gravity_m_s2'] = 0
        rest = np.array([[0.0, 0, 1], [0, 0, 0.9], [0, 0, 0.8]])
        sim = Lattice(rest, [-1, 0, 1], [True, False, False], p)
        for f in range(480):
            target = rest + np.array([0.5 * (f + 1) / 240, 0, 0])
            sim.step(target, 1 / 240)
        self.assertLess(float(np.max(np.linalg.norm(sim.pos - target, axis=1))), 1e-05)

    def test_deterministic(self):
        a, _, _ = trial(105, 20)
        b, _, _ = trial(105, 20)
        self.assertTrue(np.array_equal(a, b))

    def test_scale_and_mass(self):
        a = resolve_preset('skirt', dict(height_cm=105, mass_proxy_kg=20))
        b = resolve_preset('skirt', dict(height_cm=210, mass_proxy_kg=160))
        self.assertAlmostEqual(b['length_m'] / a['length_m'], 2)
        self.assertAlmostEqual(b['omega'] / a['omega'], math.sqrt(0.5))
        self.assertAlmostEqual(b['stiffness_n_m'] / a['stiffness_n_m'], 4)

    def test_capsule_and_sphere(self):
        c = Capsule(np.array([0.0, 0, 0]), np.array([0.0, 0, 1]), 0.2)
        d, _ = c.clearance(np.array([[0.3, 0, 0.5], [0, 0, 0.5], [0, 0, 1.3]]), 0.02)
        np.testing.assert_allclose(d, [0.08, -0.22, 0.08])
        c.b = c.a.copy()
        self.assertAlmostEqual(c.clearance(np.array([[0.3, 0, 0]]))[0][0], 0.1)

    def test_collision_resolves(self):
        p = resolve_preset('tail', dict(height_cm=100, mass_proxy_kg=20), 0.4)
        s = Lattice([[0, 0, 0.4], [0, 0, 0]], [-1, 0], [True, False], p)
        c = Capsule(np.array([0.0, 0, 0]), np.array([0.0, 0, 0]), 0.04)
        for _ in range(60):
            s.step(s.rest, 1 / 240, [c])
        self.assertGreaterEqual(c.clearance(s.pos[1:], p['radius_m'])[0][0], -1e-10)

    def test_negative_gates(self):
        x, t, _ = trial(105, 20)
        p = resolve_preset('hair_long', dict(height_cm=105, mass_proxy_kg=20))
        x[20, 2] += 100
        r = assess(x, t, 24, p, [-1, 0, 1, 2, 3, 4])
        self.assertFalse(r['explosion_pass'])
        self.assertFalse(r['jitter_pass'])

    def test_missing_measurements(self):
        with self.assertRaises(KeyError):
            resolve_preset('skirt', {})
        with self.assertRaises(ValueError):
            resolve_preset('skirt', dict(height_cm=0, mass_proxy_kg=20))

    def test_lateral_ring_has_pinned_seam_and_stable_hem(self):
        angles = np.arange(8) * 2 * math.pi / 8
        roots = np.array([[0.2 * math.cos(a), 0.2 * math.sin(a), 1] for a in angles])
        tips = np.array([[0.28 * math.cos(a), 0.28 * math.sin(a), 0.7] for a in angles])
        topo = from_roots_tips(roots, tips, 3, 'ring', True)
        p = resolve_preset('skirt', dict(height_cm=140, mass_proxy_kg=40), 0.31)
        sim = Lattice(preset=p, **topo)
        for _ in range(24):
            sim.step(topo['rest'], 1 / 240)
        np.testing.assert_array_equal(sim.pos[sim.pin], topo['rest'][sim.pin])
        self.assertTrue(np.isfinite(sim.pos).all())
        self.assertLess(np.linalg.norm(sim.pos - topo['rest'], axis=1).max(), 0.05)

    def test_regression_cast_flip(self):
        for h, m in [(112, 30), (105, 20), (185, 110), (70, 12)]:
            _, _, r = trial(h, m, flip=True)
            self.assertTrue(r['explosion_pass'])
            self.assertTrue(r['constraint_pass'])
if __name__ == '__main__':
    unittest.main()
