import sys, unittest, json, math
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from toolkit.secondary.physics import Lattice, Capsule, resolve_preset

class HairTests(unittest.TestCase):

    def test_virtual_midpoint_catches_body_missed_by_endpoints(self):
        p = resolve_preset('hair_short_bob', dict(height_cm=105, mass_proxy_kg=20), 0.6)
        p.update(segment_collision_samples=3, gravity_m_s2=0.0, max_angle_deg=70)
        rest = np.array([[0.3, 0.0, 0.3], [-0.3, 0.0, 0.3]])
        cap = Capsule(np.array([0.0, 0, 0.3]), np.array([0.0, 0, 0.3]), 0.07)
        self.assertTrue((cap.clearance(rest)[0] > 0).all())
        self.assertLess(cap.clearance(rest.mean(0)[None, :])[0][0], 0)
        sim = Lattice(rest, [-1, 0], [True, False], p, iterations=16)
        for _ in range(80):
            sim.step(rest, 1 / 240, [cap])
        points = np.linspace(sim.pos[0], sim.pos[1], 65)
        self.assertGreaterEqual(float(cap.clearance(points, p['radius_m'])[0].min()), -0.002)
        self.assertLess(abs(np.linalg.norm(sim.pos[1] - sim.pos[0]) - 0.6), 0.006)

    def test_no_motion_no_invented_sway_and_acceleration_response(self):
        profile = json.loads(Path(__file__).with_name('hair_profiles.json').read_text())['classes']['hair_short_bob']
        amplitudes = []
        for drive in [0.0, 0.005, 0.01]:
            p = resolve_preset('hair_short_bob', dict(height_cm=105, mass_proxy_kg=20), 0.16)
            omega = 2 * math.pi * profile['frequency_hz'] * math.sqrt(0.25 / 0.16)
            p.update(profile, omega=omega, gravity_m_s2=0.0)
            rest = np.array([[0.0, 0, 1], [0.02, 0, 0.92], [0.04, 0, 0.84]])
            sim = Lattice(rest, [-1, 0, 1], [True, False, False], p, iterations=8)
            values = []
            for i in range(360):
                target = rest + [drive * math.sin(2 * math.pi * i / 240), 0, 0]
                sim.step(target, 1 / 240)
                values.append(sim.pos[-1] - target[-1])
            amplitudes.append(float(np.ptp(np.array(values), axis=0)[0]))
        self.assertLess(amplitudes[0], 1e-10)
        self.assertGreater(amplitudes[1], 0.001)
        self.assertGreater(amplitudes[2] / amplitudes[1], 1.8)
        self.assertLess(amplitudes[2] / amplitudes[1], 2.2)
if __name__ == '__main__':
    unittest.main()
