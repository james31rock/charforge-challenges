import copy, math, tempfile, json, unittest
from pathlib import Path
import numpy as np
from .core import compare, measure, load, smoothness

def fixture(leg=0.8, hip=0.24, body=0.7):
    rest = {'hips': [0, 0, leg + 0.1], 'chest': [0, 0, leg + 0.1 + body * 0.6], 'head': [0, 0, leg + 0.1 + body]}
    for s, sign in [('left', 1), ('right', -1)]:
        for name, point in [('UpperLeg', [sign * hip / 2, 0, leg + 0.1]), ('LowerLeg', [sign * hip / 2, -0.03 * leg, 0.5 * leg + 0.1]), ('Foot', [sign * hip / 2, 0, 0.1]), ('Toes', [sign * hip / 2, -0.2 * leg, 0.1])]:
            rest[s + name] = point
    samples = []
    for i in range(24):
        p = copy.deepcopy(rest)
        dz = 0.02 * leg * math.cos(i / 24 * 4 * math.pi)
        for name in p:
            p[name][2] += dz
        for s, offset in [('left', 0), ('right', math.pi)]:
            u = i / 24 * 2 * math.pi + offset
            p[s + 'LowerLeg'][1] += 0.12 * leg * math.sin(u)
            p[s + 'Foot'][1] += 0.3 * leg * math.sin(u)
            p[s + 'Toes'][1] += 0.3 * leg * math.sin(u)
        samples.append({'phase': i / 24, 'joints': p})
    return {'schema_version': 1, 'kind': 'charforge-gait-trace', 'label': 'analytic', 'cycle_seconds': 1, 'rest': rest, 'samples': samples, 'provenance': {}}

class GeometryTests(unittest.TestCase):

    def test_identity(self):
        a = fixture()
        r = compare(a, a)
        for row in r['phase_matched_error'].values():
            if row['rmse'] is not None:
                self.assertAlmostEqual(row['rmse'], 0)

    def test_scaled_rotated_translated_invariance(self):
        a = fixture()
        b = copy.deepcopy(a)
        theta = 0.73
        R = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        for row in [b['rest']] + [s['joints'] for s in b['samples']]:
            for n, p in row.items():
                row[n] = (R @ np.array(p) * 2.7 + [3, 4, 5]).tolist()
        r = compare(a, b)
        for key, row in r['phase_matched_error'].items():
            if row['rmse'] is not None:
                self.assertLess(row['rmse'], 1e-07, key)

    def test_phase_origin(self):
        a = fixture()
        b = copy.deepcopy(a)
        b['phase_origin'] = 0.25
        for i in range(24):
            b['samples'][i]['joints'] = copy.deepcopy(a['samples'][(i - 6) % 24]['joints'])
        self.assertLess(compare(a, b)['phase_matched_error']['left_knee_deg']['rmse'], 1e-08)

    def test_root_pop_falsifier(self):
        a = fixture()
        b = copy.deepcopy(a)
        for p in b['samples'][8]['joints'].values():
            p[2] += 0.08
        self.assertGreater(measure(b)['summary']['pelvis']['peak_phase_jerk'], 5 * measure(a)['summary']['pelvis']['peak_phase_jerk'])

    def test_cadence_does_not_change_geometry(self):
        a = fixture()
        b = copy.deepcopy(a)
        b['cycle_seconds'] = 2
        x, y = [measure(d)['summary']['pelvis'] for d in [a, b]]
        self.assertEqual(x['peak_phase_jerk'], y['peak_phase_jerk'])
        self.assertAlmostEqual(x['peak_time_jerk'] / y['peak_time_jerk'], 8)

    def test_straight_knee_heading_unavailable(self):
        a = fixture()
        for s in a['samples']:
            for side in ['left', 'right']:
                j = s['joints']
                j[side + 'LowerLeg'] = ((np.array(j[side + 'UpperLeg']) + j[side + 'Foot']) / 2).tolist()
        self.assertTrue(all((v is None for v in measure(a)['curves']['left_knee_progression_deg'])))

    def test_ankle_dorsiflexion_sign(self):
        a = fixture()
        for row in [a['rest']] + [s['joints'] for s in a['samples']]:
            row['leftLowerLeg'] = [0.12, 0, 0.5]
            row['leftFoot'] = [0.12, 0, 0.1]
            row['leftToes'] = [0.12, -0.2, 0.1]
        for s in a['samples']:
            s['joints']['leftToes'] = [0.12, -0.2 * math.cos(math.pi / 6), 0.2]
        self.assertTrue(np.allclose(measure(a)['curves']['left_ankle_deg'], 30))

    def test_unclosed_capture_does_not_create_seam_jerk(self):
        y = np.linspace(0, 1, 24)
        self.assertLess(smoothness(y, 1, False)['peak_time_jerk'], 1e-08)
        self.assertGreater(smoothness(y, 1, True)['peak_time_jerk'], 1000)

    def test_declared_contact_not_quantized_to_sample_fraction(self):
        a = fixture()
        a['declared_stance_fraction'] = 0.6
        s = measure(a)['summary']['left_stance']
        self.assertEqual(s['fraction'], 0.6)
        self.assertEqual(s['sampled_fraction'], 0.625)

    def test_four_proportions_same_code(self):
        for L, H, B in [(0.45, 0.103, 0.55), (0.36, 0.17, 0.65), (0.76, 0.31, 1.0), (0.26, 0.14, 0.6)]:
            with self.subTest(L=L):
                self.assertAlmostEqual(measure(fixture(L, H, B))['summary']['pelvis']['range'], 0.04, delta=0.001)

    def test_invalid_trace_rejected(self):
        root = Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            p = Path(tmp) / 'bad.json'
            d = fixture()
            d['samples'][0]['joints']['hips'][0] = float('nan')
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                load(p)
if __name__ == '__main__':
    unittest.main()
