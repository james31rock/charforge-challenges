import copy, unittest
import numpy as np
from .test_core import fixture
from .naturalness import measure_full, phase_difference, rank

def full_fixture():
    d = fixture()
    d['height_m'] = 1.6
    d['stride_vector_m'] = [0, -1, 0]
    for s, sign in [('left', 1), ('right', -1)]:
        for suffix, pos in [('UpperArm', [0.2 * sign, 0, 1.4]), ('LowerArm', [0.2 * sign, 0, 1.1]), ('Hand', [0.2 * sign, 0, 0.85])]:
            d['rest'][s + suffix] = pos
            for row in d['samples']:
                row['joints'][s + suffix] = pos.copy()
    d['rest_rotations'] = {k: np.eye(3).tolist() for k in ['hips', 'chest', 'head']}
    for i, row in enumerate(d['samples']):
        row['rotations'] = copy.deepcopy(d['rest_rotations'])
        t = np.radians(8) * np.sin(2 * np.pi * i / 24)
        row['rotations']['hips'] = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
    return d

class NaturalnessTests(unittest.TestCase):

    def test_phase_opposition(self):
        x = np.sin(np.arange(48) / 48 * 2 * np.pi)
        self.assertAlmostEqual(phase_difference(x, -x), 180)
        self.assertIsNone(phase_difference(x, np.ones(48)))

    def test_no_invented_contacts(self):
        m = measure_full(full_fixture())['metrics']
        self.assertIsNone(m['double_support_fraction'])
        self.assertIsNone(m['stance_fraction'])

    def test_declared_double_support_discretization(self):
        d = full_fixture()
        d['declared_stance_fraction'] = 0.6
        m = measure_full(d)['metrics']
        self.assertAlmostEqual(m['double_support_fraction'], 0.2)

    def test_pelvis_rotation(self):
        m = measure_full(full_fixture())['metrics']
        self.assertAlmostEqual(m['hips_yaw_rom_deg'], 16)
        self.assertAlmostEqual(m['head_yaw_rom_deg'], 0)
        self.assertAlmostEqual(m['thorax_pelvis_relative_yaw_rom_deg'], 16)

    def test_scale_invariance(self):
        a = full_fixture()
        b = copy.deepcopy(a)
        for row in [b['rest']] + [x['joints'] for x in b['samples']]:
            for k, v in row.items():
                row[k] = (np.array(v) * 3).tolist()
        b['height_m'] *= 3
        b['stride_vector_m'] = (np.array(b['stride_vector_m']) * 3).tolist()
        ma, mb = (measure_full(a)['metrics'], measure_full(b)['metrics'])
        for key in ma:
            if ma[key] is not None:
                self.assertAlmostEqual(ma[key], mb[key], places=8, msg=key)

    def test_cadence_and_flat_control_rank(self):
        a = measure_full(full_fixture())
        d = full_fixture()
        d['cycle_seconds'] = 2
        b = measure_full(d)
        self.assertEqual(b['metrics']['cadence_steps_min'], 60)
        r = next((r for r in rank([a], [b]) if r['metric'] == 'cadence_steps_min'))
        self.assertEqual(r['max_excess'], 6)

    def test_heel_toe_slope(self):
        d = full_fixture()
        for row in d['samples']:
            row['feet'] = {s: {'heel': [0, 0, 0], 'toe': [0, -0.2, 0.1]} for s in ['left', 'right']}
        m = measure_full(d)['metrics']
        self.assertAlmostEqual(m['heel_strike_roll_deg'], np.degrees(np.arctan(0.5)))

    def test_non_rotation_rejected(self):
        d = full_fixture()
        d['rest_rotations']['head'][0][0] = 2
        with self.assertRaises(ValueError):
            measure_full(d)

    def test_actual_occupancy_is_not_declared_support(self):
        d = full_fixture()
        d['declared_stance_fraction'] = 0.6
        for r in d['samples']:
            r['feet'] = {s: {'min_z_m': 0} for s in ['left', 'right']}
        m = measure_full(d)['metrics']
        self.assertAlmostEqual(m['double_support_fraction'], 0.2)
        self.assertEqual(m['geometric_double_support_fraction'], 1)
if __name__ == '__main__':
    unittest.main()
