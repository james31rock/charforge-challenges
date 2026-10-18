import copy, unittest
import numpy as np
from toolkit.qa.core import evaluate, validate_trace, support_margin, foot_phases

def fixture(height=1.6):
    h = height
    frames = []
    for i in range(24):
        t = i / 24
        head = np.array([0, 0.02 * np.sin(t * 2 * np.pi), 0.9]) * h
        points = {'hips': [0, 0, 0.5 * h], 'head': head.tolist(), 'chest': [0, 0, 0.7 * h]}
        for s, sign in [('L', 1), ('R', -1)]:
            for key, y, z in [('hip', 0, 0.5), ('knee', 0.02, 0.28), ('ankle', 0, 0.04), ('shoulder', 0, 0.72), ('elbow', 0.02, 0.58), ('wrist', 0, 0.46)]:
                points[s + key] = [sign * 0.07 * h, y * h, z * h]
        feet = {s: dict(heel=[sign * 0.07 * h, -0.03 * h, 0], toe=[sign * 0.07 * h, 0.06 * h, 0], min_z=0, support=[[sign * 0.07 * h + dx * h, y * h, 0] for dx, y in [(-0.025, -0.03), (0.025, -0.03), (0.025, 0.06), (-0.025, 0.06)]]) for s, sign in [('L', 1), ('R', -1)]}
        frames.append(dict(frame=i + 1, points=points, feet=feet, stance={'L': True, 'R': True}, landing=i >= 18, hand_depth_m=0.0, hair_depth_m=0.0, hair_relative_tip=[0, 0.02 * h * np.sin(t * 2 * np.pi), -0.1 * h], section_retained=1.0, volume_retained=1.0, webbing_strain=1.0, com=[0, 0, 0.5 * h]))
    return dict(schema='qa-trace/1', units='metres-z-up', height_m=h, hip_width_m=0.14 * h, leg_length_m=0.46 * h, fps=24, intent='walk', source=dict(clip='synthetic-walk', cache_sha256='synthetic-cache'), hand_support=dict(schema='qa-hand-support/1', bound=True, fps=24, frame_count=24, source_sha256='synthetic-source', cache_sha256='synthetic-cache', hands={s: dict(intervals=[], method='Independent synthetic walk intent: no hand support') for s in 'LR'}), rest_hands={s: dict(wrist_neutral_deg=0.0, finger_bend_deg=0.0) for s in 'LR'}, floor_m=0, travel_observable=True, frames=frames, calibration={'validated': True, 'fixture': 'analytic only'}, coverage={k: True for k in ('hand_volume', 'hair_volume', 'hair_tips', 'stance', 'section_retained', 'volume_retained', 'webbing_strain', 'mass_model')})

def check(t, cid):
    return next((x for x in evaluate(t)['checks'] if x['id'] == cid))

class Tests(unittest.TestCase):

    def test_clean_analytic(self):
        self.assertEqual(evaluate(fixture())['status'], 'PASS')

    def test_hand_and_landing(self):
        t = fixture()
        t['frames'][20]['hand_depth_m'] = 0.02
        for cid in ['C1.hand_penetration', 'C6.landing_hands']:
            self.assertEqual(check(t, cid)['status'], 'FAIL')

    def test_open_surface_crossing_blocks(self):
        t = fixture()
        t['coverage']['hand_volume'] = False
        t['frames'][20]['hand_crossing_extent_m'] = 0.02
        for cid in ['C1.hand_penetration', 'C6.landing_hands']:
            self.assertEqual(check(t, cid)['status'], 'FAIL')

    def test_hair_collision(self):
        t = fixture()
        t['frames'][10]['hair_depth_m'] = 0.02
        self.assertEqual(check(t, 'C2.hair_penetration')['status'], 'FAIL')

    def test_rigid_hair(self):
        t = fixture()
        for f in t['frames']:
            f['hair_relative_tip'] = [0, 0, -0.1]
        self.assertEqual(check(t, 'C2.hair_response')['status'], 'FAIL')

    def test_unexcited_hair(self):
        t = fixture()
        for f in t['frames']:
            f['points']['head'] = [0, 0, 1.0]
        self.assertEqual(check(t, 'C2.hair_response')['status'], 'UNAVAILABLE')

    def test_tiptoe(self):
        t = fixture()
        for f in t['frames']:
            for foot in f['feet'].values():
                foot['heel'][2] = 0.07
        self.assertEqual(check(t, 'C3.tiptoe')['status'], 'FAIL')

    def test_toe_off_is_allowed(self):
        t = fixture()
        for f in t['frames'][-3:]:
            for foot in f['feet'].values():
                foot['heel'][2] = 0.07
        self.assertEqual(check(t, 'C3.tiptoe')['status'], 'PASS')

    def test_round_sole_is_not_tiptoe(self):
        t = fixture()
        for f in t['frames']:
            for foot in f['feet'].values():
                foot['heel'][2] = 0.012
        self.assertEqual(check(t, 'C3.tiptoe')['status'], 'PASS')

    def test_sustained_flat_then_heel_rise(self):
        t = fixture()
        for f in t['frames'][8:]:
            for foot in f['feet'].values():
                foot['heel'][2] = 0.07
        self.assertEqual(check(t, 'C3.tiptoe')['status'], 'PASS')

    def test_bad_threshold(self):
        for cfg in ({'pop_height': float('nan')}, {'typo': 1.0}):
            with self.assertRaises(ValueError):
                evaluate(fixture(), cfg)

    def test_skate(self):
        t = fixture()
        for i, f in enumerate(t['frames']):
            for foot in f['feet'].values():
                for key in ('heel', 'toe'):
                    foot[key][0] += 0.02 * i
                for point in foot['support']:
                    point[0] += 0.02 * i
        self.assertEqual(check(t, 'C3.skate')['status'], 'FAIL')

    def test_no_virtual_ground(self):
        t = fixture()
        t['travel_observable'] = False
        self.assertEqual(check(t, 'C3.skate')['status'], 'UNAVAILABLE')

    def test_ground(self):
        t = fixture()
        t['frames'][4]['feet']['L']['min_z'] = -0.02
        self.assertEqual(check(t, 'C3.ground_penetration')['status'], 'FAIL')

    def test_float(self):
        t = fixture()
        t['frames'][4]['feet']['L']['min_z'] = 0.1
        self.assertEqual(check(t, 'C3.float')['status'], 'FAIL')

    def test_gait_width(self):
        t = fixture()
        for f in t['frames']:
            f['points']['Lankle'][0] = 0.5
            f['points']['Lknee'][0] = 0.5
        for cid in ['C4.step_width', 'C4.knee_separation', 'C4.width_leg_length']:
            self.assertEqual(check(t, cid)['status'], 'FAIL')

    def test_volume_and_section(self):
        t = fixture()
        t['frames'][6]['volume_retained'] = 0.2
        t['frames'][6]['section_retained'] = 0.2
        for cid in ['C5.volume', 'C5.section']:
            self.assertEqual(check(t, cid)['status'], 'FAIL')

    def test_webbing(self):
        t = fixture()
        t['frames'][5]['webbing_strain'] = 4.0
        self.assertEqual(check(t, 'C5.webbing')['status'], 'FAIL')

    def test_pop(self):
        t = fixture()
        t['frames'][9]['points']['Lwrist'][0] += 0.2
        self.assertEqual(check(t, 'J1.pop')['status'], 'FAIL')

    def test_short_window_pop(self):
        t = fixture()
        t['frames'] = t['frames'][:3]
        t['frames'][1]['points']['Lwrist'][0] += 0.2
        self.assertEqual(check(t, 'J1.pop')['status'], 'FAIL')

    def test_joint_limit(self):
        t = fixture()
        f = t['frames'][8]
        f['points']['Lankle'] = f['points']['Lhip'][:]
        self.assertEqual(check(t, 'J2.joint_limits')['status'], 'FAIL')

    def test_static_support(self):
        t = fixture()
        t['intent'] = 'static'
        for f in t['frames']:
            f['com'] = [1, 0, 1]
        self.assertEqual(check(t, 'B1.static_support')['status'], 'FAIL')

    def test_scale_invariance(self):
        for h in (1.05, 1.12, 1.85, 0.9):
            self.assertEqual(evaluate(fixture(h))['status'], 'PASS')

    def test_missing_and_uncalibrated_block(self):
        t = fixture()
        t['coverage']['hand_volume'] = False
        self.assertEqual(check(t, 'C1.hand_penetration')['status'], 'UNAVAILABLE')
        self.assertFalse(evaluate(t)['presentation_allowed'])
        t = fixture()
        t['calibration'] = {}
        self.assertEqual(evaluate(t)['status'], 'BLOCKED')

    def test_bad_trace(self):
        for mutate in (lambda t: t.update(height_m=float('nan')), lambda t: t['frames'][3].update(frame=9), lambda t: t.update(units='centimetres')):
            t = fixture()
            mutate(t)
            with self.assertRaises(ValueError):
                validate_trace(t)

    def test_support_geometry(self):
        self.assertAlmostEqual(support_margin([0.5, 0.5], [[0, 0], [1, 0], [1, 1], [0, 1]]), 0.5)
        self.assertAlmostEqual(support_margin([2, 0.5], [[0, 0], [1, 0], [1, 1], [0, 1]]), -1)
if __name__ == '__main__':
    unittest.main()
