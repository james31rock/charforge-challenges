import copy, unittest
import numpy as np
from .hands import anatomy, surface_metrics, SUPPORT_IDS, REST_IDS
from .core import evaluate
from .test_core import fixture

def planted(height=1.6):
    t = fixture(height)
    t.update(scope='c7', intent='other', hand_support_required=True)
    t['source'] = dict(clip='synthetic-planted', cache_sha256='synthetic-cache')
    t['hand_support'] = dict(schema='qa-hand-support/1', bound=True, fps=24, frame_count=24, source_sha256='synthetic-source', cache_sha256='synthetic-cache', hands={s: dict(intervals=[[2, 22]], method='Synthetic independent support labels') for s in 'LR'})
    points = np.array([[-0.02, 0, 0], [0.02, 0, 0], [-0.02, 0.05, 0], [0.02, 0.05, 0]]) * height
    for f in t['frames']:
        f['hands'] = {s: dict(palm_points=points.tolist(), palm_plane_deg=0.0, palm_gap_m=0.0, ground_depth_m=0.0, wrist_extension_deg=80.0, finger_bend_deg=0.0) for s in 'LR'}
    return t

def result(t, cid):
    return next((c for c in evaluate(t)['checks'] if c['id'] == cid))

class HandTests(unittest.TestCase):

    def test_clean_four_statures(self):
        for h in (1.05, 1.12, 1.85, 0.9):
            t = planted(h)
            r = evaluate(t)
            self.assertTrue(all((c['status'] == 'PASS' for c in r['checks'])))
            self.assertFalse(r['presentation_allowed'], 'C7 subset is never full acceptance')

    def test_independent_falsifiers_four_statures(self):
        for h in (1.05, 1.12, 1.85, 0.9):
            for cid, key, value in [('C7.palm_plane', 'palm_plane_deg', 65.0), ('C7.palm_contact', 'palm_gap_m', 0.03 * h), ('C7.wrist_extension', 'wrist_extension_deg', 0.0), ('C7.wrist_extension', 'wrist_extension_deg', -80.0), ('C7.hand_ground_penetration', 'ground_depth_m', 0.02 * h), ('C7.support_fingers', 'finger_bend_deg', 60.0)]:
                t = planted(h)
                t['frames'][7]['hands']['L'][key] = value
                self.assertEqual(result(t, cid)['status'], 'FAIL', (h, cid))
            for cid, key in [('C7.rest_wrist', 'wrist_neutral_deg'), ('C7.rest_fingers', 'finger_bend_deg')]:
                t = planted(h)
                t['rest_hands']['R'][key] = 50.0
                self.assertEqual(result(t, cid)['status'], 'FAIL', (h, cid))
            t = planted(h)
            t['frames'][7]['hands']['L']['palm_points'][0][0] += 0.1 * h
            self.assertEqual(result(t, 'C7.hand_skate')['status'], 'FAIL')

    def test_missing_plan_cannot_pass_cartwheel(self):
        t = planted()
        del t['hand_support']
        t['source']['clip'] = 'suite:cartwheel'
        t['hand_support_required'] = False
        self.assertTrue(set(SUPPORT_IDS) <= set(evaluate(t)['unavailable']))
        t = planted()
        t['hand_support']['bound'] = False
        t['frames'][8]['hands']['L']['wrist_extension_deg'] = -90.0
        self.assertEqual(result(t, 'C7.wrist_extension')['status'], 'UNAVAILABLE')

    def test_missing_surface_and_rest(self):
        t = planted()
        t['frames'][8]['hands'].pop('R')
        t.pop('rest_hands')
        self.assertTrue(set(SUPPORT_IDS + REST_IDS) <= set(evaluate(t)['unavailable']))

    def test_unknown_motion_name_cannot_waive_source_coverage(self):
        t = planted()
        t.pop('hand_support')
        t['hand_support_required'] = False
        t['source']['clip'] = 'Take001'
        self.assertTrue(set(SUPPORT_IDS) <= set(evaluate(t)['unavailable']))
        t['source'] = dict(clip='rest')
        self.assertTrue(all((not c['applicable'] for c in evaluate(t)['checks'] if c['id'] in SUPPORT_IDS)))
        t = fixture()
        t['source']['clip'] = 'get-up-without-hands'
        self.assertTrue(all((not c['applicable'] for c in evaluate(t)['checks'] if c['id'] in SUPPORT_IDS)))
        t['hand_support_required'] = True
        self.assertTrue(set(SUPPORT_IDS) <= set(evaluate(t)['unavailable']))

    def test_missing_support_frames_blocks(self):
        t = planted()
        t['frames'] = t['frames'][5:]
        self.assertTrue(set(SUPPORT_IDS) <= set(evaluate(t)['unavailable']))

    def test_outside_source_support_does_not_fail(self):
        t = planted()
        t['frames'][0]['hands']['L']['wrist_extension_deg'] = -90.0
        self.assertEqual(result(t, 'C7.wrist_extension')['status'], 'PASS')

    def test_float_does_not_hide_skate(self):
        t = planted()
        for i, f in enumerate(t['frames']):
            for side in 'LR':
                a = np.asarray(f['hands'][side]['palm_points'])
                a[:, 2] = 0.3
                a[:, 0] += 0.02 * i
                f['hands'][side].update(palm_points=a.tolist(), palm_gap_m=0.3)
        self.assertEqual(result(t, 'C7.hand_skate')['status'], 'FAIL')

    def test_source_binding_mismatch_and_invalid_values(self):
        for field, value in [('cache_sha256', 'wrong'), ('fps', 30), ('frame_count', 2)]:
            t = planted()
            t['hand_support'][field] = value
            with self.assertRaises(ValueError):
                evaluate(t)
        t = planted()
        t['frames'][8]['hands']['R']['wrist_extension_deg'] = float('nan')
        with self.assertRaises(ValueError):
            evaluate(t)

    def test_signed_extension_mirror_and_finger_spread(self):
        for side, sign in [('L', 1), ('R', -1)]:
            fingers = [[np.array([s * 0.15, 1.0, 0])] * 3 for s in (-1, -0.4, 0.4, 1)]
            a = anatomy(np.array([0, 0, 1.0]), np.zeros(3), np.array([0, 1.0, 0]), np.array([sign * 0.2, 0.8, 0]), np.array([-sign * 0.2, 0.8, 0]), fingers, side)
            self.assertAlmostEqual(a['wrist_extension_deg'], 90.0)
            self.assertAlmostEqual(a['finger_bend_deg'], 0.0)
            b = anatomy(np.array([0, 0, -1.0]), np.zeros(3), np.array([0, 1.0, 0]), np.array([sign * 0.2, 0.8, 0]), np.array([-sign * 0.2, 0.8, 0]), fingers, side)
            self.assertAlmostEqual(b['wrist_extension_deg'], -90.0)

    def test_actual_skin_plane_and_dorsal_falsifier(self):
        p = np.array([[-0.02, 0, 0], [0.02, 0, 0], [-0.02, 0.05, 0], [0.02, 0.05, 0]])
        self.assertAlmostEqual(surface_metrics(p, [0, 0, -1], p, 0)['palm_plane_deg'], 0.0)
        self.assertAlmostEqual(surface_metrics(p, [0, 0, 1], p, 0)['palm_plane_deg'], 180.0)
        with self.assertRaises(ValueError):
            surface_metrics(np.zeros((4, 3)), [0, 0, -1], p, 0)

    def test_reviewed_source_wrist_band(self):
        t = planted()
        for side in 'LR':
            t['hand_support']['hands'][side].update(wrist_extension_range_deg=[45.0, 55.0], wrist_profile_reason='Independent synthetic source support profile')
            for f in t['frames']:
                f['hands'][side]['wrist_extension_deg'] = 50.0
        self.assertEqual(result(t, 'C7.wrist_extension')['status'], 'PASS')
        t['frames'][10]['hands']['L']['wrist_extension_deg'] = 80.0
        self.assertEqual(result(t, 'C7.wrist_extension')['status'], 'FAIL')

    def test_unreviewed_or_reversed_source_band_rejected(self):
        t = planted()
        t['hand_support']['hands']['L']['wrist_extension_range_deg'] = [45.0, 55.0]
        with self.assertRaises(ValueError):
            evaluate(t)
        t['hand_support']['hands']['L'].update(wrist_extension_range_deg=[90.0, 40.0], wrist_profile_reason='bad')
        with self.assertRaises(ValueError):
            evaluate(t)
if __name__ == '__main__':
    unittest.main()
