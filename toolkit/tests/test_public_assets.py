import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from toolkit.face import c8_gate
from toolkit.face.mouth_likeness import descriptors, compare as compare_mouth
from toolkit.face.surface import load_groups
from toolkit.gait.core import compare
from toolkit.layers.core import tps, voxel_surface
from toolkit.qa.hands import DEFAULTS, evaluate_hands, validate
from toolkit.rig.geometry import detect, skeleton
from toolkit.rig.weights import diffuse
from toolkit.secondary.physics import Lattice, resolve_preset
from toolkit.secondary.topology import from_roots_tips

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def blender(script, args):
    exe = os.environ.get('BLENDER') or shutil.which('blender')
    if not exe:
        raise RuntimeError('Blender is required for the complete test suite; set BLENDER')
    result = subprocess.run([exe, '-b', '--factory-startup', '-Y', '-t', '2',
                             '--python-exit-code', '1', '--python', str(script), '--',
                             *map(str, args)], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout[-3000:] + result.stderr[-1000:])


class PublicAssets(unittest.TestCase):
    def test_qa_native_support_control_and_failure(self):
        for folder, pose, expected in [
            ('tests/control', 'tests/control/poses.npz', True),
            ('assets', 'baseline/motion.npz', False),
        ]:
            with self.subTest(case=folder), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'trace.json'
                blender(ROOT / 'hand-plant/extract.py',
                        [ROOT / 'hand-plant' / folder, ROOT / 'hand-plant' / pose, out])
                trace = read(out)
                validate(trace)
                checks, _ = evaluate_hands(trace, DEFAULTS)
                self.assertEqual(all(c['status'] == 'PASS' for c in checks), expected)
                if expected:
                    mutated = copy.deepcopy(trace)
                    mutated['frames'][0]['hands']['L']['palm_plane_deg'] = 80.
                    changed, _ = evaluate_hands(mutated, DEFAULTS)
                    self.assertTrue(any(c['id'] == 'C7.palm_plane' and c['status'] == 'FAIL' for c in changed))

    def test_gait_public_pose_geometry_and_scale(self):
        with np.load(ROOT / 'hand-plant/baseline/motion.npz', allow_pickle=False) as data:
            names = data['point_names'].tolist()
            mapping = {'hips': 'hips', 'chest': 'chest', 'head': 'head'}
            for side, short in [('left', 'L'), ('right', 'R')]:
                mapping.update({side+k: short+v for k, v in
                                [('UpperLeg','hip'), ('LowerLeg','knee'), ('Foot','ankle'), ('Toes','toe')]})
            def joints(points):
                return {key: points[names.index(value)].tolist() for key, value in mapping.items()}
            n = len(data['points'])
            trace = dict(schema_version=1, kind='charforge-gait-trace', label='public-poses',
                         cycle_seconds=float(data['times'][-1] - data['times'][0]), periodic=False,
                         rest=joints(data['rest_points']), provenance={},
                         samples=[dict(phase=i/n, joints=joints(p)) for i,p in enumerate(data['points'])])
        scaled = copy.deepcopy(trace)
        for row in [scaled['rest'], *[s['joints'] for s in scaled['samples']]]:
            for key, value in row.items():
                row[key] = (np.array(value) * 1.7 + [3,4,5]).tolist()
        result = compare(trace, scaled)
        self.assertEqual(result['status'], 'diagnostic-not-acceptance')
        for row in result['phase_matched_error'].values():
            if row['rmse'] is not None:
                self.assertLess(row['rmse'], 1e-6)

    def test_rig_public_body_landmarks_skeleton_and_weights(self):
        with np.load(ROOT / 'garment-separation/assets/oracle.npz', allow_pickle=False) as data:
            v, t = data['body_vertices'], data['body_faces']
        height = float(np.ptp(v[:,2]))
        points, details, warnings = detect(v, t, height, {})
        bones = skeleton(points, details)
        self.assertGreaterEqual(len(bones), 20)
        self.assertTrue(all(np.isfinite(p).all() for p in points.values()))
        self.assertTrue(warnings, 'Unobserved anatomical joints must retain uncertainty')
        edges = np.unique(np.sort(np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1),axis=0)
        weights, report = diffuse(v, edges, bones, height, iterations=12)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
        self.assertEqual(report['unweighted'], 0)
        self.assertLessEqual(report['max_influences'], 8)

    def test_layers_public_body_warp_and_closed_proxy(self):
        with np.load(ROOT / 'garment-separation/assets/oracle.npz', allow_pickle=False) as data:
            body = data['body_vertices']
        lo, hi = body.min(0), body.max(0)
        anchors = np.array([[lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]],
                            [lo[0],hi[1],lo[2]], [lo[0],lo[1],hi[2]], hi])
        test = body[::500]
        np.testing.assert_allclose(tps(anchors, anchors*1.4, test), test*1.4, atol=1e-6)
        cells = np.unique(np.floor(body[::50] / .08).astype(int), axis=0)
        vertices, triangles = voxel_surface(cells, .08)
        triangles = np.array(triangles)
        edges = np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(counts == 2))
        self.assertTrue(np.isfinite(vertices).all())

    def test_secondary_public_hair_dimensions_repeat_exactly(self):
        with np.load(ROOT / 'garment-separation/assets/oracle.npz', allow_pickle=False) as data:
            hair = data['hair_vertices']; body = data['body_vertices']
        height = float(np.ptp(body[:,2])); length = float(np.ptp(hair[:,2]))
        preset = resolve_preset('hair_short_bob', dict(height_cm=height*100, mass_proxy_kg=40), length)
        root = hair[np.argmax(hair[:,2])]; tip = root - [0,0,length]
        topology = from_roots_tips([root], [tip], 3, 'fan', False)
        def simulate():
            lattice = Lattice(**topology, preset=preset)
            return np.array([lattice.step(topology['rest']+[.01*np.sin(i/8),0,0], 1/120) for i in range(48)])
        a, b = simulate(), simulate()
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(np.ptp(a[:,-1,0]), 0)

    def test_face_public_atlas_likeness_and_widening_falsifier(self):
        folder = ROOT / 'one-brow/assets/reference'
        binding = read(folder / 'binding.json'); groups = load_groups(folder, binding)
        ref = descriptors(groups['mouth']['mouth-closed'], binding['source_landmarks'])
        self.assertTrue(compare_mouth(ref, ref, 'closed')['passed'])
        altered = {**ref, 'width': ref['width'] * 1.35}
        self.assertFalse(compare_mouth(altered, ref, 'closed')['checks']['width'])
        for states in groups.values():
            if isinstance(states, dict):
                for patch in states.values():
                    self.assertEqual(patch.shape, (512,512,4))

    def test_c8_public_failed_movie_stays_failed(self):
        folder = ROOT / 'one-brow/assets/reference'; binding = read(folder / 'binding.json')
        groups = load_groups(folder, binding); reg = c8_gate.regions(groups, binding)
        m = dict(np.load(ROOT / 'one-brow/assets/maps/map-0.npz', allow_pickle=False))
        cap = cv2.VideoCapture(str(ROOT / 'one-brow/baseline/face-demo.mp4'))
        ok, frame = cap.read(); cap.release(); self.assertTrue(ok)
        crop = cv2.cvtColor(frame[56:568,:512], cv2.COLOR_BGR2RGB)
        track = read(folder / 'track.json')['frames'][0]
        skin = c8_gate.read(folder / 'feature-free.png')
        result = c8_gate.assess_features(crop, c8_gate.masks(m,reg), m, groups, track, binding, skin, m, 0)
        self.assertFalse(all(x['pass'] for x in result.values()))

    def test_preflight_public_native_rig(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'preflight.json'
            blender(ROOT / 'toolkit/preflight/preflight.py',
                    ['--input', ROOT / 'hand-plant/assets/rig.blend', '--name', 'public-rig', '--output', out])
            report = read(out)
            self.assertNotIn('error', report)
            self.assertEqual(len(report['predictions']), 8)
            self.assertTrue(report['knee_preflight_pass'])
            self.assertEqual(report['likely_failure_patterns'], [])


if __name__ == '__main__':
    unittest.main()
