import unittest
import numpy as np
from toolkit.layers.core import tps, fit_hip_from_leg, validate_input, voxel_surface

class FitTests(unittest.TestCase):

    def test_diagonal_voxels_have_closed_manifold_surface(self):
        for cells in [[(0, 0, 0)], [(0, 0, 0), (1, 1, 0)], [(0, 0, 0), (1, 1, 1)], [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]]:
            v, t = voxel_surface(cells, 0.01)
            t = np.array(t)
            edges = np.sort(np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            self.assertTrue(np.all(counts == 2))
            self.assertGreater(np.einsum('ij,ij->i', v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6, 0)

    def test_affine_and_scale_equivariance(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
        a = np.array([[2, 0.1, 0], [0, 0.6, 0.2], [0.1, 0, 3]])
        dst = src @ a + np.array([0.2, 0.3, 0.4])
        v = np.array([[0.2, 0.4, 0.6], [-0.1, 0.3, 0.7]])
        np.testing.assert_allclose(tps(src, dst, v), v @ a + [0.2, 0.3, 0.4], atol=1e-08)
        np.testing.assert_allclose(tps(src * 7, dst * 7, v * 7), tps(src, dst, v) * 7, atol=1e-08)

    def test_hip_ignores_garment_and_scales(self):
        k = np.array([0.057, -0.001, 0.254])
        a = np.array([0.072, 0.001, 0.068])
        x = fit_hip_from_leg(k, a, 0.426)
        self.assertLess(x, 0.057)
        self.assertAlmostEqual(fit_hip_from_leg(k * 3, a * 3, 1.278), x * 3)
        self.assertAlmostEqual(fit_hip_from_leg(k * [-1, 1, 1], a * [-1, 1, 1], 0.426), -x)

    def test_invalid_inputs(self):
        base = dict(source='mesh.fbx', mpfb_base='base.obj', out='runs/example', height_cm=105)
        for h in [None, 0, -1, float('nan'), True]:
            with self.assertRaises(ValueError):
                validate_input({**base, 'height_cm': h})
        with self.assertRaises(ValueError):
            validate_input({**base, 'source': 'private/mesh.fbx'})
        with self.assertRaises(ValueError):
            validate_input({**base, 'character_id': 'reference-rig', 'out': 'runs/fixture-a'})
        self.assertEqual(validate_input({**base, 'character_id': 'fixture-a', 'out': 'runs/fixture-a'})['character_id'], 'fixture-a')
if __name__ == '__main__':
    unittest.main()
