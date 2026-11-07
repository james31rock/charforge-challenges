"""Circular-heading regressions; run as toolkit.gait.test_circular."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from . import core
from .test_core import fixture


class CircularHeadingTests(unittest.TestCase):
    def test_real_trace_crosses_heading_branch_cut(self):
        traces = [fixture(), fixture()]
        for trace, degrees in zip(traces, [179, -179]):
            theta = math.radians(degrees)
            for sample in trace['samples']:
                joints = sample['joints']
                joints['leftToes'] = (np.array(joints['leftFoot']) +
                    .16 * np.array([math.sin(theta), -math.cos(theta), 0])).tolist()
        result = core.compare(*traces)['phase_matched_error']['left_foot_progression_deg']
        self.assertAlmostEqual(result['rmse'], 2)
        self.assertEqual(result['coverage'], 1)

    def test_midpoint_and_periodic_seam_use_short_arc(self):
        result = core.phase_values([179, -179], 0, np.array([.25, .75]), circular=True)
        np.testing.assert_allclose(abs(result), [180, 180])

    def test_missing_segment_not_interpolated(self):
        result = core.phase_values([170, None, -170], 0,
                                   np.array([0, 1/6, 1/3, .5, 2/3]), circular=True)
        self.assertEqual(result[0], 170)
        self.assertEqual(result[-1], -170)
        self.assertTrue(np.isnan(result[1:4]).all())

    def test_antipodal_arc_unavailable_but_endpoints_retained(self):
        result = core.phase_values([0, 180], 0, np.array([0, .25, .5, .75]), circular=True)
        self.assertEqual(result[0], 0)
        self.assertEqual(abs(result[2]), 180)
        self.assertTrue(np.isnan(result[[1, 3]]).all())

    def test_phase_origin_and_full_rotations(self):
        result = core.phase_values([539, 181], .25, np.array([0, .5]), circular=True)
        np.testing.assert_allclose(abs(result), [180, 180])

    def test_scalar_interpolation_stays_linear(self):
        result = core.phase_values([179, -179], 0, np.array([.25, .75]))
        np.testing.assert_array_equal(result, [0, 0])

    def test_all_heading_channels_wrap_but_hinge_angle_does_not(self):
        # Isolate routing and residual math; the real-trace case above exercises measure().
        keys = ['left_foot_progression_deg', 'right_knee_progression_deg',
                'left_knee_plane_pelvis_deg', 'left_knee_deg']
        measurements = [{'curves': {key: [value, value] for key in keys},
                         'phase_origin': 0} for value in [179, -179]]
        with patch.object(core, 'measure', side_effect=measurements):
            result = core.compare({}, {})['phase_matched_error']
        for key in keys[:-1]:
            self.assertAlmostEqual(result[key]['rmse'], 2)
        self.assertAlmostEqual(result[keys[-1]]['rmse'], 358)

    def test_nonperiodic_capture_still_masks_seam(self):
        trace = fixture()
        trace['periodic'] = False
        result = core.compare(trace, copy.deepcopy(trace))['phase_matched_error']
        row = result['left_foot_progression_deg']
        self.assertAlmostEqual(row['rmse'], 0)
        self.assertGreater(row['coverage'], .9)
        self.assertLess(row['coverage'], 1)


if __name__ == '__main__':
    unittest.main()
