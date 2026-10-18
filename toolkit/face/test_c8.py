import unittest
import numpy as np
import cv2
from toolkit.face.c8_gate import ink, components, exclusive, eye_evidence, assess_brow, camera_yaw

class C8Controls(unittest.TestCase):

    def test_clean_skin_has_zero_feature_colour(self):
        im = np.full((128, 128, 3), (245, 196, 152), np.uint8)
        self.assertEqual(int(ink(im).sum()), 0)

    def test_single_and_duplicate_brows(self):
        mask = np.zeros((128, 128), np.uint8)
        cv2.line(mask, (20, 40), (90, 32), 1, 4)
        self.assertEqual(len(components(mask, 100)), 1)
        cv2.line(mask, (20, 57), (90, 49), 1, 4)
        self.assertEqual(len(components(mask, 100)), 2)

    def test_open_eye_and_duplicate_eye(self):
        mask = np.zeros((128, 128), np.uint8)
        cv2.ellipse(mask, (38, 45), (14, 21), 0, 0, 360, 1, -1)
        self.assertEqual(len(components(mask, 100, True)), 1)
        cv2.ellipse(mask, (84, 45), (14, 21), 0, 0, 360, 1, -1)
        self.assertEqual(len(components(mask, 100, True)), 2)

    def test_eye_in_union_brow_region_is_not_a_second_brow(self):
        brow = np.zeros((128, 128), bool)
        brow[30:35, 20:90] = True
        eye = np.zeros_like(brow)
        eye[60:80, 30:70] = True
        roi = np.ones_like(brow)
        assigned = exclusive(roi, 'brow-left', {'brow-left': brow, 'eye-left': eye})
        self.assertEqual(len(components((brow | eye) & assigned, 100)), 1)
        duplicate = np.zeros_like(brow)
        duplicate[45:49, 20:90] = True
        self.assertEqual(len(components((brow | eye | duplicate) & assigned, 100)), 2)

    def test_blush_is_assigned_once_and_does_not_hide_duplicate_eye(self):
        eye = np.zeros((128, 128), bool)
        eye[20:45, 30:65] = True
        blush = np.zeros_like(eye)
        blush[85:100, 30:65] = True
        duplicate = np.zeros_like(eye)
        duplicate[55:72, 30:65] = True
        roi = exclusive(np.ones_like(eye), 'eye-left', {'eye-left': eye, 'blush-left': blush})
        self.assertEqual(len(components((eye | blush) & roi, 100, True)), 1)
        self.assertEqual(len(components((eye | blush | duplicate) & roi, 100, True)), 2)

    def test_sclera_is_part_of_one_eye(self):
        skin = np.full((128, 128, 3), (245, 190, 145), np.uint8)
        im = skin.copy()
        cv2.circle(im, (48, 60), 24, (245, 245, 245), -1)
        cv2.circle(im, (48, 60), 14, (25, 25, 25), -1)
        cv2.line(im, (24, 45), (24, 65), (20, 20, 20), 3)
        self.assertEqual(len(components(eye_evidence(im, skin), 100, True)), 1)
        cv2.ellipse(im, (108, 60), (12, 23), 0, 0, 360, (30, 30, 30), -1)
        self.assertEqual(len(components(eye_evidence(im, skin), 100, True)), 2)

    def test_depth_occluded_fragment_passes_but_unexplained_dropout_fails(self):
        mask = np.zeros((128, 128), bool)
        mask[30:33, 40:42] = True
        proof = {'source_strokes': 1, 'front_strokes': 1, 'predicted_visible_strokes': 0, 'depth_hidden_fraction': 0.85}
        self.assertTrue(assess_brow(mask, mask, 100, proof)['pass'])
        self.assertFalse(assess_brow(mask, np.zeros_like(mask), 100, {**proof, 'depth_hidden_fraction': 0})['pass'])
        self.assertFalse(assess_brow(np.zeros_like(mask), mask, 100, proof)['pass'])
        self.assertFalse(assess_brow(mask, mask, 100, {**proof, 'source_strokes': 2})['pass'])

    def test_occlusion_cannot_excuse_a_duplicate_or_a_missing_visible_brow(self):
        expected = np.zeros((128, 128), bool)
        expected[30:34, 20:90] = True
        proof = {'source_strokes': 1, 'front_strokes': 1, 'predicted_visible_strokes': 1, 'depth_hidden_fraction': 0.85}
        self.assertTrue(assess_brow(expected, expected, 100, proof)['pass'])
        self.assertFalse(assess_brow(np.zeros_like(expected), expected, 100, proof)['pass'])
        duplicate = expected.copy()
        duplicate[48:52, 20:90] = True
        self.assertFalse(assess_brow(duplicate, expected, 100, proof)['pass'])

    def test_occlusion_identity_is_continuous_across_readability_width(self):
        expected = np.zeros((128, 128), bool)
        expected[30:34, 20:32] = True
        observed = np.zeros_like(expected)
        observed[31:34, 22:32] = True
        proof = {'source_strokes': 1, 'front_strokes': 1, 'predicted_visible_strokes': 1, 'depth_hidden_fraction': 0.53}
        self.assertTrue(assess_brow(observed, expected, 145, proof)['pass'])
        self.assertFalse(assess_brow(np.zeros_like(observed), expected, 145, proof)['pass'])
        self.assertFalse(assess_brow(observed, expected, 145, {**proof, 'depth_hidden_fraction': 0})['pass'])

    def test_continuous_turn_has_smooth_endpoints_and_no_quantized_steps(self):
        angles = [camera_yaw(i / 24, True) for i in range(240)]
        self.assertEqual(angles[0], 0)
        self.assertEqual(angles[-1], 0)
        self.assertTrue(any((abs(a / 2.5 - round(a / 2.5)) > 0.1 for a in angles)))
        self.assertLess(max((abs(b - a) for a, b in zip(angles, angles[1:]))), 2)
        for t in (6.7, 7.95, 8.4, 9.6):
            self.assertLess(abs(camera_yaw(t + 0.0001, True) - camera_yaw(t - 0.0001, True)), 1e-05)

    def test_hair_split_fragments_keep_one_original_stroke_identity(self):
        expected = np.zeros((128, 128), bool)
        expected[30, 40] = True
        proof = {'source_strokes': 1, 'unoccluded_front_strokes': 1, 'front_strokes': 2, 'depth_hidden_fraction': 0.75, 'native_hair_hidden_fraction': 0.4}
        self.assertTrue(assess_brow(np.zeros_like(expected), expected, 100, proof)['pass'])
        self.assertFalse(assess_brow(np.zeros_like(expected), expected, 100, {**proof, 'unoccluded_front_strokes': 2})['pass'])
        self.assertFalse(assess_brow(np.zeros_like(expected), expected, 100, {**proof, 'depth_hidden_fraction': 0, 'native_hair_hidden_fraction': 0})['pass'])

    def test_present_fragment_does_not_require_majority_occlusion(self):
        expected = np.zeros((128, 128), bool)
        expected[30:36, 40:46] = True
        proof = {'source_strokes': 1, 'unoccluded_front_strokes': 1, 'front_strokes': 1, 'depth_hidden_fraction': 0.35, 'native_hair_hidden_fraction': 0.19}
        self.assertTrue(assess_brow(expected, expected, 145, proof)['pass'])
        self.assertFalse(assess_brow(np.zeros_like(expected), expected, 145, proof)['pass'])
        self.assertFalse(assess_brow(expected, expected, 145, {**proof, 'depth_hidden_fraction': 0, 'native_hair_hidden_fraction': 0})['pass'])
        self.assertFalse(assess_brow(expected, expected, 145, {**proof, 'source_strokes': 2})['pass'])
        duplicate = expected.copy()
        duplicate[50:56, 40:46] = True
        self.assertFalse(assess_brow(duplicate, expected, 145, proof)['pass'])
if __name__ == '__main__':
    unittest.main()
