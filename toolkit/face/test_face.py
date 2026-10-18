import json, tempfile, unittest, wave
from pathlib import Path
import numpy as np
from toolkit.face.timing import make_track
from toolkit.face.surface import coordinates, render
from toolkit.face.timing_v2 import make_smooth_track
from toolkit.face.mouth_morph import morph, profile

class FaceContracts(unittest.TestCase):

    def test_mouth_targets_are_eased_on_every_frame_and_eyes_stay_on_twos(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            audio = p / 'tone.wav'
            rate = 16000
            samples = np.int16(np.sin(np.arange(10 * rate) * 2 * np.pi * 220 / rate) * 12000)
            with wave.open(str(audio), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(samples.tobytes())
            cues = p / 'cues.json'
            cues.write_text(json.dumps({'mouthCues': [{'start': 3, 'end': 5, 'value': 'D'}]}))
            t = make_smooth_track(audio, cues)
            self.assertEqual(len(t['frames']), 240)
            weights = np.array([r['mouth_weights'] for r in t['frames']])
            np.testing.assert_allclose(weights.sum(1), 1)
            self.assertLessEqual(abs(np.diff(weights[:, 3])).max(), 0.375)
            self.assertGreater(len(np.unique(weights[:, 3])), 3)
            self.assertTrue((weights[:60, 0] == 1).all())
            self.assertTrue((weights[144:, 0] == 1).all())
            self.assertTrue(all((t['frames'][i]['eyes'] == t['frames'][i + 1]['eyes'] for i in range(0, 240, 2))))

    def test_morph_moves_one_contour_and_preserves_endpoint_art(self):
        import cv2
        a = np.zeros((80, 120, 4), np.uint8)
        b = a.copy()
        a[:, :, :3] = [60, 25, 15]
        b[:, :, :3] = [100, 30, 20]
        cv2.ellipse(a, (60, 40), (35, 2), 0, 0, 360, (60, 25, 15, 255), -1)
        cv2.ellipse(b, (60, 40), (35, 22), 0, 0, 360, (100, 30, 20, 255), -1)
        profiles = [profile(a), profile(b)]
        np.testing.assert_array_equal(morph([a, b], profiles, [1, 0]), a)
        np.testing.assert_array_equal(morph([a, b], profiles, [0, 1]), b)
        heights = []
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            im = morph([a, b], profiles, [1 - t, t])
            mask = im[:, :, 3] > 128
            ys, xs = np.where(mask)
            heights.append(np.ptp(ys))
            self.assertEqual(cv2.connectedComponents(mask.astype(np.uint8))[0], 2)
        self.assertTrue(all((a < b for a, b in zip(heights, heights[1:]))))

    def test_silence_closes_even_wrong_phonetic_cue_and_seed_repeats(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            audio = p / 'silent.wav'
            with wave.open(str(audio), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(np.zeros(160000, dtype='<i2').tobytes())
            cues = p / 'cues.json'
            cues.write_text(json.dumps({'mouthCues': [{'start': 0, 'end': 10, 'value': 'D'}]}))
            for step in (2, 3):
                a = make_track(audio, cues, step=step)
                b = make_track(audio, cues, step=step)
                self.assertEqual(a, b)
                self.assertEqual({r['mouth'] for r in a['frames']}, {'closed'})
                self.assertEqual(len(a['frames']), 240 // step)
                self.assertGreaterEqual(len(a['blink_start_frames']), 2)
                self.assertEqual(len(set((r['expression'] for r in a['frames']))), 3)
                self.assertTrue(all(((r['frame'] - 1) % step == 0 for r in a['frames'])))

    def test_projection_is_a_rest_surface_attribute_not_camera_facing(self):
        world = np.array([[[0.1, -0.2, 1.4]]])
        b = {'canvas': [512, 512], 'camera': {'center_x': 0, 'center_z': 1.4, 'span': 0.5}, 'front_to_source': [[1, 0, 0], [0, 1, 0]]}
        a = coordinates({'world': world, 'yaw': 0}, b)
        c = coordinates({'world': world, 'yaw': 35}, b)
        np.testing.assert_array_equal(a, c)
        self.assertAlmostEqual(a[0][0, 0], 358.4)

    def test_transparent_layers_never_change_original_albedo(self):
        r = {'color': np.full((2, 2, 3), 123, np.uint8), 'world': np.zeros((2, 2, 3)), 'normal': np.tile([0, -1, 0], (2, 2, 1)), 'ids': np.zeros((2, 2), int)}
        r['world'][:, :, 1] = -0.1
        b = {'canvas': [2, 2], 'camera': {'center_x': 0, 'center_z': 0, 'span': 1}, 'front_to_source': [[1, 0, 0], [0, 1, 0]]}
        z = np.zeros((2, 2, 4), np.uint8)
        groups = {k: {'expression-neutral': z} for k in ('base', 'blush-left', 'blush-right', 'eye-left', 'eye-right', 'brow-left', 'brow-right')}
        groups['mouth'] = {'mouth-closed': z}
        im, qa = render(r, b, groups, shade=False)
        np.testing.assert_array_equal(im, r['color'])
        self.assertEqual(qa['outside_support_max_byte_delta'], 0)

    def test_back_of_head_cannot_receive_face_art(self):
        r = {'color': np.full((2, 2, 3), 123, np.uint8), 'world': np.zeros((2, 2, 3)), 'normal': np.tile([0, 1, 0], (2, 2, 1)), 'ids': np.zeros((2, 2), int)}
        r['world'][:, :, 1] = 0.1
        b = {'canvas': [2, 2], 'camera': {'center_x': 0, 'center_z': 0, 'span': 1}, 'front_to_source': [[1, 0, 0], [0, 1, 0]]}
        z = np.full((2, 2, 4), 255, np.uint8)
        groups = {k: {'expression-neutral': z} for k in ('base', 'blush-left', 'blush-right', 'eye-left', 'eye-right', 'brow-left', 'brow-right')}
        groups['mouth'] = {'mouth-closed': z}
        im, _ = render(r, b, groups, shade=False)
        np.testing.assert_array_equal(im, r['color'])

    def test_blink_keeps_cheek_art_when_source_expression_is_already_closed(self):
        r = {'color': np.full((2, 2, 3), 123, np.uint8), 'world': np.zeros((2, 2, 3)), 'normal': np.tile([0, -1, 0], (2, 2, 1)), 'ids': np.zeros((2, 2), int)}
        r['world'][:, :, 1] = -0.1
        b = {'canvas': [2, 2], 'camera': {'center_x': 0, 'center_z': 0, 'span': 1}, 'front_to_source': [[1, 0, 0], [0, 1, 0]], 'native_closed_expressions': {'eye-left': ['happy'], 'eye-right': ['happy']}}
        clear = np.zeros((2, 2, 4), np.uint8)
        native = np.full((2, 2, 4), 255, np.uint8)
        native[:, :, :3] = [230, 110, 110]
        blink = native.copy()
        blink[:, :, :3] = [240, 210, 190]
        groups = {k: {'expression-happy': clear} for k in ('base', 'blush-left', 'blush-right', 'brow-left', 'brow-right')}
        groups['mouth'] = {'mouth-closed': clear}
        for key in ('eye-left', 'eye-right'):
            groups[key] = {'expression-happy': native, 'eyes-closed': blink}
        opened, _ = render(r, b, groups, expression='happy', shade=False)
        closed, _ = render(r, b, groups, expression='happy', eyes='closed', shade=False)
        np.testing.assert_array_equal(opened, closed)
if __name__ == '__main__':
    unittest.main()
