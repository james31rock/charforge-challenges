import json, wave
from pathlib import Path
import numpy as np
from toolkit.face.timing import make_track, RHUBARB
from toolkit.face.mouth_morph import MOUTHS

def make_smooth_track(audio, cues, duration=10.0, fps=24, step=2, seed=1017):
    old = make_track(audio, cues, duration, fps, step, seed)
    n = old['frame_count']
    with wave.open(str(audio), 'rb') as w:
        rate = w.getframerate()
        arr = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, w.getnchannels()).mean(1) / 32768
    cue_rows = json.loads(Path(cues).read_text())['mouthCues']
    raw = np.zeros((n, len(MOUTHS)))
    rms = []
    for i in range(n):
        a = int(i / fps * rate)
        b = min(len(arr), int((i + 1) / fps * rate))
        level = float(np.sqrt(np.mean(arr[a:b] ** 2))) if b > a else 0
        rms.append(level)
        t = (i + 0.5) / fps
        cue = next((c['value'] for c in cue_rows if c['start'] <= t < c['end']), 'X')
        name = RHUBARB[cue] if level >= old['silence_threshold_rms'] else 'closed'
        raw[i, MOUTHS.index(name)] = 1
    kernel = np.array([1, 4, 6, 4, 1]) / 16
    weights = np.stack([np.convolve(np.pad(raw[:, k], (2, 2), mode='edge'), kernel, 'valid') for k in range(len(MOUTHS))], axis=1)
    frames = []
    for i in range(n):
        row = dict(old['frames'][min(i // step, len(old['frames']) - 1)])
        row.update({'frame': i + 1, 'time_s': i / fps, 'mouth': MOUTHS[int(np.argmax(raw[i]))], 'mouth_weights': weights[i].tolist(), 'rms': rms[i]})
        frames.append(row)
    old.update({'schema_version': 'charforge.face.track/2', 'exposure_frames': 1, 'eye_exposure_frames': step, 'frames': frames, 'method': 'per-frame phonetic/RMS targets; eased geometric mouth coarticulation; stepped eyes and brows', 'mouth_easing_kernel': kernel.tolist(), 'closed_mouth_frames': int((weights[:, 0] == 1).sum())})
    return old
