import argparse, json, random, subprocess, wave
from pathlib import Path
import numpy as np
from toolkit.face.layers import write, sha
RHUBARB = {'X': 'closed', 'A': 'mbp', 'B': 'ee', 'C': 'small-a', 'D': 'wide-a', 'E': 'o', 'F': 'o', 'G': 'fv', 'H': 'lth'}

def make_track(audio, cues, duration=10.0, fps=24, step=2, seed=1017):
    if duration <= 0 or fps <= 0 or step not in (2, 3):
        raise ValueError('Positive duration/fps and exposure of 2 or 3 frames required')
    with wave.open(str(audio), 'rb') as w:
        if w.getsampwidth() != 2:
            raise ValueError('Use PCM16 WAV audio; convert with ffmpeg before running S5')
        rate = w.getframerate()
        arr = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float) / 32768
        arr = arr.reshape(-1, w.getnchannels()).mean(axis=1)
    n = int(round(duration * fps))
    cue_rows = json.loads(Path(cues).read_text())['mouthCues']
    rng = random.Random(seed)
    blinks = []
    t = rng.uniform(1.0, 1.7)
    while t < duration - 0.4:
        start = int(round(t * fps / step)) * step
        blinks.append(start)
        t += rng.uniform(2.4, 3.8)
    frames = []
    rms = []
    for f in range(0, n, step):
        a = int(f / fps * rate)
        b = int(min(len(arr), (f + step) / fps * rate))
        rms.append(float(np.sqrt(np.mean(arr[a:b] ** 2))) if b > a else 0)
    floor = max(0.002, max(rms, default=0) * 0.025)
    for j, f in enumerate(range(0, n, step)):
        t = (f + step / 2) / fps
        cue = next((c['value'] for c in cue_rows if c['start'] <= t < c['end']), 'X')
        mouth = RHUBARB[cue]
        if rms[j] < floor:
            mouth = 'closed'
        eyes = 'open'
        for start in blinks:
            rel = f - start
            if rel in (0, 2 * step):
                eyes = 'half'
            if rel == step:
                eyes = 'closed'
        expression = 'neutral' if t < 3.3 else 'happy' if t < 6.6 else 'worried'
        frames.append({'frame': f + 1, 'time_s': f / fps, 'mouth': mouth, 'eyes': eyes, 'brows': expression, 'expression': expression, 'rms': rms[j], 'phonetic_cue': cue})
    return {'schema_version': 'charforge.face.track/1', 'fps': fps, 'duration_s': duration, 'frame_count': n, 'exposure_frames': step, 'seed': seed, 'audio_sha256': sha(audio), 'phonetic_cues_sha256': sha(cues), 'method': 'Rhubarb phonetic cues + RMS silence closure; constant exposure holds', 'limitations': ['Sung phonemes are approximate; review against audio.', 'No OO target in S1: Rhubarb F maps to rounded O.'], 'silence_threshold_rms': floor, 'blink_start_frames': [x + 1 for x in blinks], 'expression_changes_s': [3.3, 6.6], 'frames': frames}
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--cues', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--step', type=int, default=2)
    a = p.parse_args()
    write(a.out, make_track(a.audio, a.cues, step=a.step))
