from __future__ import annotations
import argparse, hashlib, json, sys
sys.dont_write_bytecode = True
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)

def read(p):
    return np.array(Image.open(p).convert('RGB'))

def save(p, v):
    Image.fromarray(np.uint8(np.clip(v, 0, 255))).save(p)

def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def components(mask):
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return ([(i, stats[i], centres[i]) for i in range(1, n)], labels)

def landmarks(im):
    h, w = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    mask = (g < 105).astype(np.uint8)
    mask[:int(h * 0.34)] = 0
    mask[int(h * 0.76):] = 0
    mask[:, :int(w * 0.23)] = 0
    mask[:, int(w * 0.78):] = 0
    cs, _ = components(mask)
    candidates = []
    for i, (x, y, bw, bh, area), centre in cs:
        if area < 0.00035 * w * h or not 0.3 < bw / max(bh, 1) < 1.8:
            continue
        if bw > 0.17 * w or bh > 0.23 * h or x <= int(0.23 * w) or (x + bw >= int(0.78 * w)):
            continue
        candidates.append((centre, area, [int(x), int(y), int(bw), int(bh)]))
    pairs = []
    for l in candidates:
        for r in candidates:
            d = r[0][0] - l[0][0]
            if not 0.16 * w < d < 0.46 * w or abs(l[0][1] - r[0][1]) > 0.07 * h:
                continue
            score = np.sqrt(l[1] * r[1]) / (1 + abs(l[0][1] - r[0][1]))
            pairs.append((score, l, r))
    if not pairs:
        raise ValueError('No confident paired eyes; supply measured landmarks')
    _, l, r = max(pairs, key=lambda p: p[0])
    eyes = np.array([l[0], r[0]])
    d = float(eyes[1, 0] - eyes[0, 0])
    ey = float(eyes[:, 1].mean())
    cx = float(eyes[:, 0].mean())
    mask = (g < 135).astype(np.uint8)
    yy, xx = np.indices((h, w))
    mask *= ((abs(xx - cx) < d * 0.43) & (yy > ey + 0.34 * d) & (yy < ey + 1.28 * d)).astype(np.uint8)
    cs, _ = components(mask)
    mouth = []
    for _, (x, y, bw, bh, area), centre in cs:
        if d * 0.15 < bw < 1.15 * d and bh < 0.35 * bw and (abs(centre[0] - cx) < d * 0.17) and (area > 8):
            mouth.append((bw, centre, [int(x), int(y), int(bw), int(bh)]))
    if not mouth:
        raise ValueError('No confident closed-mouth landmark')
    _, m, mb = max(mouth, key=lambda q: q[0])
    return {'eyes': eyes.tolist(), 'mouth': m.tolist(), 'eye_boxes': [l[2], r[2]], 'mouth_box': mb, 'interocular_px': d}

def register(base, source, lm):
    h, w = base.shape[:2]
    d = lm['interocular_px']
    ey = np.mean(np.array(lm['eyes'])[:, 1])
    cx = np.mean(np.array(lm['eyes'])[:, 0])
    yy, xx = np.indices((h, w))
    stable = np.uint8(((yy < ey - d * 0.8) | (abs(xx - cx) > d * 1.05)) & (yy < lm['mouth'][1] + d * 0.4)) * 255
    stable[:8] = 0
    stable[-8:] = 0
    stable[:, :8] = 0
    stable[:, -8:] = 0
    bg = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    sg = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    p = cv2.goodFeaturesToTrack(bg, 500, 0.02, 7, mask=stable, blockSize=5)
    if p is None:
        raise ValueError('No registration landmarks')
    q, st, err = cv2.calcOpticalFlowPyrLK(bg, sg, p, None, winSize=(21, 21), maxLevel=3)
    back, bs, _ = cv2.calcOpticalFlowPyrLK(sg, bg, q, None, winSize=(21, 21), maxLevel=3)
    max_flow = min(h, w) * 0.016
    ok = (st[:, 0] > 0) & (bs[:, 0] > 0) & (np.linalg.norm(back[:, 0] - p[:, 0], axis=1) < 0.35) & (err[:, 0] < 12) & (np.linalg.norm(q[:, 0] - p[:, 0], axis=1) < max_flow)
    x = p[ok, 0]
    y = q[ok, 0]
    train = np.arange(len(x)) % 3 != 0
    if len(x) < 18:
        raise ValueError('Insufficient invariant matches')
    cv2.setRNGSeed(1017)
    m, _ = cv2.estimateAffinePartial2D(y[train], x[train], method=cv2.RANSAC, ransacReprojThreshold=2, maxIters=5000)
    if m is None:
        raise ValueError('Registration did not converge')
    e = np.linalg.norm(y @ m[:, :2].T + m[:, 2] - x, axis=1)
    rms = float(np.sqrt(np.mean(e[~train] ** 2)))
    aligned = cv2.warpAffine(source, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    return (aligned, {'source_to_base': m.tolist(), 'holdout_rms_px': rms, 'holdout_n': int((~train).sum()), 'accepted': rms <= 1.5, 'input_contract_max_flow_px': max_flow, 'observations': [{'base': a.tolist(), 'edit': b.tolist(), 'error_px': float(v), 'holdout': not bool(t)} for a, b, v, t in zip(x, y, e, train)]})

def windows(lm, shape):
    h, w = shape[:2]
    d = lm['interocular_px']
    out = {}
    ey = np.mean(np.array(lm['eyes'])[:, 1])
    for side, (x, y) in zip(('left', 'right'), lm['eyes']):
        out['eye-' + side] = [x - 0.36 * d, y - 0.32 * d, x + 0.36 * d, y + 0.34 * d]
        out['brow-' + side] = [x - 0.4 * d, y - 0.66 * d, x + 0.4 * d, y - 0.32 * d]
        out['blush-' + side] = [x - 0.32 * d, y + 0.26 * d, x + 0.32 * d, min(y + 0.56 * d, lm['mouth'][1] + 0.06 * d)]
    x, y = lm['mouth']
    out['mouth'] = [x - 0.48 * d, y - 0.2 * d, x + 0.48 * d, min(h, y + 0.42 * d)]
    return {k: [int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))] for k, (x0, y0, x1, y1) in out.items()}

def soft_box(shape, box, feather):
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    yy, xx = np.indices((h, w))
    distance = np.minimum(np.minimum(xx - x0, x1 - 1 - xx), np.minimum(yy - y0, y1 - 1 - yy))
    t = np.clip(distance / max(feather, 1), 0, 1)
    return t * t * (3 - 2 * t)

def extract(character, out):
    character = Path(character)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = read(character / 'face-grids/master.png')
    lm = landmarks(base)
    boxes = windows(lm, base.shape)
    save(out / 'base.png', base)
    regs = {}
    entries = {}
    aligned = {}
    for target in sorted((character / 'face-targets').glob('*.png')):
        source = read(target)
        if source.shape != base.shape:
            source = cv2.resize(source, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
        im, reg = register(base, source, lm)
        regs[target.stem] = reg
        aligned[target.stem] = im
    d = lm['interocular_px']
    mx, my = lm['mouth']
    extents = []
    for im in [base, *aligned.values()]:
        x0, y0, x1, y1 = map(int, [max(0, mx - d), my - 0.32 * d, min(base.shape[1], mx + d), my + 0.55 * d])
        crop = im[y0:y1, x0:x1]
        dark = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) < 140
        cs, _ = components(dark)
        valid = []
        for _, (x, y, bw, bh, area), c in cs:
            if x <= 1 or y <= 1 or x + bw >= x1 - x0 - 1 or (y + bh >= y1 - y0 - 1):
                continue
            if bw < d * 0.13 or abs(c[0] + x0 - mx) > 0.18 * d or abs(c[1] + y0 - my) > 0.32 * d:
                continue
            valid.append((area, [x + x0, y + y0, x + x0 + bw, y + y0 + bh]))
        if valid:
            extents.append(max(valid, key=lambda p: p[0])[1])
    if extents:
        e = np.array(extents)
        pad = max(4, int(d * 0.055))
        boxes['mouth'] = [int(e[:, 0].min() - pad), int(e[:, 1].min() - pad), int(e[:, 2].max() + pad), int(e[:, 3].max() + pad)]
    supports = {k: soft_box(base.shape, b, max(3, lm['interocular_px'] * 0.045)) for k, b in boxes.items()}
    for k in ('blush-left', 'blush-right'):
        for other in ('eye-left', 'eye-right', 'brow-left', 'brow-right', 'mouth'):
            supports[k] *= supports[other] <= 0
    for name, im in aligned.items():
        channels = ['mouth'] if name.startswith('mouth-') else list(boxes)
        rows = []
        for key in channels:
            alpha = supports[key]
            x0, y0, x1, y1 = boxes[key]
            rgba = np.dstack((im, alpha * 255)).astype(np.uint8)
            dest = out / 'layers' / name / (key + '.png')
            dest.parent.mkdir(parents=True, exist_ok=True)
            save(dest, rgba[y0:y1, x0:x1])
            rows.append({'feature': key, 'path': str(dest.relative_to(out)).replace('\\', '/'), 'offset': [x0, y0], 'size': [x1 - x0, y1 - y0], 'alpha': 'straight', 'sha256': sha(dest)})
        entries[name] = rows
    index = {'schema_version': 'charforge.face.layers/1', 'character_id': character.name, 'base_sha256': sha(character / 'face-grids/master.png'), 'canvas': [base.shape[1], base.shape[0]], 'landmarks': lm, 'windows': boxes, 'entries': entries, 'registration': regs, 'status': 'review_required', 'method': 'invariant LK similarity + fixed base-relative replacement supports'}
    write(out / 'layers.json', index)
    np.savez_compressed(out / 'aligned.npz', **aligned)
    proof = Image.fromarray(base)
    dr = ImageDraw.Draw(proof)
    for k, b in boxes.items():
        dr.rectangle(b, outline='magenta', width=1)
        dr.text((b[0], b[1]), k, fill='blue')
    proof.save(out / 'landmarks.png')
    return index
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--character', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = extract(a.character, a.out)
    print(json.dumps({'character': r['character_id'], 'landmarks': r['landmarks'], 'registration_max': max((x['holdout_rms_px'] for x in r['registration'].values()))}))
