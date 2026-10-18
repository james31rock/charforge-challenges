import json, hashlib
from pathlib import Path
import cv2, numpy as np
from toolkit.face.layers import write, save
MOUTHS = ['closed', 'mbp', 'small-a', 'wide-a', 'ee', 'o', 'fv', 'lth']

def profile(p):
    mask = p[:, :, 3] > 80
    ys, xs = np.where(mask)
    if len(xs) < 3:
        raise ValueError('Mouth has no measurable contour')
    x0, x1 = (xs.min(), xs.max())
    u = np.linspace(0, 1, 129)
    cols = np.arange(x0, x1 + 1)
    top = []
    bottom = []
    known = []
    for x in cols:
        yy = np.where(mask[:, x])[0]
        if len(yy):
            known.append(x)
            top.append(yy.min())
            bottom.append(yy.max())
    xp = x0 + u * (x1 - x0)
    return {'x0': float(x0), 'x1': float(x1), 'top': np.interp(xp, known, top) - 0.75, 'bottom': np.interp(xp, known, bottom) + 0.75}

def morph(patches, profiles, weights):
    active = [i for i, v in enumerate(weights) if v > 1e-08]
    if len(active) == 1:
        return patches[active[0]].copy()
    h, w = patches[0].shape[:2]
    yy, xx = np.indices((h, w))
    x0 = sum((weights[i] * profiles[i]['x0'] for i in active))
    x1 = sum((weights[i] * profiles[i]['x1'] for i in active))
    top = sum((weights[i] * profiles[i]['top'] for i in active))
    bottom = sum((weights[i] * profiles[i]['bottom'] for i in active))
    u = (xx - x0) / max(x1 - x0, 1)
    tt = np.interp(u.ravel(), np.linspace(0, 1, 129), top).reshape(h, w)
    bb = np.interp(u.ravel(), np.linspace(0, 1, 129), bottom).reshape(h, w)
    v = (yy - tt) / np.maximum(bb - tt, 1)
    rgb = np.zeros((h, w, 3), float)
    alpha = np.zeros((h, w), float)
    for i in active:
        p = profiles[i]
        sx = p['x0'] + u * (p['x1'] - p['x0'])
        st = np.interp(u.ravel(), np.linspace(0, 1, 129), p['top']).reshape(h, w)
        sb = np.interp(u.ravel(), np.linspace(0, 1, 129), p['bottom']).reshape(h, w)
        sy = st + v * (sb - st)
        source = patches[i].astype(np.float32) / 255
        source[:, :, :3] *= source[:, :, 3:4]
        warped = cv2.remap(source, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb += weights[i] * warped[:, :, :3]
        alpha += weights[i] * warped[:, :, 3]
    alpha *= (u > -0.03) & (u < 1.03)
    rgb = np.divide(rgb, np.maximum(alpha[:, :, None], 1e-08))
    rgb[alpha < 0.001] = 0
    return np.dstack((np.uint8(np.clip(rgb * 255, 0, 255)), np.uint8(np.clip(alpha * 255, 0, 255))))

def build(folder, groups, track, transport=False):
    renderer = morph
    profiler = profile
    if transport:
        from toolkit.face.mouth_transport import profile as profiler, morph as renderer
    f = Path(folder)
    patches = [groups['mouth']['mouth-' + x] for x in MOUTHS]
    profiles = [profiler(p) for p in patches]
    unique = {}
    count = 0
    for row in track['frames']:
        weights = row['mouth_weights']
        active = np.flatnonzero(np.array(weights) > 1e-07)
        if len(active) == 1:
            name = 'mouth-' + MOUTHS[active[0]]
        else:
            token = hashlib.sha256(json.dumps(weights).encode()).hexdigest()[:12]
            name = 'morph-' + token
            if name not in unique:
                unique[name] = weights
                groups['mouth'][name] = renderer(patches, profiles, weights)
                count += 1
        row['mouth_sample'] = name
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (9 * 160, 160), '#e6e9ef')
    draw = ImageDraw.Draw(sheet)
    for i in range(9):
        t = i / 8
        s = t * t * (3 - 2 * t)
        weights = [1 - s, 0, 0, s, 0, 0, 0, 0]
        p = renderer(patches, profiles, weights)
        ys, xs = np.where(np.maximum(patches[0][:, :, 3], patches[3][:, :, 3]) > 0)
        box = [max(0, int(xs.min()) - 10), max(0, int(ys.min()) - 10), int(xs.max()) + 11, int(ys.max()) + 11]
        im = Image.fromarray(p).crop(box)
        im.thumbnail((150, 120))
        tile = Image.new('RGBA', (160, 160), '#e6e9ef')
        tile.alpha_composite(im, ((160 - im.width) // 2, 25 + (120 - im.height) // 2))
        sheet.paste(tile.convert('RGB'), (i * 160, 0))
        draw.text((i * 160 + 5, 5), f'{s:.3f}', fill='black')
    method = 'separate contour/interior/shadow transport; reference line width; closed ink never stretched into cavity' if transport else 'measured lip-envelope geometric warp; premultiplied interior blend'
    sheet.save(f / 'mouth-inbetweens.png')
    write(f / 'track.json', track)
    write(f / 'mouth-morph.json', {'schema_version': 'charforge.face.morph/2' if transport else 'charforge.face.morph/1', 'method': method, 'unique_playback_inbetweens': count, 'diagnostic_sweep_inbetweens': 7, 'samples': unique, 'mouths': MOUTHS, 'timing': 'on ones; binomial easing [1,4,6,4,1]/16'})
    return count
