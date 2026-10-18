import argparse, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw
from toolkit.face.layers import read, write, sha, landmarks, components
BOUNDS = {'closed': (0.78, 1.12, 0.02, 0.23), 'mbp': (0.65, 1.12, 0.02, 0.24), 'happy': (0.85, 1.18, 0.03, 0.3), 'neutral': (0.78, 1.12, 0.02, 0.23), 'worried': (0.65, 1.12, 0.02, 0.28), 'sad': (0.65, 1.12, 0.02, 0.3), 'determined': (0.65, 1.12, 0.02, 0.24), 'surprised': (0.23, 0.65, 0.1, 0.6), 'small-a': (0.55, 1.1, 0.1, 0.48), 'wide-a': (0.72, 1.18, 0.25, 0.72), 'ee': (0.8, 1.18, 0.06, 0.36), 'o': (0.28, 0.65, 0.18, 0.6), 'fv': (0.65, 1.12, 0.04, 0.35), 'lth': (0.6, 1.12, 0.08, 0.48)}

def identify(im, lm):
    h, w = im.shape[:2]
    yy, xx = np.indices((h, w))
    d = lm['interocular_px']
    mx, my = lm['mouth']
    gray = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2GRAY)
    domain = (abs(xx - mx) < 0.9 * d) & (yy > my - 0.3 * d) & (yy < my + 0.85 * d)
    dark = (gray < 145) & domain
    if im.shape[2] == 4:
        dark &= im[:, :, 3] > 100
    cs, labels = components(dark)
    valid = [(s[4], i) for i, s, c in cs if s[2] > 0.13 * d and abs(c[0] - mx) < 0.2 * d and (abs(c[1] - my) < 0.5 * d) and (s[1] > my - 0.3 * d + 1)]
    if not valid:
        raise ValueError('No isolated mouth component')
    mask = labels == max(valid)[1]
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return (mask, filled.astype(bool))

def descriptors(im, lm):
    mask, filled = identify(im, lm)
    ys, xs = np.where(filled)
    x0, x1 = (int(xs.min()), int(xs.max()))
    y0, y1 = (int(ys.min()), int(ys.max()))
    width = x1 - x0 + 1
    d = lm['interocular_px']
    rgb = im[:, :, :3].astype(float)
    ink = mask & (rgb[:, :, 1] / np.maximum(rgb[:, :, 0], 1) > 0.43)
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 0.01) & (dt >= 1)
    widths = 2 * dt[ridge] - 1
    line = float(np.median(widths)) if len(widths) else 0
    central = range(round(x0 + 0.3 * width), round(x0 + 0.7 * width))
    top = []
    for x in central:
        yy = np.where(filled[:, x])[0]
        if len(yy):
            top.append(yy[0])
    left_y = float(np.median(np.where(filled[:, x0:x0 + max(2, round(0.08 * width))])[0]))
    right_y = float(np.median(np.where(filled[:, x1 - max(2, round(0.08 * width)) + 1:x1 + 1])[0]))
    top_y = float(np.median(top))
    white = filled & (rgb.min(2) > 155) & (rgb.max(2) - rgb.min(2) < 45)
    return {'box': [x0, y0, x1 + 1, y1 + 1], 'width': width, 'height': y1 - y0 + 1, 'line_width': line, 'center_x': (x0 + x1) / 2, 'upper_lip_y': top_y, 'corner_y': [left_y, right_y], 'upper_lip_curvature': (top_y - (left_y + right_y) / 2) / width, 'white_area_fraction': float(white.sum() / max(1, filled.sum())), 'interocular': d}

def compare(m, ref, state):
    lo, hi, hlo, hhi = BOUNDS[state]
    wr = m['width'] / ref['width']
    hr = m['height'] / ref['width']
    lr = m['line_width'] / max(ref['line_width'], 0.01)
    checks = {'width': lo <= wr <= hi, 'opening': hlo <= hr <= hhi, 'line_weight': 0.58 <= lr <= 1.55, 'cheek_center': abs(m['center_x'] - ref['center_x']) / ref['interocular'] <= 0.055, 'upper_lip_anchor': abs(m['upper_lip_y'] - ref['upper_lip_y']) / ref['interocular'] <= 0.14, 'teeth_not_white_block': m['white_area_fraction'] <= 0.58 if state == 'ee' else True}
    return {'passed': all(checks.values()), 'checks': checks, 'width_ratio': wr, 'height_per_reference_width': hr, 'line_width_ratio': lr, 'center_offset_interocular': (m['center_x'] - ref['center_x']) / ref['interocular'], 'upper_lip_offset_interocular': (m['upper_lip_y'] - ref['upper_lip_y']) / ref['interocular'], 'descriptors': m}

def audit(reference, sources, out):
    base = read(reference)
    lm = landmarks(base)
    ref = descriptors(base, lm)
    rows = []
    tiles = []
    for name, im in sources.items():
        state = name.removeprefix('expression-').removeprefix('mouth-')
        m = descriptors(im, lm)
        r = compare(m, ref, state)
        r['state'] = name
        rows.append(r)
        tiles.append((name, im, r['passed']))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'charforge.mouth-likeness/1', 'gate_sha256': sha(__file__), 'reference_sha256': sha(reference), 'landmarks': lm, 'reference': ref, 'limits': {'state_bounds': BOUNDS, 'line_width_ratio': [0.58, 1.55], 'cheek_center_max_interocular': 0.055, 'upper_lip_max_interocular': 0.14, 'ee_max_white_area': 0.58}, 'passed': all((r['passed'] for r in rows)), 'records': rows, 'scope': 'Measured mouth style constraints; human likeness review still required.'}
    write(out / 'mouth-likeness.json', result)
    d = lm['interocular_px']
    mx, my = lm['mouth']
    box = tuple(map(round, [mx - 1.0 * d, my - 0.5 * d, mx + 1.0 * d, my + 1.0 * d]))
    sheet = Image.new('RGB', (5 * 256, (len(tiles) + 1 + 4) // 5 * 230), '#e5e8ed')
    draw = ImageDraw.Draw(sheet)
    for i, (name, im, ok) in enumerate([('REFERENCE', base, True), *tiles]):
        tile = Image.fromarray(im).convert('RGBA').crop(box)
        tile.thumbnail((250, 190), Image.Resampling.LANCZOS)
        plate = Image.new('RGBA', tile.size, '#efb17c')
        plate.alpha_composite(tile)
        x = i % 5 * 256
        y = i // 5 * 230
        sheet.paste(plate.convert('RGB'), (x, y + 30))
        draw.text((x + 5, y + 8), name + (' PASS' if ok else ' FAIL'), fill='black' if ok else '#ba1111')
    sheet.save(out / 'source-mouths-zoom.png')
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--sources', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = audit(a.reference, {p.stem: read(p) for p in sorted(Path(a.sources).glob('*.png'))}, a.out)
    print(json.dumps({'passed': r['passed'], 'failed': [x['state'] for x in r['records'] if not x['passed']]}))
