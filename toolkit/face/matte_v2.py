import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from toolkit.face.layers import read, save, write, components, soft_box

def skin_pixels(im):
    r, g, b = np.moveaxis(im.astype(float), -1, 0)
    ratio = (g - b) / np.maximum(r - b, 1)
    return (r > 130) & (r - g > 8) & (r - b > 28) & (ratio > 0.2) & (ratio < 0.8)

def fill(mask):
    contours, _ = cv2.findContours(np.uint8(mask) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, contours, -1, 1, cv2.FILLED)
    return out.astype(bool)

def main_region(mask, point):
    cs, labels = components(mask)
    x, y = point
    if not cs:
        return np.zeros(mask.shape, bool)
    eligible = [(np.linalg.norm(c - [x, y]) / np.sqrt(max(1, s[4])), i) for i, s, c in cs if s[4] > 8]
    return labels == min(eligible)[1] if eligible else np.zeros(mask.shape, bool)

def reconstruct(folder):
    f = Path(folder)
    idx = json.loads((f / 'layers.json').read_text())
    lm = idx['landmarks']
    base = read(f / 'base.png')
    ims = dict(np.load(f / 'aligned.npz'))
    h, w = base.shape[:2]
    yy, xx = np.indices((h, w))
    d = lm['interocular_px']
    mx, my = lm['mouth']
    ey = np.mean(np.array(lm['eyes'])[:, 1])
    stack = np.stack([base, *ims.values()])
    valid = np.stack([skin_pixels(im) for im in stack])
    anatomy = (abs(xx - mx) < 1.18 * d) & (yy > ey - 1.05 * d) & (yy < my + 0.6 * d)
    union = valid.any(0) & anatomy
    region = main_region(union, (mx, ey + 0.35 * d))
    region = fill(region)
    valid = np.stack([cv2.erode(v.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for v in valid])
    values = np.where(valid[:, :, :, None], stack, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        skin = np.nanmedian(values, axis=0)
    missing = ~np.isfinite(skin[:, :, 0])
    skin = np.nan_to_num(skin).astype(np.uint8)
    weight = ~missing & region
    radius = d * 0.13
    den = cv2.GaussianBlur(weight.astype(float), (0, 0), radius)
    skin_smooth = np.stack([cv2.GaussianBlur(skin[:, :, ch] * weight, (0, 0), radius) / np.maximum(den, 1e-06) for ch in range(3)], axis=-1)
    skin_smooth = np.uint8(np.clip(skin_smooth, 0, 255))
    env = np.zeros((h, w), bool)
    for cx, cy in lm['eyes']:
        env |= (abs(xx - cx) < d * 0.62) & (yy > cy - d * 0.95) & (yy < cy + d * 0.42)
    env |= (abs(xx - mx) < d * 0.95) & (yy > my - d * 0.3) & (yy < my + d * 0.55)
    erase = env & region
    clean = base.copy()
    clean[region] = skin_smooth[region]
    fade = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    fade = np.clip(fade / max(8, d * 0.15), 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    groups = {k: {} for k in ['base', 'eye-left', 'eye-right', 'brow-left', 'brow-right', 'blush-left', 'blush-right', 'mouth']}
    metrics = []
    masks = {}
    for name, im in ims.items():
        delta = np.linalg.norm(im.astype(float) - skin_smooth.astype(float), axis=2)
        non_skin = (delta > 35) & region
        selected = {}
        if name.startswith('expression-'):
            for side, (cx, cy) in zip(['left', 'right'], lm['eyes']):
                gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                ink = (gray < 150) & region & (abs(xx - cx) < d * 0.55) & (yy > cy - d * 0.95) & (yy < cy - d * 0.12)
                span = max(5, int(d * 0.045))
                span += 1 - span % 2
                opened = np.zeros((h, w), np.uint8)
                for slope in (-0.6, -0.3, 0, 0.3, 0.6):
                    kernel = np.zeros((span, span), np.uint8)
                    for kx in range(span):
                        kernel[int(round(span // 2 + slope * (kx - span // 2))), kx] = 1
                    opened |= cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                cs, labels = components(opened)
                candidates = []
                for label, (x, y, bw, bh, area), cent in cs:
                    if bw > d * 0.18 and bw > 1.18 * bh and (cent[1] < cy - d * 0.18) and (abs(cent[0] - cx) < d * 0.25):
                        score = area * np.exp(-0.5 * ((cent[1] - (cy - d * 0.5)) / (d * 0.15)) ** 2)
                        candidates.append((score, label, bh))
                brow = np.zeros((h, w), bool)
                split = int(cy - d * 0.32)
                if candidates:
                    _, label, bh = max(candidates)
                    seed = labels == label
                    py, px = np.where(seed)
                    brow = (cv2.dilate(np.uint8(seed), np.ones((5, 5), np.uint8)) > 0) & ink
                    brow = main_region(brow, (float(px.mean()), float(py.mean())))
                    if bh > d * 0.4:
                        py = np.where(brow.any(1))[0]
                        lo = int(py[0] + d * 0.18)
                        hi = int(cy - d * 0.17)
                        ys = np.arange(lo, hi + 1)
                        density = brow.sum(1)
                        split = int(ys[np.argmin(density[ys] + 0.05 * abs(ys - (cy - d * 0.28)))])
                        brow &= yy < split
                brow = fill(brow)
                eye_domain = (abs(xx - cx) < d * 0.43) & (yy > cy - d * 0.38) & (yy < cy + d * 0.36) & ~brow
                eye = fill(main_region(non_skin & eye_domain, (cx, cy)))
                selected['eye-' + side] = eye
                selected['brow-' + side] = brow
                metrics.append({'source': name, 'side': side, 'split_y': split, 'eye_pixels': int(eye.sum()), 'brow_pixels': int(brow.sum())})
        mouth_domain = (abs(xx - mx) < d * 0.92) & (yy > my - d * 0.31) & (yy < my + d * 0.55)
        mouth = main_region(non_skin & mouth_domain, (mx, my))
        selected['mouth'] = fill(mouth)
        for key, mask in selected.items():
            outer = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            alpha = cv2.GaussianBlur(outer.astype(float), (0, 0), 0.55)
            alpha *= region
            a = alpha[:, :, None]
            rgb = np.where(a > 0.02, (im.astype(float) - skin_smooth * (1 - a)) / np.maximum(a, 0.02), 0)
            groups[key][name] = np.dstack((np.clip(rgb, 0, 255), np.round(alpha * 255))).astype(np.uint8)
            masks[name, key] = mask
        if name.startswith('expression-'):
            groups['base'][name] = np.dstack((clean, np.round(fade * 255))).astype(np.uint8)
            occupied = np.zeros((h, w), bool)
            for mask in selected.values():
                occupied |= cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            for side, (cx, cy) in zip(['left', 'right'], lm['eyes']):
                diff = np.linalg.norm(im.astype(float) - base.astype(float), axis=2)
                red = im[:, :, 0].astype(float) - im[:, :, 1] - (base[:, :, 0].astype(float) - base[:, :, 1])
                a = np.clip((red - 5) / 18, 0, 1) * (diff > 10) * (abs(xx - cx) < d * 0.4) * (yy > cy + d * 0.18) * (yy < cy + d * 0.65) * region * ~occupied
                a = cv2.GaussianBlur(a, (0, 0), 0.7) * ~occupied
                groups['blush-' + side][name] = np.dstack((im, np.round(a * 255))).astype(np.uint8)
    native = {}
    blink_rows = []
    for side, (cx, cy) in zip(['left', 'right'], lm['eyes']):
        key = 'eye-' + side
        native[key] = []
        eye = groups[key]['expression-neutral']
        happy = groups[key]['expression-happy']
        for name, p in groups[key].items():
            py, px = np.where(p[:, :, 3] > 128)
            if len(py) and np.ptp(py) < d * 0.27:
                native[key].append(name.removeprefix('expression-'))
        if 'happy' in native[key]:
            closed = happy.copy()
        else:
            mask = eye[:, :, 3] > 128
            points = []
            for x in np.where(mask.any(0))[0]:
                ys = np.where(mask[:, x])[0]
                if len(ys):
                    points.append([x, float(ys.min())])
            points = np.array(points)
            xp = points[:, 0]
            yp = np.polyval(np.polyfit((xp - cx) / d, points[:, 1], 2), (xp - cx) / d)
            yp -= yp.mean()
            yp *= min(1, d * 0.1 / max(np.ptp(yp), 1))
            yp += cy
            closed = np.zeros_like(eye)
            line = np.zeros((h, w), np.uint8)
            cv2.polylines(line, [np.int32(np.stack([xp, yp], 1))], False, 255, max(2, round(d * 0.023)), cv2.LINE_AA)
            colors = eye[:, :, :3][mask]
            dark = colors[np.argsort(colors.mean(1))[:max(1, len(colors) // 5)]]
            closed[:, :, :3] = np.median(dark, axis=0)
            closed[:, :, 3] = line
        mapping = np.float32([[1, 0, 0], [0, 0.45, cy * 0.55]])
        half = cv2.warpAffine(eye, mapping, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        groups[key]['eyes-half'] = half
        groups[key]['eyes-closed'] = closed
        blink_rows.append({'channel': key, 'skin_in_blink_layer': False, 'native_closed': native[key]})
    occupied = np.zeros((h, w), np.uint8)
    for key in ('eye-left', 'eye-right', 'mouth'):
        for p in groups[key].values():
            occupied = np.maximum(occupied, p[:, :, 3])
    occupied = cv2.dilate(np.uint8(occupied > 0), np.ones((7, 7), np.uint8))
    nose = (abs(xx - mx) < d * 0.34) & (yy > ey + d * 0.08) & (yy < my - d * 0.12) & ~occupied.astype(bool)
    nose_weight = cv2.distanceTransform(nose.astype(np.uint8), cv2.DIST_L2, 5)
    nose_weight = np.clip(nose_weight / max(4, d * 0.045), 0, 1)[:, :, None]
    clean = np.uint8(clean * (1 - nose_weight) + base * nose_weight)
    for name in groups['base']:
        groups['base'][name] = np.dstack((clean, np.round(fade * 255))).astype(np.uint8)
    save(f / 'skin-field.png', skin_smooth)
    save(f / 'feature-free.png', clean)
    save(f / 'feature-free-mask.png', np.uint8(fade * 255))
    save(f / 'skin-evidence.png', np.uint8(valid.sum(0) / len(stack) * 255))
    save(f / 'feature-erasure.png', np.uint8(erase * 255))
    proof = Image.new('RGB', (6 * 256, 3 * 256), '#d8dfeb')
    draw = ImageDraw.Draw(proof)
    for i, name in enumerate((n for n in ims if n.startswith('expression-'))):
        rgba = np.zeros((h, w, 4), np.uint8)
        for key in ('eye-left', 'eye-right', 'brow-left', 'brow-right', 'mouth'):
            p = groups[key][name]
            a = p[:, :, 3:4] / 255
            rgba[:, :, :3] = p[:, :, :3] * a + rgba[:, :, :3] * (1 - a)
            rgba[:, :, 3] = np.maximum(rgba[:, :, 3], p[:, :, 3])
        composed = clean.astype(float)
        for key in ('blush-left', 'blush-right', 'brow-left', 'brow-right', 'eye-left', 'eye-right', 'mouth'):
            p = groups[key][name]
            a = p[:, :, 3:4] / 255
            composed = composed * (1 - a) + p[:, :, :3] * a
        for j, im in enumerate((ims[name], np.uint8(composed), rgba)):
            tile = Image.fromarray(im).resize((256, 256))
            proof.paste(tile, (i * 256, j * 256), tile if im.shape[2] == 4 else None)
        draw.text((i * 256 + 3, 3), name, fill='blue')
    proof.save(f / 'semantic-v2-grid.png')
    write(f / 'semantic-v2.json', {'method': 'temporal skin evidence; single feature-free base; feature-only straight alpha', 'measurements': metrics, 'blinks': blink_rows, 'status': 'review_required'})
    return (groups, native, fade)
