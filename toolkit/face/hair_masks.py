import cv2, numpy as np
from toolkit.face.surface import coordinates, load_groups
from toolkit.face.raster import sample
from toolkit.face.matte_v2 import skin_pixels

def original_masks(image, b, groups, source_coords=None):
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w))
    d = b['source_landmarks']['interocular_px']
    scale = h / b['canvas'][1]
    dp = d * scale
    if source_coords is None:
        sx, sy = (xx.astype(float), yy.astype(float))
    else:
        sx, sy = source_coords
    masks = {}
    for key in ('brow-left', 'brow-right', 'eye-left', 'eye-right', 'mouth'):
        name = 'mouth-closed' if key == 'mouth' else 'expression-neutral'
        p = groups[key][name]
        a = p[:, :, 3].astype(float) / 255
        a = sample(a, sx, sy)
        radius = max(2, round(dp * 0.022))
        masks[key] = cv2.dilate(np.uint8(a > 0.1), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))) > 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    limit = max(2, round(dp * 0.06))
    brow = np.zeros((h, w), bool)
    for key in ('brow-left', 'brow-right'):
        p = groups[key]['expression-neutral']
        a = sample(p[:, :, 3], sx, sy) > 100
        py, px = np.where(a)
        if not len(px):
            continue
        x0, x1 = (px.min(), px.max() + 1)
        y0, y1 = (py.min(), py.max() + 1)
        template = np.uint8(a[y0:y1, x0:x1])
        best = None
        for dy in range(-limit, limit + 1):
            for dx in range(-limit, limit + 1):
                if min(y0 + dy, x0 + dx) < 0 or y1 + dy > h or x1 + dx > w:
                    continue
                on = gray[y0 + dy:y1 + dy, x0 + dx:x1 + dx] < 120
                score = (on * template).sum() / template.sum() - 0.2 * (on * (1 - template)).sum() / max(1, (1 - template).sum()) - 0.001 * (dx * dx + dy * dy) / (scale * scale)
                if best is None or score > best[0]:
                    best = (score, dx, dy)
        _, dx, dy = best
        placed = np.zeros((h, w), np.uint8)
        placed[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = template
        rad = max(2, round(dp * 0.022))
        placed = cv2.dilate(placed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))) > 0
        masks[key] = placed
        brow |= placed
    ey = np.mean(np.array(b['source_landmarks']['eyes'])[:, 1])
    cx = np.mean(np.array(b['source_landmarks']['eyes'])[:, 0])
    dark = ~skin_pixels(image) & (np.min(image, axis=-1) < 200)
    eligible = dark & (sy < ey + d * 0.75)
    vertical = cv2.morphologyEx(np.uint8(eligible), cv2.MORPH_OPEN, np.ones((max(5, round(dp * 0.16)), 1), np.uint8)) > 0
    eligible = dark & ~np.logical_or.reduce(list(masks.values())) & (sy < ey + d * 0.75)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(eligible))
    seed = (sy < ey - d * 0.72) & (abs(sx - cx) < d * 0.7)
    ids = np.unique(labels[seed & eligible])
    ids = ids[ids > 0]
    hair = np.isin(labels, ids)
    hair = (cv2.dilate(np.uint8(hair), np.ones((3, 3), np.uint8)) > 0) & dark
    hair &= ~brow
    return (masks, hair)

def skin_field(image, mask):
    known = skin_pixels(image) & ~mask
    radius = max(3, image.shape[0] / 28)
    weight = known.astype(np.float32)
    den = cv2.GaussianBlur(weight, (0, 0), radius)
    field = cv2.GaussianBlur(image.astype(np.float32) * weight[:, :, None], (0, 0), radius) / np.maximum(den[:, :, None], 1e-06)
    out = image.copy()
    out[mask] = np.uint8(np.clip(field[mask], 0, 255))
    return out
