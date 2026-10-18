import cv2, numpy as np
from toolkit.face.matte_v2 import skin_pixels

def skin_field(image, mask):
    if not mask.any():
        return image.copy()
    yy, xx = np.where(mask)
    pad = max(8, round(image.shape[0] / 28))
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(image.shape[1], int(xx.max()) + pad + 1)
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(image.shape[0], int(yy.max()) + pad + 1)
    original = image[y0:y1, x0:x1]
    hole = mask[y0:y1, x0:x1]
    known = skin_pixels(original) & ~hole
    weight = known.astype(np.float32)
    radius = max(3, image.shape[0] / 28)
    den = cv2.GaussianBlur(weight, (0, 0), radius)
    field = cv2.GaussianBlur(original.astype(np.float32) * weight[:, :, None], (0, 0), radius) / np.maximum(den[:, :, None], 1e-06)
    work = field.copy()
    work[known] = original[known]
    kernel = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    for iteration in range(1200):
        proposal = cv2.filter2D(work, -1, kernel, borderType=cv2.BORDER_REFLECT)
        delta = float(np.max(np.abs(proposal[hole] - work[hole]))) if iteration % 32 == 0 else 1
        work[hole] = proposal[hole]
        if delta < 0.025:
            break
    out = image.copy()
    patch = out[y0:y1, x0:x1]
    patch[hole] = np.uint8(np.clip(np.rint(work[hole]), 0, 255))
    return out
