import argparse, json, hashlib
from pathlib import Path
import cv2, numpy as np
from PIL import Image

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return np.array(Image.open(p).convert('RGB'))

def measure(original, candidate, hair, face_region):
    kernel = np.ones((5, 5), np.uint8)
    core = cv2.erode(np.uint8(hair), kernel) > 0
    skin = ~(cv2.dilate(np.uint8(hair), kernel) > 0) & face_region
    if core.sum() < 10 or skin.sum() < 10:
        raise ValueError('Insufficient source labels')
    hc = np.median(original[core], axis=0)
    sc = np.median(original[skin], axis=0)
    v = hc - sc

    def alpha(im):
        return np.clip((im.astype(float) - sc) @ v / max(float(v @ v), 1), 0, 1)
    original_alpha = alpha(original)
    candidate_alpha = alpha(candidate)
    zone = (cv2.dilate(np.uint8(hair), kernel) > 0) & face_region
    zone &= (original_alpha <= 0.5) | hair
    before = (original_alpha > 0.5) & zone
    after = (candidate_alpha > 0.5) & zone
    iou = float((before & after).sum() / max(1, (before | after).sum()))

    def staircase(a):
        binary = np.uint8(a > 0.5)
        boundary = (cv2.dilate(binary, kernel) != cv2.erode(binary, kernel)) & zone
        lap = abs(cv2.Laplacian(a, cv2.CV_64F, ksize=1))
        gx = abs(cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=1))
        gy = abs(cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=1))
        return float(lap[boundary].sum() / max((gx + gy)[boundary].sum(), 1e-08))
    oldscore = staircase(original_alpha)
    score = staircase(candidate_alpha)
    return {'hair_iou': iou, 'hair_iou_minimum': 0.99, 'hair_silhouette_pass': iou >= 0.99, 'source_hair_pixels': int(before.sum()), 'candidate_hair_pixels': int(after.sum()), 'source_staircase_score': oldscore, 'candidate_staircase_score': score, 'staircase_ratio': score / max(oldscore, 1e-08), 'aliasing_pass': score <= oldscore * 1.02, 'staircase_limit_ratio': 1.02}
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--masks', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    m = np.load(a.masks)
    report = measure(read(a.original), read(a.candidate), m['hair'], m['face_region'])
    report.update(original_sha256=sha(a.original), candidate_sha256=sha(a.candidate), mask_sha256=sha(a.masks), gate_sha256=sha(__file__))
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
