"""Turn raw SAM masks into a list of item boxes (the "zero-shot" count).

Filters tuned on the first 17 real hardware photos (see autolabel prototype):
  - size limits and "same item" size consistency
  - Lab colour distance from the background (median) colour, so grey/silver parts survive
  - smooth + colourless masks are shadows, not items
  - masks spanning most of the image are paper edges / UI strips
  - masks mostly inside a larger mask's filled outline are parts (heads, threads, washer holes)
"""
import cv2
import numpy as np

MIN_FRAC, MAX_FRAC = 0.0002, 0.05   # mask area as a fraction of the image
BG_DIST = 14.0                      # min Lab distance between mask mean colour and background colour
MIN_CHROMA = 9.0                    # Lab a/b distance that counts as "coloured"
MIN_TEXTURE = 16.0                  # L-channel std that counts as "textured" (specular metal)
SPAN_FRAC = 0.6                     # drop masks whose bbox spans >60% of the image
DEDUP_IOU = 0.6
PART_CONTAIN = 0.7                  # drop a mask if >=70% of it lies inside a larger kept mask
MIN_REL = 0.4                       # drop masks smaller than 40% of the typical (larger-half median) area


def _iou(a, b):
    return np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())


def find_items(img_bgr, masks):
    """img_bgr: HxWx3 uint8. masks: (N, h, w) bool/float from SAM. Returns [(x0, y0, x1, y1)] normalised."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(lab.reshape(-1, 3), axis=0)

    kept = []
    for mk in masks:
        mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        area = int(mk.sum())
        if not (MIN_FRAC * h * w <= area <= MAX_FRAC * h * w):
            continue
        px = lab[mk]
        mean = px.mean(axis=0)
        if np.linalg.norm(mean - bg) < BG_DIST:
            continue
        if np.linalg.norm(mean[1:] - bg[1:]) < MIN_CHROMA and px[:, 0].std() < MIN_TEXTURE:
            continue  # smooth and colourless: a shadow
        ys, xs = np.where(mk)
        if (xs.max() - xs.min()) > SPAN_FRAC * w or (ys.max() - ys.min()) > SPAN_FRAC * h:
            continue
        kept.append((area, mk))
    kept.sort(key=lambda t: -t[0])

    final, filled = [], []
    for area, mk in kept:
        if all(_iou(mk, f) < DEDUP_IOU and np.logical_and(mk, ff).sum() / area < PART_CONTAIN
               for (_, f), ff in zip(final, filled)):
            final.append((area, mk))
            cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fm = np.zeros(mk.shape, np.uint8)
            cv2.drawContours(fm, cnts, -1, 1, cv2.FILLED)
            filled.append(fm > 0)
    if final:
        ref = np.median([a for a, _ in final[: max(1, (len(final) + 1) // 2)]])
        final = [(a, m) for a, m in final if a >= MIN_REL * ref]

    boxes = []
    for _, mk in final:
        ys, xs = np.where(mk)
        boxes.append((xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h))
    return boxes
