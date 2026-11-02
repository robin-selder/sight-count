"""SAM 2 auto-label prototype: SAM masks filtered by colour/shadow/part rules, compared to the filename count.

Offline tool (not a service). Reads /data/training/{bolt,nut,washer,screw}/*_Nx.jpg, writes YOLO labels,
overlay images, summary.csv and a raw-mask cache to /out. Run in a throwaway container, e.g. on the
AMD iGPU (SC_DEVICE=cuda:0, yolo-amd image, ~3 min/image) or CPU (default). On the RTX 3060 use SC_GPU_FRAC
and read the VRAM rules in CLAUDE.md first. The filter logic is mirrored in deploy/zeroshot/zs_core.py.
"""
import csv, gc, os, re, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM2Predictor

SRC = Path('/data/training')
OUT = Path('/out')
CLASSES = ['bolt', 'nut', 'washer', 'screw']
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else 'sam2.1_s.pt'

MIN_FRAC, MAX_FRAC = 0.0002, 0.05   # mask area as fraction of image
MIN_CHROMA = 9.0                    # Lab a/b distance from paper colour that counts as 'coloured'
MIN_TEXTURE = 16.0                  # L-channel std inside a mask that counts as 'textured' (specular metal)
SPAN_FRAC = 0.6                     # drop masks whose bbox spans >60% of the image
BG_DIST = 14.0                      # min Lab distance between mask mean colour and background (median) colour
DEDUP_IOU = 0.6
PART_CONTAIN = 0.7                  # drop a mask if >=70% of it lies inside a larger kept mask
MIN_REL = 0.4                       # drop masks smaller than 40% of the typical (larger-half median) area

(OUT / 'overlay').mkdir(parents=True, exist_ok=True)
(OUT / 'labels').mkdir(parents=True, exist_ok=True)


def to_lab_bg(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab, np.median(lab.reshape(-1, 3), axis=0)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(1, np.logical_or(a, b).sum())


if os.environ.get('SC_GPU_FRAC') and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ['SC_GPU_FRAC']))  # hard VRAM cap for this process
def make_predictor(device):
    return SAM2Predictor(overrides=dict(task='segment', mode='predict', imgsz=1024, model=WEIGHTS, device=device, save=False, verbose=False))


predictor = make_predictor(os.environ.get('SC_DEVICE', 'cpu'))
cpu_predictor = None
rows = []
for cls_id, cls in enumerate(CLASSES):
    for p in sorted((SRC / cls).glob('*.jpg')):
        m = re.search(r'_(\d+)x\.jpg$', p.name)
        expected = int(m.group(1)) if m else None
        img = cv2.imread(str(p))
        scale = 1600 / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        lab, bg = to_lab_bg(img)

        cache = (OUT / 'cache' / f'{p.stem}.npz')
        dt = 0.0
        if cache.exists():
            masks = np.load(cache)['masks']
        else:
            t0 = time.time()
            try:
                res = predictor(source=img, points_batch_size=16, conf_thres=0.75, stability_score_thresh=0.88)[0]
            except torch.OutOfMemoryError:
                # too many candidate masks for the VRAM cap: free the GPU and redo this image on the CPU
                gc.collect()
                torch.cuda.empty_cache()
                print(f'  {p.name}: GPU cap hit, falling back to CPU', flush=True)
                if cpu_predictor is None:
                    cpu_predictor = make_predictor('cpu')
                res = cpu_predictor(source=img, points_batch_size=16, conf_thres=0.75, stability_score_thresh=0.88)[0]
            dt = time.time() - t0
            masks = res.masks.data.cpu().numpy() if res.masks is not None else np.zeros((0, h, w))
            cache.parent.mkdir(exist_ok=True)
            np.savez_compressed(cache, masks=masks.astype(bool))
        kept = []
        for mk in masks:
            mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            area = mk.sum()
            if not (MIN_FRAC * h * w <= area <= MAX_FRAC * h * w):
                continue
            px = lab[mk]
            dist = np.linalg.norm(px.mean(axis=0) - bg)
            chroma = np.linalg.norm(px.mean(axis=0)[1:] - bg[1:])
            if dist < BG_DIST:
                continue
            if chroma < MIN_CHROMA and px[:, 0].std() < MIN_TEXTURE:   # smooth, colourless -> shadow
                continue
            ys_, xs_ = np.where(mk)
            if (xs_.max() - xs_.min()) > SPAN_FRAC * w or (ys_.max() - ys_.min()) > SPAN_FRAC * h:
                continue                                              # border strips / paper edges
            kept.append((area, mk))
        kept.sort(key=lambda t: -t[0])
        final = []
        filled = []   # each kept mask with its holes filled, so a washer's hole counts as "inside" it
        for area, mk in kept:
            # drop duplicates and parts (head/thread slices, washer/nut holes) covered by a larger kept mask
            if all(iou(mk, f) < DEDUP_IOU and np.logical_and(mk, ff).sum() / area < PART_CONTAIN
                   for (_, f), ff in zip(final, filled)):
                final.append((area, mk))
                cnts_, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                fm = np.zeros(mk.shape, np.uint8)
                cv2.drawContours(fm, cnts_, -1, 1, cv2.FILLED)
                filled.append(fm > 0)
        if final:
            ref = np.median([a for a, _ in final[: max(1, (len(final) + 1) // 2)]])
            final = [(a, m) for a, m in final if a >= MIN_REL * ref]

        lines, vis = [], img.copy()
        for _, mk in final:
            ys, xs = np.where(mk)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            lines.append(f'{cls_id} {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f}')
            cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), max(2, w // 800))
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), max(2, w // 800))
        (OUT / 'labels' / f'{p.stem}.txt').write_text('\n'.join(lines))
        cv2.putText(vis, f'{len(final)} / expected {expected}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    max(1.0, w / 1600), (0, 255, 255), max(2, w // 700))
        small = cv2.resize(vis, (1024, int(1024 * h / w)))
        cv2.imwrite(str(OUT / 'overlay' / f'{p.stem}.jpg'), small)
        rows.append((p.name, expected, len(final), len(masks)))
        print(f'{p.name:28s} expected={expected!s:>3} found={len(final):>3} (raw masks {len(masks)}) {dt:.1f}s', flush=True)

with open(OUT / 'summary.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['file', 'expected', 'found', 'raw_masks'])
    wr.writerows(rows)
ok = sum(1 for _, e, f, _ in rows if e == f)
print(f'\nexact matches: {ok}/{len(rows)}; within ±1: {sum(1 for _, e, f, _ in rows if abs(e - f) <= 1)}/{len(rows)}')
