"""One-shot SAM 2 worker: counts items in one image, prints a JSON line, exits.

Spawned per request by server.py so the CUDA context and all VRAM are released when it exits.
Exit code 3 = hit the VRAM budget (out of memory); the controller may retry with stricter thresholds.
Usage: worker.py IMAGE CONF STABILITY POINTS_BATCH [POINTS_PER_SIDE]
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM2Predictor

from zs_core import find_items

WEIGHTS = os.environ.get("SAM_WEIGHTS", "/weights/sam2.1_s.pt")
BUDGET_MIB = int(os.environ.get("SAM_BUDGET_MIB", "2150"))


def main():
    path, conf, stab, batch = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    grid = int(sys.argv[5]) if len(sys.argv) > 5 else 32   # prompt points per side; fewer = far fewer candidate masks in VRAM
    img = cv2.imread(path)
    t0 = time.time()
    try:
        _, total = torch.cuda.mem_get_info()
        # hard cap on this process's tensor memory (the CUDA context itself sits outside the cap)
        torch.cuda.set_per_process_memory_fraction(min(1.0, BUDGET_MIB * 2**20 / total))
        predictor = SAM2Predictor(overrides=dict(
            task="segment", mode="predict", imgsz=1024, model=WEIGHTS,
            device="cuda:0", save=False, verbose=False))
        res = predictor(source=img, points_batch_size=batch, points_stride=grid, conf_thres=conf,
                         stability_score_thresh=stab)[0]
        masks = res.masks.data.cpu().numpy() if res.masks is not None else np.zeros((0, 1, 1))
    except torch.OutOfMemoryError:
        sys.exit(3)
    boxes = find_items(img, masks)
    print(json.dumps({"total": len(boxes),
                      "boxes": [[round(float(v), 4) for v in b] for b in boxes],
                      "peak_mib": int(torch.cuda.max_memory_allocated() / 2**20),
                      "sam_ms": int((time.time() - t0) * 1000)}))


if __name__ == "__main__":
    main()
