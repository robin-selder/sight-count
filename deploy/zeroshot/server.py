"""Zero-shot item counter controller (SAM 2 via a short-lived worker process).

Runs as the sam-zs container on the RTX 3060. yolo-amd proxies /count/zeroshot to it, so the
Cloudflare tunnel and the app's server URL do not change.

Shares the 12 GB card with qwen3.5-claw, so it is built to hold nothing while idle and never push:
  - this process imports no torch/CUDA, so there is no standing VRAM or CUDA context
  - each request runs worker.py in a fresh subprocess, capped at SAM_BUDGET_MIB; VRAM is freed on exit
  - preflight: if less than SAM_MIN_FREE_MIB of VRAM is free, refuse with 503 instead of loading
  - one worker at a time; out-of-memory retries in a clean process with a coarser prompt grid
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

MIN_FREE_MIB = int(os.environ.get("SAM_MIN_FREE_MIB", "3400"))
WORKER_TIMEOUT_S = int(os.environ.get("SAM_WORKER_TIMEOUT_S", "60"))
QUEUE_WAIT_S = int(os.environ.get("SAM_QUEUE_WAIT_S", "25"))
MAX_SIDE = 1600
# (conf, stability, points batch, prompt grid per side). VRAM is dominated by stored candidate masks, so a
# coarser grid is what keeps this inside SAM_BUDGET_MIB (grid 20 peaked at ~1.6 GiB on the test photos);
# the retry uses a coarser grid still, in a fresh process.
ATTEMPTS = [(0.75, 0.88, 8, 20), (0.75, 0.88, 8, 16)]
OOM_EXIT = 3

app = FastAPI()
_lock = threading.Lock()


def free_vram_mib():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10)
    return int(r.stdout.strip().splitlines()[0])


@app.get("/health")
def health():
    try:
        free = free_vram_mib()
    except Exception:
        free = None
    return {"status": "ok", "free_vram_mib": free, "min_free_mib": MIN_FREE_MIB}


@app.post("/zeroshot")
def zeroshot(file: UploadFile = File(...)):
    raw = file.file.read()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    t0 = time.time()
    if not _lock.acquire(timeout=QUEUE_WAIT_S):
        raise HTTPException(status_code=503, detail="busy")
    try:
        try:
            free = free_vram_mib()
        except Exception:
            raise HTTPException(status_code=503, detail="gpu_unreadable")
        if free < MIN_FREE_MIB:  # e.g. qwen is (re)loading: do not touch the GPU
            raise HTTPException(status_code=503, detail=f"gpu_busy free={free}MiB<{MIN_FREE_MIB}MiB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
        cv2.imwrite(path, img)
        try:
            for attempt, (conf, stab, batch, grid) in enumerate(ATTEMPTS, 1):
                try:
                    p = subprocess.run([sys.executable, "worker.py", path, str(conf), str(stab), str(batch), str(grid)],
                                       capture_output=True, text=True, timeout=WORKER_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    raise HTTPException(status_code=503, detail="worker_timeout")
                if p.returncode == 0:
                    out = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1])
                    out["attempts"] = attempt
                    out["ms"] = int((time.time() - t0) * 1000)
                    return out
                if p.returncode != OOM_EXIT:
                    raise HTTPException(status_code=500, detail="worker_failed")
            raise HTTPException(status_code=503, detail="gpu_budget_exceeded")
        finally:
            os.unlink(path)
    finally:
        _lock.release()
