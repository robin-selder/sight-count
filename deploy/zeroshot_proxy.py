# ─────────────────────────────────────────────────────────────────────────────
# Add this to server.py in the yolo-amd container on m8-server.
#
# /count/zeroshot forwards the photo to the sam-zs container (see deploy/zeroshot/), which runs
# SAM 2 on the RTX 3060. Proxying keeps the Cloudflare tunnel pointed at yolo-amd only.
#
# Deploy steps:
#   1. Add the endpoint below to server.py (requests is already a dependency of ultralytics).
#   2. Make sure sam-zs is on the same compose network as yolo-amd (m8-compute-stack_backend)
#      and named "sam-zs" so the hostname below resolves.
#   3. Roll out with a rollback point:
#        docker tag yolo-amd:latest yolo-amd:prev
#        docker build -t yolo-amd ~/yolo-amd/ && docker compose up -d --no-deps yolo-amd
#      Rollback: docker tag yolo-amd:prev yolo-amd:latest && docker compose up -d --no-deps yolo-amd
# ─────────────────────────────────────────────────────────────────────────────

# ── Extra imports to add at the top of server.py ────────────────────────────
import requests
from fastapi.concurrency import run_in_threadpool

ZS_URL = "http://sam-zs:8000/zeroshot"

# ── New endpoint ─────────────────────────────────────────────────────────────
@app.post("/count/zeroshot")
async def count_zeroshot(file: UploadFile = File(...)):
    """Second-opinion count from SAM 2 (no training). Returns {total, boxes, ms}."""
    raw = await file.read()
    files = {"file": (file.filename or "image.jpg", raw, file.content_type or "image/jpeg")}
    try:
        r = await run_in_threadpool(lambda: requests.post(ZS_URL, files=files, timeout=30))
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="zeroshot service unreachable")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail="zeroshot service error")
    return r.json()
