# ─────────────────────────────────────────────────────────────────────────────
# Add this to server.py in the yolo-amd container on m8-server.
#
# Deploy steps:
#   1. Copy this endpoint into ~/server.py (or wherever the yolo-amd image
#      builds its server from) alongside the existing /count and /health routes.
#   2. Add the extra imports listed below if they aren't already present.
#   3. Ensure /models/training is persisted — add a volume to docker-compose.yml:
#
#        yolo-amd:
#          volumes:
#            - ./models:/models          # existing
#            - ./training:/models/training  # add this line
#
#      Images are saved inside /models/training/{cls}/ which keeps them on the
#      host even after container restarts. Adjust the path if you prefer a
#      different host mount point.
#   4. Rebuild and restart:
#        docker compose build yolo-amd
#        docker compose up -d yolo-amd
# ─────────────────────────────────────────────────────────────────────────────

# ── Extra imports to add at the top of server.py ────────────────────────────
from pathlib import Path
from fastapi import Form  # add Form if not already imported

# ── Config (add near the top, after app = FastAPI(...)) ─────────────────────
TRAINING_DIR = Path("/models/training")

# ── New endpoint ─────────────────────────────────────────────────────────────
@app.post("/training/upload")
async def training_upload(
    file: UploadFile,
    cls: str      = Form(...),
    item_count: int = Form(1),
):
    """Receive a training image from the mobile app and save it to disk."""
    valid = {"bolt", "nut", "washer", "screw"}
    if cls not in valid:
        raise HTTPException(status_code=400, detail=f"cls must be one of {valid}")

    # Read and apply EXIF orientation fix (same as /count — required for iPhone)
    raw = await file.read()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))

    save_dir = TRAINING_DIR / cls
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sequential filename: bolt_0001_5x.jpg  (number of existing files + 1)
    seq      = len(list(save_dir.glob("*.jpg"))) + 1
    filename = f"{cls}_{seq:04d}_{item_count}x.jpg"
    img.convert("RGB").save(save_dir / filename, "JPEG", quality=92)

    return {"saved": True, "cls": cls, "item_count": item_count,
            "filename": filename, "seq": seq}
