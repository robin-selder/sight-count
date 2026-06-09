# ─────────────────────────────────────────────────────────────────────────────
# Add this to server.py in the yolo-amd container on m8-server.
#
# Deploy steps:
#   1. Copy this endpoint into ~/yolo-amd/app/server.py alongside the existing
#      /training/upload, /count, and /health routes.
#   2. Ensure the extra imports listed below are present at the top.
#   3. Rebuild and restart:
#        docker build -t yolo-amd ~/yolo-amd/ && docker compose up -d --no-deps yolo-amd
# ─────────────────────────────────────────────────────────────────────────────

# ── Extra imports to add at the top of server.py ────────────────────────────
import cv2
import numpy as np

# ── New endpoint ─────────────────────────────────────────────────────────────
@app.post("/domino/count")
async def domino_count(file: UploadFile):
    """Count domino dots in an image using Hough circle detection.

    No training data required — uses classical CV on the assumption that
    domino dots are dark circles on a lighter tile surface.

    Tuning guide (adjust if results are poor):
      param2: lower = more detections (more false positives);
              higher = fewer detections (more misses). Start at 18–25.
      minRadius/maxRadius: based on expected dot pixel size at typical
              phone distance (~30–60 cm). Increase if phone is held far back.
    """
    raw = await file.read()
    img_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    img_rgb = np.array(img_pil.convert("RGB"))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Gaussian blur suppresses tile texture, shadows, and JPEG noise.
    # Kernel size 9 works well for photos ≥720p.
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape
    min_dim = min(h, w)

    # Dot radius bounds calibrated for a phone held 30–60 cm from a table.
    # A standard domino dot is ~5 mm; at 40 cm with a 720p frame,
    # one tile fills ~160 px wide → dot radius ≈ 8–12 px.
    # Using a generous range to tolerate distance variation.
    min_r = max(5, int(min_dim * 0.010))   # ~7 px at 720p
    max_r = max(min_r + 6, int(min_dim * 0.040))  # ~29 px at 720p

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max_r * 2,   # prevent double-counting dots that touch
        param1=60,            # Canny high threshold
        param2=20,            # accumulator threshold — tune if over/under-detecting
        minRadius=min_r,
        maxRadius=max_r,
    )

    total = 0 if circles is None else int(len(circles[0]))
    return {"total": total, "message": f"{total} dot{'s' if total != 1 else ''} detected"}
