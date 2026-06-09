# Sight Count — Project Context

## What this is
A single-page HTML camera app for counting hardware items (bolts, nuts, washers, screws)
from a phone/tablet photo. Frontend on GitHub Pages; inference on a self-hosted YOLO model
on m8-server, reached through a Cloudflare tunnel.

## Repo
- GitHub: https://github.com/robin-selder/sight-count
- Live URL: https://robin-selder.github.io/sight-count
- Single-file deployment: `index.html` at repo root

## Current Direction (DECIDED)
Collect real photographs of the actual hardware, label them in Roboflow (auto-label with
Grounding DINO + manual review), train a YOLO model on that real data, and deploy it as the
`yolo-amd` inference container. This replaces the current model, which was trained on
mismatched public data. See "Why the current model fails" below — the fix is a
domain-matched dataset, not a new platform or a different model architecture.

## Infrastructure (m8-server — 10.0.0.136)
All services run in Docker under the `m8-compute-stack` compose stack at `~/docker-compose.yml`.

| Container     | Port        | GPU                 | Purpose                                |
|---------------|-------------|---------------------|----------------------------------------|
| yolo-amd      | 8010→8000   | Radeon 680M (ROCm)  | YOLOv8s fastener inference (FastAPI)    |
| cloudflared   | —           | —                   | Cloudflare quick tunnel to yolo-amd     |
| ollama-nvidia | 11434       | RTX 3060            | Primary LLM (Qwen3 14.8B)               |
| ollama-amd    | 11435       | Radeon 680M (Vulkan)| Lightweight LLMs (Moondream, Qwen3 1.7b)|
| whisper-amd   | 9000        | Radeon 680M (ROCm)  | Whisper transcription                   |

Training note: use the **RTX 3060 (ollama-nvidia's GPU)** for YOLO training runs — it is the
capable GPU. Inference currently runs on the 680M via ROCm in the yolo-amd container.

## Inference API (yolo-amd container)
- Model path in container: `/models/fasteners-combined/weights/best.pt` (to be replaced)
- `GET /health` → `{status, device, rocm, model}`
- `POST /count?conf=0.20` (multipart `file`) → `{total, message}`
  - Returns total count only. Per-class output was intentionally dropped due to
    bolt/screw confusion in the current model. Re-enable per-class + detections array
    once the retrained model is accurate.
- `POST /training/upload` (multipart `file`, form fields `cls`, `item_count`) → `{saved, cls, item_count, filename, seq}`
  - Saves EXIF-corrected JPEG to `/models/training/{cls}/` on the host volume.
  - Filename format: `bolt_0001_5x.jpg` (class, sequence, item count per photo).
  - `item_count` is set by the user in the app and embedded in the filename to assist
    Roboflow label validation (expected box count per image is known up front).
- `server.py` applies `PIL.ImageOps.exif_transpose()` to fix iPhone/Android EXIF rotation
  (orientation 6 was causing sideways inference → zero detections). KEEP THIS on all endpoints.
- CORS: `allow_origins=["*"]`
- Source: `~/yolo-amd/app/server.py` on host. Image must be rebuilt after edits:
  `docker build -t yolo-amd ~/yolo-amd/ && docker compose up -d --no-deps yolo-amd`

## Why the current model fails (root cause)
The current model combines TWO public Roboflow Universe datasets. It is not a platform problem
and not synthetic data — it is **domain gap + conflicting labels**:
- The public images don't match real-world photos (camera, lighting, background, the specific
  zinc/brass finish, scale), so the model never learned features matching actual captures.
- Evidence: at `conf=0.01` the model returns ~zero/noise on real photos, yet a one-line HSV
  **saturation threshold cleanly isolates every nut/bolt/washer** (see "Segmentation is solved").
  The items are trivially present; the detector just wasn't trained on matching images.
- Merging two datasets made it worse: dataset A's "bolt" vs dataset B's "screw" gave
  contradictory supervision (→ bolt/screw confusion). Washers were underrepresented
  (~13.2% mAP). DO NOT fix this by combining more public datasets.

## Segmentation is solved (important evidence)
A simple HSV saturation Otsu threshold (`cv_count2.py`) produces near-perfect binary masks of
the hardware on both white-paper and grey-table backgrounds — clean rings/blobs, black
background. Implication: the hard part (separating metal from background) is easy in this
domain, so a YOLO model trained on real photos should perform very well. The only thing that
ever failed at counting was blob-splitting params in a hand-rolled watershed — not the model's
ability to see the items.

## Approaches evaluated (history — don't re-litigate)
- **Current YOLO (public combined dataset):** fails on real photos. Root cause above.
- **Moondream (ollama-amd, 1B Q4):** unreliable, ignores instructions, empty/garbage output.
- **Other local VLM considered:** `minicpm-v` on the RTX 3060 (not pursued).
- **Gemini 2.5 Flash (vision API):** works, but off-by-one on 2 of 3 real images. VLMs estimate
  rather than enumerate — not acceptable for an accurate counter.
- **Classical CV — brightness threshold (`cv_count.py`):** failed; reflective metal + shadow
  gradient broke brightness-based Otsu.
- **Classical CV — saturation threshold (`cv_count2.py`):** segmentation excellent; only the
  counting/splitting step needs work. Viable same-day fallback via Hough circles for round
  items (nuts/washers) + connected components for separated bolts/screws.
- **Class-agnostic / zero-shot counting (CountGD, GeCo, CounTR):** no-train, exemplar/zero-shot
  counters that run on the 3060 — fallback option if we ever want to avoid training.

## Chosen plan — Real-photo capture → Roboflow → YOLO retrain
1. **Capture:** 40–50 photos per item type (bolt, nut, washer, screw). Vary quantity (1–20),
   arrangement (separated AND touching/overlapping), background (white paper + grey bench),
   lighting, and angle. Shoot with the actual phone(s) the app will use. Include mixed-type
   shots too. Hold orientation/EXIF as the app produces it.
2. **Label in Roboflow:** create a project, use auto-label (Grounding DINO) with explicit
   prompts ("bolt", "hex nut", "washer", "wood screw") — review/correct every box. Keep ONE
   consistent class taxonomy. Avoid the cross-dataset label conflicts that broke the old model.
3. **Split:** 70% train / 20% val / 10% test.
4. **Train:** YOLOv8s on the RTX 3060. Start from yolov8s.pt; augment for lighting/scale.
5. **Evaluate:** check per-class mAP (esp. washers) and bolt-vs-screw confusion on the test set.
6. **Deploy:** export weights, drop into `/models/<new>/weights/best.pt`, point `server.py`
   at the new path, rebuild yolo-amd, restart via compose.
7. **Re-enable** per-class counts + detections array in `server.py`, then turn the frontend
   bounding-box overlay back on (code already supports a `detections` array).

## Cloudflare Tunnel
- Quick tunnel (ephemeral, no account). URL changes only if the `cloudflared` container restarts
  (restarting `yolo-amd` alone does NOT change it — confirmed).
- Current: `https://kiss-harold-gap-rainbow.trycloudflare.com`
- Get current URL: `docker logs cloudflared 2>&1 | grep trycloudflare | tail -1`
- App ⚙ Server button (home screen, bottom-left) updates the base URL saved to localStorage.
- Future: move to a named tunnel for a stable URL once out of prototype.

## Frontend (index.html) — Training Data Collection Mode
The app is currently in **training data collection mode** (inference/analyze disabled).

- Home screen shows per-class progress bars (captured / 200 target) loaded from localStorage.
- Camera screen:
  - Class tab strip (BOLT / NUT / WASHER / SCREW) — switches active class.
  - Info bar: shooting rules ("same size & type per shot, vary bg & lighting") + item count
    stepper (− / N / +, default 5). Item count is embedded in the saved filename.
  - Viewfinder → tap shutter → flash → frozen frame preview → Save or Retake.
  - Save flow: POST to `/training/upload` on server (8 s timeout) → fallback to Web Share
    (iOS) or download (desktop) if server unreachable; toast shown on fallback.
- ⚙ Server button (home, bottom-left) sets the Cloudflare tunnel URL.
- ? Guide button (home, bottom-right) opens the collection guide modal.
- Dark industrial aesthetic, Bebas Neue + DM Mono, accent `#e8ff47`.

To restore analyze mode after retraining: re-enable `/count` POST in the capture flow,
re-add the result overlay and bounding-box renderer, and re-enable per-class + detections
in `server.py`. The JS skeleton for bounding boxes is gone from index.html now — refer to
git history (commit `33c631b`) for the original renderBoxes() implementation.

## Training data — current status (as of 2026-06-08)
Collection is **in progress**. Images upload automatically to `~/yolo-amd/models/training/`
on m8-server (persisted via the existing `/models` volume mount).

| Class  | Images collected | Target |
|--------|-----------------|--------|
| bolt   | 3               | 200    |
| nut    | 2               | 200    |
| washer | 1               | 200+   |
| screw  | 0               | 200    |

Check live counts: `find ~/yolo-amd/models/training -type f | sort`

## Known issues / watch-items
- Tunnel URL changes on `cloudflared` container restart (update via ⚙ Server in app).
- High-res iPhone photos (4032×3024) — EXIF transpose is required; keep it on all endpoints.
- Reflective metal + shadows defeat brightness thresholding (not relevant to YOLO path, noted
  for any CV fallback).
- SSH to m8-server: `ssh -i ~/.ssh/id_ed25519 robin@10.0.0.136`

## Next action (when training data collection is complete)
1. Bulk-upload images from `~/yolo-amd/models/training/` to a new Roboflow project.
2. Auto-label with Grounding DINO ("bolt", "hex nut", "washer", "wood screw") — the item
   count in each filename (e.g. `_5x.jpg`) tells you how many boxes to expect per image.
3. Manual review: correct missed/wrong boxes, especially overlapping items.
4. Export as YOLOv8 format, 70/20/10 split.
5. Train on RTX 3060 (`ollama-nvidia` GPU): `yolo train model=yolov8s.pt data=...`
6. Evaluate per-class mAP — target washers and bolt/screw confusion specifically.
7. Drop new weights into `/models/<new>/weights/best.pt`, update path in `server.py`,
   rebuild yolo-amd, restart.
8. Re-enable per-class counts + detections array in `server.py`.
9. Restore analyze mode in `index.html` (see git history for renderBoxes implementation).