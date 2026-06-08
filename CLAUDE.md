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
- `server.py` applies `PIL.ImageOps.exif_transpose()` to fix iPhone/Android EXIF rotation
  (orientation 6 was causing sideways inference → zero detections). KEEP THIS.
- CORS: `allow_origins=["*"]`

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
- Quick tunnel (ephemeral, no account). Current URL changes on container restart.
- Current: `https://kiss-harold-gap-rainbow.trycloudflare.com`
- Get current URL: `docker compose logs --tail=20 cloudflared | grep trycloudflare`
- App ⚙ API Config button updates the base URL (saved to localStorage) — no redeploy needed.
- Future: move to a named tunnel for a stable URL once out of prototype.

## Frontend (index.html)
- Dark industrial aesthetic, Bebas Neue + DM Mono, accent `#e8ff47`.
- Home → "Start Analyzing" → live viewfinder → tap shutter → frame frozen → POST to `/count`
  → count shown; "Retake" to repeat. ⚙ API Config bottom-left of home.
- Bounding-box overlay code exists (reads a `detections[{label,confidence,bbox}]` array) but is
  dormant because the API currently returns total-only. Re-enable after retrain.

## Known issues / watch-items
- Tunnel URL changes on cloudflared restart (update via ⚙ in app).
- High-res iPhone photos (4032×3024) — EXIF transpose is required; keep it.
- Reflective metal + shadows defeat brightness thresholding (not relevant to YOLO path, noted
  for any CV fallback).

## Next action
Begin the capture session per "Chosen plan" step 1, then set up the Roboflow project.