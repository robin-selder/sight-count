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
Collect real photographs of the actual hardware, label them (SAM 2 auto-label first pass via
`tools/autolabel/`, then Roboflow review; Grounding DINO is the alternative), train a YOLO model on that real data, and deploy it as the
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
| sam-zs        | — (internal)| RTX 3060 (guarded)  | SAM 2 zero-shot counter, proxied by yolo-amd |

### RTX 3060 VRAM rules (IMPORTANT — shared with the primary agent)
`ollama-nvidia` keeps `qwen3.5-claw` resident (8.25 GB of the 12 GB card, loaded at
**num_ctx 73,728**, 2 parallel slots). That is the primary agent (OpenClaw/OpenCode); an OOM there is
not acceptable. Free VRAM is only ~3.6 GB.
- Qwen's VRAM is allocated once at load and does not grow under load (measured: constant 8,252 MiB
  with both slots busy, a 41k-token prompt, and a 2.3 GB ballast held). The risk is a **reload/restart**
  of `ollama-nvidia` while something else holds VRAM — never leave anything resident on the 3060.
- **Never send requests to `qwen3.5-claw` yourself with different options.** The Modelfile default
  `num_ctx` is 114,688 but the running instance is 73,728; a mismatched request makes Ollama reload the model.
- CUDA has no preemption: a process cannot "yield" VRAM. Safety comes from a reserve + preflight, not eviction.
- `sam-zs` is built for this: its controller has no CUDA (zero idle VRAM), each count runs in a
  short-lived worker capped at `SAM_BUDGET_MIB=2150`, and it returns 503 if less than
  `SAM_MIN_FREE_MIB=3400` is free. Measured peak while counting: ~+1.8 GB over baseline.
- Any training run on the 3060 is the same concern (YOLO11s smoke test, batch 8, peaked +1.5 GB). Do
  not start one without the user choosing when (e.g. agent idle).
- The dashboard's "44% → 55%" the user saw was host **RAM**, not VRAM (VRAM baseline is 67%).
- Training/lab containers: image `sc-train:latest` (ultralytics 8.4.198 + CUDA). Always run with
  `-w /tmp` — the base image's `/ultralytics` checkout shadows the upgraded package. Scratch dir `~/sc-lab/`.
- SAM 2 also runs on the 680M iGPU but takes ~170–195 s/image (vs ~3 s on the 3060, ~100 s on 4 CPU
  cores), so it is only usable as an overnight offline batch there.
- Inference runs on the 680M via ROCm in the yolo-amd container.

## Inference API (yolo-amd container)
- Model path in container: `/models/fasteners-combined/weights/best.pt` (to be replaced)
- `GET /health` → `{status, device, rocm, model}`
- `POST /count?conf=0.20` (multipart `file`) → `{total, message}`
  - Returns total count only. Per-class output was intentionally dropped due to
    bolt/screw confusion in the current model. Re-enable per-class + detections array
    once the retrained model is accurate.
- `POST /count/zeroshot` (multipart `file`) → `{total, boxes, ms, attempts, peak_mib}` — second-opinion
  count from SAM 2 (no training). Proxies to the `sam-zs` container (`deploy/zeroshot/`,
  `deploy/zeroshot_proxy.py`). Returns 502 if sam-zs is down and 503 if it refuses (low free VRAM / busy).
  ~5 s per photo; 12/17 exact and 17/17 within ±1 on the first 17 real photos. Not a replacement for the
  retrained YOLO model — a labelling aid and sanity check shown in the save toast.
- Domino counting was removed from the server (2026-11-02): no domino model, no `/domino/count`, and
  `/training/upload` only accepts bolt/nut/washer/screw. Rollback image: `yolo-amd:prev-domino`; old
  source: `~/yolo-amd/app/server.py.pre-domino-removal`. Domino weights and data remain on disk under
  `~/yolo-amd/models/` (`domino-*`, `training/domino/`) and were not deleted.
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
- Evidence: at `conf=0.01` the model returns ~zero/noise on real photos, yet SAM 2 finds every
  item in the same photos (see "Segmentation — what actually works"). The items are clearly
  present; the detector just wasn't trained on matching images.
- Merging two datasets made it worse: dataset A's "bolt" vs dataset B's "screw" gave
  contradictory supervision (→ bolt/screw confusion). Washers were underrepresented
  (~13.2% mAP). DO NOT fix this by combining more public datasets.

## Segmentation — what actually works (important evidence)
**Correction (2026-11):** the earlier claim that an HSV saturation threshold isolates everything
was only true for the brass/yellow-zinc parts. On grey/silver items (washers, zinc screws) it
rejected them (7/17 exact counts; 0 on the zinc screws).

What works: **SAM 2 automatic masks + a few filters** (`deploy/zeroshot/zs_core.py`): Lab colour
distance from the background colour (not saturation), reject smooth colourless masks (shadows),
reject masks spanning most of the image (paper edges / the dark UI strip in some screenshots),
drop parts of a larger mask (heads/threads, and washer/nut holes via the filled outline), and size
consistency. That gets **12/17 exact, 17/17 within ±1** on the first real photos (vs filename counts).
The remaining misses are heavily overlapping bolts, merged screws, and one off-by-one nut/washer
photo — review those by eye when labelling.

Classical CV counting without SAM does not work here: three attempts (colour distance from median,
fitted background surface, local texture, each + watershed) scored 2/17, 1/17 and 0/17. Strong
lighting gradients, shadows and a finger in frame defeat a hand-built foreground model. Don't retry.

## Approaches evaluated (history — don't re-litigate)
- **Current YOLO (public combined dataset):** fails on real photos. Root cause above.
- **Moondream (ollama-amd, 1B Q4):** unreliable, ignores instructions, empty/garbage output.
- **Other local VLM considered:** `minicpm-v` on the RTX 3060 (not pursued).
- **Gemini 2.5 Flash (vision API):** works, but off-by-one on 2 of 3 real images. VLMs estimate
  rather than enumerate — not acceptable for an accurate counter.
- **Classical CV — brightness threshold (`cv_count.py`):** failed; reflective metal + shadow
  gradient broke brightness-based Otsu.
- **Classical CV — saturation threshold (`cv_count2.py`):** worked on brass parts only; fails on
  grey/silver (see "Segmentation — what actually works"). Not a viable fallback.
- **SAM 2 auto-labelling / zero-shot count (2026-11):** works (12/17 exact, 17/17 ±1). Runs on the
  3060 only inside the guarded `sam-zs` service; ~3 s on 3060, ~100 s CPU (4 cores), ~180 s on the
  680M iGPU. SAM's default 32×32 prompt grid needs ~3 GB; a 20×20 grid fits in ~1.6 GB with the same accuracy.
- **Local VLM as a crop classifier (2026-11):** not as a counter (VLMs estimate), but as a class check
  on SAM boxes. `qwen3.5-lite` (4.7B, already on `ollama-amd`, vision-capable) classified 95 SAM crops
  as bolt/screw/nut/washer: **92/95 (97%)** after correcting the mislabeled folder (see Training data),
  ~3.3 s/crop on the 680M iGPU, nothing on the 3060. Remaining misses were one extra SAM box, one nut→bolt,
  one washer→nut. Only 4 screw crops, so screws are barely tested. Moondream is no longer installed and
  Qwen3-VL was not needed. Test scripts were throwaway (`~/sc-lab/vlm_test.py`, `make_crops.py`); wire it
  into `tools/autolabel/` as a class check when labelling at scale. Use `think:false` and `num_predict` ~12.
- **Ultralytics upgrade:** 8.4.60 → 8.4.198 in the training image only (`sc-train`). `yolo-amd` is
  still on 8.4.60; upgrade it only when deploying weights that need it (pin the ROCm torch wheel).
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
- Current: `https://imperial-nancy-richmond-surgeons.trycloudflare.com`
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
    (iOS/Android) or download (desktop) if server unreachable; toast shown on fallback.
  - On save it also calls `/count` and `/count/zeroshot`; toast shows `model: N` immediately and
    `#seq model: N · zs: N · set: N` a few seconds later (zs never blocks saving; silent if unavailable).
- Domino counting mode was removed (UI and server; see "Inference API").
- Android/PWA: `manifest.webmanifest` + `icon-192.png`/`icon-512.png` allow "Add to Home screen"
  (optional — works in a plain Chrome tab). A screen wake lock keeps the screen on while the camera is open.
  Saved JPEGs come from a canvas, so they have no EXIF rotation tag.
- ⚙ Server button (home, bottom-left) sets the Cloudflare tunnel URL.
- ? Guide button (home, bottom-right) opens the collection guide modal.
- Dark industrial aesthetic, Bebas Neue + DM Mono, accent `#e8ff47`.

To restore analyze mode after retraining: re-enable `/count` POST in the capture flow,
re-add the result overlay and bounding-box renderer, and re-enable per-class + detections
in `server.py`. The JS skeleton for bounding boxes is gone from index.html now — refer to
git history (commit `33c631b`) for the original renderBoxes() implementation.

## Training data — current status (as of 2026-11-02)
Collection is **in progress**. Images upload automatically to `~/yolo-amd/models/training/`
on m8-server (persisted via the existing `/models` volume mount).

| Class  | Images collected | Target |
|--------|-----------------|--------|
| bolt   | 4               | 200    |
| nut    | 8               | 200    |
| washer | 4               | 200+   |
| screw  | 4               | 200    |

**Data-quality finding (2026-11-02):** two photos in `training/bolt/` were actually nuts
(`bolt_0004_11x`, `bolt_0005_6x`). A local VLM caught it: `qwen3.5-lite` called every crop from those
photos a "nut", and the user confirmed. They were moved to `training/nut/` as `nut_0007_11x` and
`nut_0008_6x`, and `bolt_0006_1x` was renumbered to `bolt_0004_1x`. A mislabeled folder would have
re-created the bolt/nut confusion that broke the old model — spot-check class folders before training.

**Filename sequence rule (do not break):** `/training/upload` names files `{cls}_{count+1}_{N}x.jpg`
from the number of files already in the folder. Folder contents must stay contiguous (count == highest
sequence). If you remove or move a file, renumber the rest, otherwise a later upload overwrites an
existing photo. Training directories are root-owned (written by the container); do file moves from a
throwaway root container mounting `~/yolo-amd/models/training`. The browser-local progress bars on
each phone are not corrected by server-side moves.

(`training/domino/` holds 189 images from the removed domino mode; unused.)
SAM auto-label prototype output for these is in `~/sc-lab/out/` (labels, overlays, `summary.csv`,
raw-mask `cache/`). Root-owned (written by containers).

Check live counts: `find ~/yolo-amd/models/training -type f | sort`

## Known issues / watch-items
- Tunnel URL changes on `cloudflared` container restart (update via ⚙ Server in app).
- High-res iPhone photos (4032×3024) — EXIF transpose is required; keep it on all endpoints.
- Reflective metal + shadows defeat brightness thresholding (not relevant to YOLO path, noted
  for any CV fallback).
- SSH to m8-server: `ssh -i ~/.ssh/id_ed25519 robin@10.0.0.136`

## Next action (when training data collection is complete)
1. Bulk-upload images from `~/yolo-amd/models/training/` to a new Roboflow project.
2. Auto-label: run `tools/autolabel/autolabel.py` (SAM 2) over the folder — the item count in each
   filename (e.g. `_5x.jpg`) tells you how many boxes to expect; images where the found count differs
   are the ones to review. Run it on the 680M iGPU as an overnight batch (`SC_DEVICE=cuda:0`, throwaway
   container from the `yolo-amd` image) or briefly on the 3060 only with the VRAM rules above.
   Grounding DINO ("bolt", "hex nut", "washer", "wood screw") remains an option.
3. Manual review: correct missed/wrong boxes, especially overlapping items.
4. Export as YOLOv8 format, 70/20/10 split.
5. Train on RTX 3060 (`ollama-nvidia` GPU) in the `sc-train` container, batch 8 (see VRAM rules —
   schedule it with the user): `yolo train model=yolo11s.pt data=...` (or `yolov8s.pt`).
6. Evaluate per-class mAP — target washers and bolt/screw confusion specifically.
7. Drop new weights into `/models/<new>/weights/best.pt`, update path in `server.py`,
   rebuild yolo-amd, restart.
8. Re-enable per-class counts + detections array in `server.py`.
9. Restore analyze mode in `index.html` (see git history for renderBoxes implementation).