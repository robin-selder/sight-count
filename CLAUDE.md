# Sight Count — Project Context

## What this is
A single-page HTML camera app for counting hardware items (bolts, nuts, washers, screws) using a YOLO inference API. Deployed to GitHub Pages, designed for phone and tablet use.

## Repo
- GitHub: https://github.com/robin-selder/sight-count
- Live URL: https://robin-selder.github.io/sight-count
- Single file deployment: `index.html` at repo root

## Infrastructure (m8-server — 10.0.0.136)
All services run in Docker under the `m8-compute-stack` compose stack at `~/docker-compose.yml`.

| Container | Port | GPU | Purpose |
|---|---|---|---|
| yolo-amd | 8010→8000 | Radeon 680M (ROCm) | YOLOv8s fastener inference |
| cloudflared | — | — | Cloudflare quick tunnel to yolo-amd |
| ollama-nvidia | 11434 | RTX 3060 | Primary LLM (Qwen3 14.8B) |
| ollama-amd | 11435 | Radeon 680M (Vulkan) | Lightweight LLMs (Moondream, Qwen3 1.7b) |
| whisper-amd | 9000 | Radeon 680M (ROCm) | Whisper transcription |

## YOLO API
- Image: `yolo-amd` (built from `~/yolo-amd/` on m8-server)
- Model: `fasteners-combined-yolov8s` at `/models/fasteners-2/weights/best.pt`
- Endpoints:
  - `GET /health` — returns device, rocm status, model name, class names
  - `POST /count` — accepts multipart `file`, returns `{total, counts, detections[{label, confidence, bbox}]}`
- CORS: `allow_origins=["*"]`

## Cloudflare Tunnel
- Type: Quick tunnel (no account, ephemeral URL)
- Current URL: `https://kiss-harold-gap-rainbow.trycloudflare.com`
- **URL changes on container restart** — update via ⚙ API Config button in the app (saves to localStorage)
- To get current URL: `docker compose logs --tail=20 cloudflared | grep trycloudflare`

## App behaviour
- Home screen → "Start Analyzing" → live camera viewfinder
- Tap shutter button → frame captured → sent to `/count` → result shown with bounding boxes overlaid
- "Retake" to shoot again
- Bounding box colours: bolt=#e8ff47, nut=#47e8ff, screw=#ff8c47, washer=#c847ff

## Known limitations (prototype)
- Washer detection poor (13.2% mAP) — training data was synthetic
- Some bolt/screw class confusion — model needs real hardware photos to improve
- Tunnel URL is ephemeral — will need updating after each cloudflared restart

## Design
- Dark industrial aesthetic, Bebas Neue + DM Mono fonts
- Accent colour: #e8ff47 (yellow-green)
- Responsive: portrait phone and landscape tablet

## Next steps
- Retrain model with real hardware photos
- Move to named Cloudflare tunnel (stable URL) when out of prototype phase
- Add per-class breakdown to results view
