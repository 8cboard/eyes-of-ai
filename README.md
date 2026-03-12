# 🎯 VisionTracker — Real-Time Object Detection, Tracking & LLM Identification

> **One-line quickstart (OpenRouter):**
> ```bash
> bash install.sh && export OPENROUTER_API_KEY=sk-or-v1-... && python main.py
> ```
>
> **One-line quickstart (Self-hosted Gemma):**
> ```bash
> bash install.sh && python main.py --use-remote-gemma --remote-gemma-url https://xxx.ngrok.io
> ```

**Two identification backends:**
1. **OpenRouter (free tier)** → `nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free`
   - No credit card needed. Get a key at **https://openrouter.ai**
   - ~20 RPM rate limit, ~52 effective IDs/min with batching

2. **Self-hosted Gemma 3 4B** (recommended for privacy)
   - Run on Google Colab or Kaggle free tier
   - No rate limits, your data stays private
   - Setup: See `remote_server/README.md`

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           VisionTracker Pipeline                             │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────────┐   │
│  │  Camera  │───▶│ Detector │───▶│ Tracker  │───▶│ Stillness Detector  │   │
│  │ Capture  │    │(YOLOv8n) │    │(ByteTrack│    │  (IoU + velocity)   │   │
│  │ 720p/30  │    │every N   │    │ /centroid│    │  per-track history  │   │
│  │ frames   │    │ frames)  │    │ fallback)│    └────────┬────────────┘   │
│  └──────────┘    └──────────┘    └──────────┘             │ stable?        │
│                                                            ▼                │
│  ┌──────────┐    ┌──────────┐                   ┌─────────────────────┐    │
│  │  Cache   │◀───│  Cropper │◀──────────────────│   ID Service        │    │
│  │(per-track│    │(bbox crop│                   │                     │    │
│  │ TTL dict)│    │ + resize)│                   │ Priority Queue      │    │
│  └──────────┘    └──────────┘                   │ Token Bucket (RPM)  │    │
│        │                                         │ Batch Dispatcher    │    │
│        │                                         │   ↓                 │    │
│        │                                         │ OpenRouter API      │    │
│        │                                         │ Nemotron 12B VL     │    │
│        │                                         │ (free, batched)     │    │
│        │                                         └────────┬────────────┘    │
│        │              Background Thread (non-blocking)    │                │
│        ▼                                                   ▼                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        UI Overlay (OpenCV)                          │   │
│  │   [Track ID] [Label / "Identifying… 42%"] [Confidence] [FPS]       │   │
│  │   [████░░░░] progress bar   ● green=done   ● orange=in-progress    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | File | Role |
|---|---|---|
| **Camera Capture** | `main.py` | Reads 720p frames via `cv2.VideoCapture`. Forces resolution, reduces buffer latency. |
| **Detector** | `detector.py` | YOLOv8n (or configurable variant) runs every N frames. Frame-skip keeps the main loop fast on CPU. |
| **Tracker** | `tracker.py` | `ByteTrackWrapper` (via `supervision`) is primary; `CentroidTracker` (pure-Python + scipy) is the fallback. Both expose the same `update(det)` API. |
| **Stillness Detector** | `stability.py` | Per-track deques of centroid positions and IoU values. Fires identification only when velocity < threshold AND IoU > threshold for M consecutive frames. Also provides an optical-flow variant for moving cameras. |
| **Cropper** | `main.py` | Slices the bbox region from the raw frame with padding, copies it for the background thread. |
| **ID Service** | `id_service.py` / `gemma_remote_service.py` | Single background dispatcher thread. Collects crops into batches, fires API calls (OpenRouter or remote Gemma server). Writes results to a shared `progress` dict. |
| **Cache** | `id_service.py` / `gemma_remote_service.py` | Per-track TTL cache (`IdentificationCache`). Prevents re-identification until the TTL expires. |
| **UI Overlay** | `ui_overlay.py` | Draws coloured bounding boxes, label blocks, progress bars, status dots, and a HUD bar. Reads from the shared progress dict — never blocks. |

---

## Rate Limit Strategy

OpenRouter free models allow ~20 requests/minute. VisionTracker handles this with three layers:

```
Layer 1 — Batch dispatch
  Up to 4 crops packed into ONE API call.
  The model returns a numbered list (one description per image).
  → 1 request identifies up to 4 objects simultaneously.
  → At ~13 RPM × batch 4 = ~52 effective identifications/minute.

Layer 2 — Token-bucket rate limiter (id_service.py: TokenBucket)
  Refills at 0.22 tokens/sec (13.2 RPM), burst capacity 3.
  Single dispatcher thread acquires a token before every HTTP call.
  → Can never exceed the rate limit, even with many objects.

Layer 3 — HTTP 429 back-off + re-queue
  On rate limit: sleep Retry-After seconds, re-queue crops at front
  of the priority queue (priority = now − 1000).
  → Zero items are ever lost.
```

---

## Installation

### Option A — Quick script
```bash
bash install.sh
```

### Option B — Manual
```bash
pip install -r requirements.txt
# Pre-download YOLOv8n weights (6 MB, automatic on first run):
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Option C — Conda
```bash
conda env create -f environment.yml
conda activate visiontracker
# Install torch for your platform — see environment.yml comments
```

---

## Getting an OpenRouter API Key (free)

1. Go to **https://openrouter.ai** and create a free account
2. Navigate to **Keys** → **Create Key**
3. No credit card needed — the `:free` model tier is completely free
4. Set the key:
   ```bash
   export OPENROUTER_API_KEY=sk-or-v1-...
   # Or add to .env file:
   echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
   ```

---

## Self-Hosted Gemma 3 4B (Privacy-First Option)

For fully private identification without sending images to third-party APIs:

### Quick Setup

1. **Open the Colab notebook** (recommended for beginners):
   ```
   remote_server/colab_setup.ipynb
   ```
   - Upload to Google Colab
   - Runtime → GPU (T4)
   - Follow the 6 cells to start server
   - Copy the **Public URL** from output

2. **Or use Kaggle** (more GPU RAM):
   ```
   remote_server/kaggle_setup.ipynb
   ```
   - Upload to Kaggle
   - Settings → Accelerator → GPU, Internet → On
   - Add ngrok token to Secrets
   - Run all cells

3. **Run VisionTracker with your server**:
   ```bash
   python main.py --use-remote-gemma --remote-gemma-url https://xxx.ngrok.io
   ```

### Why Self-Host?

| Feature | OpenRouter Free | Self-Hosted Gemma |
|---------|-----------------|-------------------|
| **Rate Limit** | ~20 RPM | Unlimited |
| **Privacy** | Sends images to third-party | Your server, your data |
| **Cost** | Free | Free (Colab/Kaggle tier) |
| **Latency** | 5-15s per batch | 1-3s after warmup |
| **Setup** | API key only | ~5 minute setup |

See `remote_server/README.md` for detailed setup instructions.

---

## Running

```bash
# Standard with OpenRouter (reads key from env var)
python main.py

# Pass OpenRouter key directly
python main.py --openrouter-key sk-or-v1-...

# Use self-hosted Gemma instead
python main.py --use-remote-gemma --remote-gemma-url https://xxx.ngrok.io

# Faster on CPU: skip every other frame, small detector
python main.py --skip-frames 3 --detector yolov8n.pt --width 640 --height 360

# Use centroid tracker (no supervision needed)
python main.py --tracker centroid

# GPU acceleration (auto-detected, or force)
python main.py --device cuda

# Single-file all-in-one version
python single_file_demo.py --openrouter-key sk-or-v1-...
```

---

## Key Flags

### OpenRouter (Default)
```
--openrouter-key KEY   OpenRouter API key (also: OPENROUTER_API_KEY env var)
--model SLUG           Override model slug (default: nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free)
```

### Self-Hosted Gemma (Privacy)
```
--use-remote-gemma          Use self-hosted Gemma instead of OpenRouter
--remote-gemma-url URL      Server URL (also: REMOTE_GEMMA_URL env var)
--remote-gemma-key KEY      Optional API key (also: REMOTE_GEMMA_KEY env var)
```

### Camera & Detection
```
--input 0              Camera index (0) or path to video file
--width 1280           Capture width
--height 720           Capture height
--detector yolov8n.pt  Detector model (n/s/m/x)
--skip-frames 2        Run detector every N frames (default: 2)
--conf 0.35            Detection confidence threshold
--imgsz 640            Detector input resolution (320/640/1280)
--tracker bytetrack    Tracker: bytetrack or centroid
```

### Stillness & Identification
```
--still-frames 10      M: consecutive below-threshold frames to trigger ID
--still-window 15      N: history window for stillness
--id-ttl 45            Cache TTL in seconds
--batch-size 4         Max crops per API call (1–4 for OpenRouter, 1–16 for remote)
--batch-wait 1500      ms to wait for a fuller batch
```

### System
```
--device cpu/cuda/mps  Compute device (default: auto)
--grayscale            Force grayscale (faster on slow CPUs)
--optical-flow         Use optical-flow stillness (for moving cameras)
--show-velocity        Show velocity indicator on boxes
--no-display           Headless mode (no OpenCV window)
```

---

## Performance Expectations

| Configuration | Expected FPS | Notes |
|---|---|---|
| CPU, YOLOv8n, ByteTrack, skip=2 | **15–22 FPS** | MacBook M1 / Intel i7 |
| CPU, YOLOv8n, centroid, skip=2 | **18–26 FPS** | Lighter tracker |
| CPU, 640×360, grayscale, skip=3 | **28–40 FPS** | Lowest latency |
| GPU (CUDA), YOLOv8s, skip=1 | **50–80 FPS** | RTX 3060 class |

> Identification always runs in a background thread — **does not affect FPS**.
> Expect 5–15 s per batch (OpenRouter network latency + model inference).

### CPU Speed Tips
```bash
# Fastest CPU config
python main.py --skip-frames 3 --width 640 --height 360 --grayscale --tracker centroid

# Balanced
python main.py --skip-frames 2 --width 1280 --height 720 --tracker bytetrack
```

---

## Switching Detector Models

| Model | Size | mAP50-95 | Use case |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | 37.3 | Best for CPU |
| `yolov8s.pt` | 22 MB | 44.9 | CPU/GPU balance |
| `yolov8m.pt` | 52 MB | 50.2 | GPU recommended |
| `yolov8x.pt` | 131 MB | 53.9 | GPU only |

---

## Privacy & Security

### OpenRouter Backend
> **Crops are sent over the internet to OpenRouter servers.**
> - OpenRouter may log requests per their [Privacy Policy](https://openrouter.ai/privacy)
> - Do **not** use with sensitive footage (faces, private spaces, etc.)
> - The `:free` model tier is rate-limited; verify current terms at openrouter.ai

### Self-Hosted Gemma Backend (Private)
> **Crops are sent to YOUR server on Colab/Kaggle via HTTPS.**
> - Images never touch third-party APIs
> - Your data stays within your control
> - ngrok tunnels are encrypted end-to-end
> - Add API key for additional security: `--remote-gemma-key`

### Fully Offline Option
For 100% offline operation, replace `id_service.py` with a local Ollama/LLaVA backend running on the same machine.

---

## Tests
```bash
pytest tests/ -v
```

---

## File Layout

```
vision_tracker/
├── README.md
├── requirements.txt
├── environment.yml
├── install.sh
├── .env.example              ← copy to .env and fill in your key
├── main.py                   ← entry point (modular, supports both backends)
├── detector.py               ← YOLOv8 wrapper + frame-skip
├── tracker.py                ← ByteTrack + centroid fallback
├── stability.py              ← stillness policy + optical flow
├── id_service.py             ← OpenRouter batched ID + rate limiter + cache
├── gemma_remote_service.py   ← Self-hosted Gemma 3 4B HTTP client
├── ui_overlay.py             ← OpenCV drawing + progress bars
├── single_file_demo.py       ← all-in-one self-contained runnable
├── remote_server/            ← Colab/Kaggle server setup
│   ├── README.md
│   ├── requirements.txt
│   ├── colab_setup.ipynb     ← Google Colab notebook
│   └── kaggle_setup.ipynb    ← Kaggle notebook
└── tests/
    ├── __init__.py
    ├── conftest.py           ← Shared pytest fixtures
    ├── test_smoke.py         ← Smoke tests (no network, no webcam required)
    └── test_gemma_remote.py  ← Tests for remote Gemma service
```
