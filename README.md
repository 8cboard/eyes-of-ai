# 🎯 VisionTracker — Real-Time Edge Detection, Tracking & Remote LLM Identification

> **Simplified Architecture — Edge Detection + Remote LLM**
>
> Local machine runs edge detection and tracking, sends annotated frames to remote Kaggle/Colab server running vision-language model.

**Quickstart:**
```bash
# Windows
.\install.ps1

# Linux/macOS
bash install.sh

# Set remote server URL (from Colab/Kaggle notebook)
export REMOTE_LLM_URL=https://xxx.ngrok.io  # Linux/macOS
$env:REMOTE_LLM_URL="https://xxx.ngrok.io"  # Windows PowerShell

# Run
python main.py
```

---

## What's New (Simplified Architecture)

This is a major architecture simplification:

| Before | After |
|--------|-------|
| YOLO object detection | OpenCV edge/contour detection |
| ByteTrack + Centroid | Centroid only (ByteTrack hardcoded disabled) |
| Stillness gating | Continuous identification (no stillness logic) |
| Individual crop submission | Full annotated frame submission |
| OpenRouter API support | Remote Kaggle/Colab server only |
| Single model (Nemotron) | Flexible: GGUF or safetensors (<10GB) |

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      VisionTracker Pipeline (Simplified)                     │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────────┐   │
│  │  Camera  │───▶│  Edge    │───▶│Centroid  │───▶│  Draw Colored Boxes │   │
│  │ Capture  │    │ Detector │    │ Tracker  │    │  & Track Labels     │   │
│  │ 720p/30  │    │(Canny + │    │          │    └────────┬────────────┘   │
│  │ frames   │    │ Contours)│    │          │             │                │
│  └──────────┘    └──────────┘    └──────────┘             │                │
│                                                            ▼                │
│  ┌──────────┐    ┌──────────┐                   ┌─────────────────────┐    │
│  │  Cache   │◀───│ Remote   │◀──────────────────│  Annotated Frame    │    │
│  │(per-track│    │  LLM     │                   │  (with boxes drawn) │    │
│  │ TTL dict)│    │ Service  │                   │                     │    │
│  └──────────┘    │          │                   │  Sent every 10th    │    │
│        │         │ Priority │                   │  frame with visible │    │
│        │         │ Queue    │                   │  tracks             │    │
│        │         │          │                   └─────────────────────┘    │
│        │         │   ↓      │                                              │
│        │         │ HTTP POST│                                              │
│        │         │ to remote│                                              │
│        │         │ server   │                                              │
│        │         └──────────┘                                              │
│        │                                                   │                │
│        │              Background Thread (non-blocking)      │                │
│        ▼                                                   ▼                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        UI Overlay (OpenCV)                          │   │
│  │   [Track ID] [Label / "Identifying…"] [Colored Box] [FPS]          │   │
│  │   [● green=done] [● orange=in-progress]                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | File | Role |
|---|---|---|
| **Camera Capture** | `main.py` | Reads 720p frames via `cv2.VideoCapture`. Forces resolution, reduces buffer latency. |
| **Edge Detector** | `edge_detector.py` | OpenCV Canny edge detection + contour finding. Returns bounding boxes for "interesting" regions. No ML models needed. |
| **Tracker** | `tracker.py` | `CentroidTracker` (pure-Python + scipy). ByteTrack is hardcoded disabled for simplified architecture. |
| **Remote LLM Service** | `remote_llm_service.py` | Sends full annotated frames (with colored boxes) to remote server. Receives descriptions for all visible tracks. |
| **Cache** | `remote_llm_service.py` | Per-track TTL cache prevents re-identification until TTL expires. |
| **UI Overlay** | `main.py` | Draws colored bounding boxes, label blocks, status dots, and a HUD bar. |

---

## Remote Server (Kaggle/Colab)

The remote server runs a vision-language model that receives annotated frames and returns descriptions for objects in colored boxes.

### Supported Model Formats

| Format | Library | Best For | Example Models |
|--------|---------|----------|----------------|
| **GGUF** | llama-cpp-python | <10GB quantized models | gemma-3-4b-it-Q4_K_M |
| **safetensors** | transformers | HuggingFace models | Llama-3.2-Vision |

### Quick Setup

1. **Colab (recommended for beginners)**:
   - Open `remote_server/colab_setup.ipynb` in Google Colab
   - Runtime → GPU (T4)
   - Run all cells
   - Copy the **Public URL** from output

2. **Kaggle (more GPU RAM)**:
   - Open `remote_server/kaggle_setup.ipynb` in Kaggle
   - Settings → Accelerator → GPU, Internet → On
   - Add ngrok token to Secrets
   - Run all cells

3. **Run VisionTracker**:
   ```bash
   export REMOTE_LLM_URL=https://xxx.ngrok.io
   python main.py
   ```

---

## Installation

### Windows (PowerShell)
```powershell
.\install.ps1
```

### Linux/macOS
```bash
bash install.sh
```

### Manual
```bash
pip install opencv-python numpy scipy requests python-dotenv
```

---

## Running

```bash
# Basic usage (requires REMOTE_LLM_URL env var)
python main.py

# Pass remote URL directly
python main.py --remote-url https://xxx.ngrok.io

# With optional API key
python main.py --remote-url https://xxx.ngrok.io --remote-key your-key

# Save output to video file
python main.py --save-video output.mp4

# Faster on slow CPUs
python main.py --skip-frames 2 --width 640 --height 360 --grayscale

# Headless mode (no display window)
python main.py --no-display
```

---

## Key Flags

### Remote Server
```
--remote-url URL       Remote Kaggle/Colab server URL (env: REMOTE_LLM_URL)
--remote-key KEY       Optional API key (env: REMOTE_LLM_KEY)
--id-ttl 45            Cache TTL in seconds
--batch-wait 500       ms to wait before dispatching frame
```

### Camera & Detection
```
--input 0              Camera index (0) or path to video file
--width 1280           Capture width
--height 720           Capture height
--grayscale            Force grayscale (faster on slow CPUs)
--skip-frames 1        Run detector every N frames
--min-area 1000        Minimum contour area (pixels)
--max-area 500000      Maximum contour area (pixels)
--canny-low 50         Canny lower threshold
--canny-high 150       Canny upper threshold
```

### Tracker
```
--max-disappeared 20   Frames before track is dropped
```

### Output
```
--no-display           Headless mode (no OpenCV window)
--save-video FILE.mp4  Save annotated video to file
```

---

## Performance Expectations

| Configuration | Expected FPS | Notes |
|---|---|---|
| CPU, edge detection, skip=1 | **15–25 FPS** | Intel i5 / Ryzen 5 |
| CPU, skip=2, grayscale | **25–40 FPS** | Recommended for slow CPUs |
| CPU, 640×360, skip=2 | **35–50 FPS** | Lowest latency mode |

> Identification runs in a background thread — **does not affect FPS**.
> Expect 1–5s per frame depending on remote server hardware.

---

## Privacy & Security

**Annotated frames are sent to YOUR server on Colab/Kaggle via HTTPS.**
- Images never touch third-party APIs (no OpenRouter)
- Your data stays within your control
- ngrok tunnels are encrypted end-to-end
- Add API key for additional security: `--remote-key`

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
├── install.sh              ← Linux/macOS install
├── install.ps1             ← Windows install (NEW)
├── .env.example            ← copy to .env and fill in REMOTE_LLM_URL
├── main.py                 ← entry point (simplified architecture)
├── edge_detector.py        ← NEW: OpenCV edge/contour detection (replaces YOLO)
├── tracker.py              ← CentroidTracker only (ByteTrack hardcoded disabled)
├── remote_llm_service.py   ← NEW: Remote LLM client (replaces OpenRouter)
├── ui_overlay.py           ← OpenCV drawing utilities
├── remote_server/          ← Colab/Kaggle server setup
│   ├── requirements.txt
│   ├── colab_setup.ipynb   ← Updated for GGUF/safetensors
│   └── kaggle_setup.ipynb  ← Updated for GGUF/safetensors
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_smoke.py       ← Updated for new architecture
```

---

## Migration from Old Architecture

If you were using the previous version:

| Old | New |
|-----|-----|
| `--openrouter-key KEY` | `--remote-url URL` (from Colab/Kaggle) |
| `--use-remote-gemma` | Default behavior (only remote server) |
| `--detector yolov8n.pt` | Removed — now uses edge detection |
| `--tracker bytetrack` | Ignored — only CentroidTracker |
| `--still-frames N` | Removed — continuous identification |
| `--batch-size N` | Removed — one frame per request |

---

## Troubleshooting

### "Remote LLM URL is required"
- You must run the remote server first (Colab/Kaggle notebook)
- Set the URL: `export REMOTE_LLM_URL=https://xxx.ngrok.io`

### No objects detected
- Adjust `--min-area` and `--max-area` for your scene
- Adjust `--canny-low` and `--canny-high` for lighting conditions

### Connection errors
- Ensure the Colab/Kaggle notebook is still running
- ngrok free tier sessions expire after ~2 hours — restart the notebook

---

## License

This project follows the same license terms as the original repository.
