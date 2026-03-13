# 🎯 VisionTracker — Real-Time Object Detection, Tracking & Remote LLM Identification

VisionTracker performs real-time edge detection, multi-object tracking, and sends annotated frames to a remote vision-language model for object identification.

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

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VisionTracker Pipeline                               │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────────┐   │
│  │  Camera  │───▶│  Edge    │───▶│ Tracker  │───▶│  Draw Colored Boxes │   │
│  │ Capture  │    │ Detector │    │(ByteTrack│    │  & Object Labels    │   │
│  │          │    │(Canny + │    │/Centroid)│    └────────┬────────────┘   │
│  │          │    │ Contours)│    │          │             │                │
│  └──────────┘    └──────────┘    └──────────┘             │                │
│                                                            ▼                │
│  ┌──────────┐    ┌──────────┐                   ┌─────────────────────┐    │
│  │  Cache   │◀───│ Remote   │◀──────────────────│  Annotated Frame    │    │
│  │(per-track│    │  LLM     │                   │  (with boxes drawn) │    │
│  │ TTL dict)│    │ Service  │                   │                     │    │
│  └──────────┘    │          │                   │  Sent periodically  │    │
│        │         │ Priority │                   │  with visible tracks│    │
│        │         │ Queue    │                   └─────────────────────┘    │
│        │         │          │                                              │
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
│  │   [Object Name] [Colored Box] [Status Dot] [FPS]                   │   │
│  │   [● green=done] [● orange=in-progress]                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Role |
|---|---|---|
| **Camera Capture** | `main.py` | Reads frames via OpenCV. Configurable resolution, reduced buffer latency. |
| **Edge Detector** | `edge_detector.py` | Canny edge detection + contour finding. Detects regions of interest without ML models. |
| **Tracker** | `tracker.py` | `ByteTrack` (via supervision) or `CentroidTracker` (pure Python). Maintains persistent object IDs. |
| **Remote LLM Service** | `remote_llm_service.py` | Sends annotated frames to remote server. Receives object identifications. |
| **Cache** | `remote_llm_service.py` | Per-track TTL cache prevents redundant identifications. |
| **UI Overlay** | `main.py` | Draws colored bounding boxes, object labels, status indicators. |

---

## Remote Server (Kaggle/Colab)

The remote server runs a vision-language model that receives annotated frames and returns object identifications as single common nouns (e.g., "person", "car", "chair").

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

# Use ByteTrack instead of CentroidTracker
python main.py --tracker bytetrack

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

### Tracker
```
--tracker centroid     Tracker type: centroid (default) or bytetrack
--max-disappeared 20   Frames before track is dropped (CentroidTracker only)
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

### Output
```
--no-display           Headless mode (no OpenCV window)
--save-video FILE.mp4  Save annotated video to file
```

---

## Performance

| Configuration | Expected FPS | Notes |
|---|---|---|
| CPU, edge detection, skip=1 | **15–25 FPS** | Intel i5 / Ryzen 5 |
| CPU, skip=2, grayscale | **25–40 FPS** | Recommended for slow CPUs |
| CPU, 640×360, skip=2 | **35–50 FPS** | Lowest latency mode |

Identification runs in a background thread and does not affect FPS. Response time depends on remote server hardware (typically 1–5s per frame).

---

## Privacy & Security

Annotated frames are sent to YOUR server on Colab/Kaggle via HTTPS:
- Images never touch third-party APIs
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
├── install.ps1             ← Windows install
├── .env.example            ← Environment variables template
├── main.py                 ← Entry point
├── edge_detector.py        ← OpenCV edge/contour detection
├── tracker.py              ← ByteTrack + CentroidTracker
├── remote_llm_service.py   ← Remote LLM client
├── ui_overlay.py           ← OpenCV drawing utilities
├── remote_server/          ← Colab/Kaggle server setup
│   ├── requirements.txt
│   ├── colab_setup.ipynb
│   └── kaggle_setup.ipynb
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_smoke.py
```

---

## Troubleshooting

### "Remote LLM URL is required"
- Run the remote server first (Colab/Kaggle notebook)
- Set the URL: `export REMOTE_LLM_URL=https://xxx.ngrok.io`

### No objects detected
- Adjust `--min-area` and `--max-area` for your scene
- Adjust `--canny-low` and `--canny-high` for lighting conditions

### Connection errors
- Ensure the Colab/Kaggle notebook is still running
- ngrok free tier sessions expire after ~2 hours — restart the notebook

### ByteTrack not available
- Install supervision: `pip install supervision`
- Falls back to CentroidTracker if not installed

---

## License

This project follows the same license terms as the original repository.
