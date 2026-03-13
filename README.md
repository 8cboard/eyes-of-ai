# VisionTracker — Real-Time Edge Detection, Tracking & LLM Identification

VisionTracker is a real-time computer vision pipeline that detects objects via edge/contour detection, tracks them across frames, and uses a remote LLM to identify objects by their colored bounding boxes.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VisionTracker Pipeline                               │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────────┐    │
│  │  Camera  │───▶│  Edge    │───▶│ Tracker  │───▶│  Draw Colored       │    │
│  │ Capture  │    │ Detector │    │(ByteTrack│    │  Bounding Boxes     │    │
│  │ 720p/30  │    │(Canny + │    │ /centroid│    │  (visual IDs)       │    │
│  │ frames   │    │contours) │    │ fallback)│    └─────────┬───────────┘    │
│  └──────────┘    └──────────┘    └──────────┘              │                │
│                                                            ▼                │
│  ┌──────────┐    ┌──────────┐                   ┌─────────────────────┐    │
│  │  Cache   │◀───│ Remote   │◀──────────────────│  ID Service         │    │
│  │(per-track│    │ LLM      │                   │                     │    │
│  │ TTL dict)│    │ Server   │                   │ Priority Queue      │    │
│  └──────────┘    │(GGUF/    │                   │ Batch Dispatcher    │    │
│        │         │Safeten.) │                   │   ↓                 │    │
│        │         └──────────┘                   │ HTTP POST           │    │
│        │                                         │ Annotated Frame     │    │
│        │                                         └────────┬────────────┘    │
│        │              Background Thread (non-blocking)    │                │
│        ▼                                                   ▼                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        UI Overlay (OpenCV)                          │   │
│  │   [Colored Box] [Label / "Identifying… 42%"] [FPS]                 │   │
│  │   [████░░░░] progress bar   ● green=done   ● orange=in-progress    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Edge/Contour Detection**: Canny edge detection finds objects without requiring YOLO or training data
- **Multi-Object Tracking**: ByteTrack (via supervision) with CentroidTracker fallback
- **Visual Identification**: Objects are identified by colored bounding boxes sent to LLM
- **Remote LLM Support**: Supports both GGUF (llama-cpp-python) and Safetensors (transformers) models
- **Optional Background Removal**: Use `rembg` to remove backgrounds before edge detection
- **Windows Support**: PowerShell install script included

## Installation

### Linux/macOS
```bash
bash install.sh
```

### Windows
```powershell
.\install.ps1
```

### Manual
```bash
pip install -r requirements.txt
```

## Remote LLM Server Setup

The remote server supports both GGUF and Safetensors vision-language models under 10GB.

### Quick Start with GGUF (Recommended for Colab)

1. **Download a vision-capable GGUF model** (e.g., Gemma 3 4B IT):
   ```bash
   wget https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf
   ```

2. **Start the server**:
   ```bash
   cd remote_server
   pip install -r requirements.txt
   python server.py --model-path /path/to/gemma-3-4b-it-Q4_K_M.gguf
   ```

3. **For public access** (use ngrok or cloudflared):
   ```bash
   ngrok http 8000
   ```

### Using Safetensors Models

```bash
python server.py --model-path /path/to/safetensors-model-dir
```

Supported models include:
- microsoft/Phi-4-multimodal-instruct
- meta-llama/Llama-3.2-11B-Vision-Instruct
- google/gemma-3-4b-it (via transformers)

### Server API Endpoints

```
GET /health
Response: {"status": "healthy", "model": "...", "type": "gguf|safetensors"}

POST /identify
Request: {
  "annotated_image": "base64_jpeg",
  "color_map": {"red": 1, "blue": 2, "green": 3}
}
Response: {
  "results": [
    {"track_id": 1, "description": "person"},
    {"track_id": 2, "description": "car"}
  ]
}
```

## Running VisionTracker

### Basic usage
```bash
python main.py --remote-url https://your-server.ngrok.io
```

### With video file
```bash
python main.py --input test_video.mp4 --remote-url https://your-server.ngrok.io
```

### With background removal
```bash
python main.py --remote-url https://your-server.ngrok.io --use-bg-removal
```

### CPU-optimized settings
```bash
python main.py --remote-url https://your-server.ngrok.io \
  --skip-frames 2 \
  --edge-min-area 1000 \
  --tracker centroid
```

## Key Flags

### Camera & Detection
```
--input 0                 Camera index or video file path
--width 1280              Capture width
--height 720              Capture height
--edge-min-area 500       Minimum contour area (pixels)
--edge-max-area 100000    Maximum contour area (pixels)
--canny-low 50            Canny lower threshold
--canny-high 150          Canny upper threshold
--use-bg-removal          Enable background removal
--skip-frames 1           Run detector every N frames
```

### Tracking
```
--tracker bytetrack       Tracker type: bytetrack or centroid
```

### Identification
```
--remote-url URL          Remote LLM server URL
--remote-key KEY          API key for server (optional)
--id-ttl 45               Cache TTL in seconds
--id-interval 5           Seconds between ID attempts per track
--batch-size 8            Max tracks per API call
--batch-wait 1000         ms to wait for fuller batch
```

### System
```
--grayscale               Force grayscale processing
--no-display              Headless mode
```

## How It Works

1. **Edge Detection**: The frame is processed with Canny edge detection and contours are found
2. **Tracking**: Contours are tracked across frames using ByteTrack or CentroidTracker
3. **Colored Boxes**: Each tracked object gets a unique colored bounding box (20 colors cycle)
4. **LLM Identification**: The annotated frame with colored boxes is sent to the remote LLM
5. **Response Parsing**: The LLM returns single common nouns for each colored box (e.g., "person", "car")
6. **UI Display**: Results are shown on-screen with colored boxes and labels

## LLM Prompt Engineering

The remote server sends prompts engineered for single common noun responses:

```
You see N colored boxes on objects in an image.
The colors and their object numbers are: red (object #1), blue (object #2), ...

Identify each object with exactly ONE common noun.
Use simple everyday words like: person, car, dog, chair, table, etc.

Respond in this exact format (one per line):
1. [common noun for the red box]
2. [common noun for the blue box]
...
```

## File Layout

```
visiontracker/
├── README.md
├── requirements.txt
├── install.sh              # Linux/macOS install
├── install.ps1             # Windows install
├── main.py                 # Entry point
├── edge_detector.py        # Edge/contour detection
├── tracker.py              # ByteTrack + CentroidTracker
├── id_service.py           # Remote LLM client
├── ui_overlay.py           # OpenCV visualization
├── remote_server/          # Remote LLM server
│   ├── server.py           # FastAPI server (GGUF + Safetensors)
│   ├── requirements.txt
│   ├── colab_setup.ipynb   # Google Colab notebook
│   └── kaggle_setup.ipynb  # Kaggle notebook
└── tests/
    ├── test_smoke.py
    └── conftest.py
```

## Performance Tips

| Configuration | Expected FPS | Notes |
|--------------|--------------|-------|
| CPU, ByteTrack, skip=1 | 20-30 FPS | Modern CPU |
| CPU, Centroid, skip=2 | 30-45 FPS | Lightweight tracker |
| With background removal | 10-15 FPS | rembg is slower |

- Higher `edge-min-area` = fewer contours = faster
- `skip-frames` trades detection frequency for FPS
- Identification runs in background — doesn't affect FPS

## Privacy

- Video frames are sent to your **self-hosted** remote LLM server
- No third-party APIs (no OpenRouter, no cloud services)
- Use HTTPS/ngrok for encrypted transport
- Add `--remote-key` for additional authentication

## Requirements

- Python 3.10+
- OpenCV with GUI support
- 4GB+ RAM
- For remote server: GPU optional but recommended (T4, P100, or better)

## License

MIT License
