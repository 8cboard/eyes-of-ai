# VisionTracker — Real-Time Edge Detection, Tracking & LLM Identification

VisionTracker detects objects via edge/contour detection, tracks them across frames,
and uses a self-hosted remote VLM to identify what each object is.

---

## Architecture

```
Camera → EdgeDetector → Tracker → per-object: draw single box → IDService (bg thread)
                                                                      ↓
                                                          Remote LLM /identify
                                                                      ↓
                                                              UIOverlay (OpenCV)
```

---

## Quick Start

### 1 — Install client dependencies

```bash
bash install.sh
```

### 2 — Start the remote server (Kaggle P100 recommended)

Open `remote_server/kaggle_setup.ipynb` and run all cells.
It will print a URL like `https://abc123.ngrok.io`.

### 3 — Run

```bash
# With identification
python main.py --remote-url https://abc123.ngrok.io

# Camera-only (no identification)
python main.py

# Video file
python main.py --input video.mp4 --remote-url https://abc123.ngrok.io
```

---

## Remote Server — Model

**LLaVA-1.6-Mistral-7B** (recommended — best quality under 10 GB, most stable GGUF)

| File | Size | Source |
|------|------|--------|
| `llava-1.6-mistral-7b.Q4_K_M.gguf` | ~4.4 GB | [cjpais/llava-1.6-mistral-7b-gguf](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf) |
| `mmproj-model-f16.gguf` | ~631 MB | same repo |

> ⚠️ **The mmproj (vision encoder) file is required.** Without it the model is
> text-only and cannot see images. Pass `--mmproj-path` when starting the server.

### Install llama-cpp-python (CUDA 12.1 prebuilt wheels)

```bash
# DO NOT use plain pip install — it compiles CPU-only from source
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Other CUDA versions: cu122  cu124  cu125  cpu (slow)
# Full list: https://abetlen.github.io/llama-cpp-python/whl/
```

### Start the server

```bash
cd remote_server
python server.py \
    --model-path  /path/to/llava-1.6-mistral-7b.Q4_K_M.gguf \
    --mmproj-path /path/to/mmproj-model-f16.gguf
```

### API

```
GET  /health   → {"status":"healthy","model":"...","type":"gguf","version":"3.0.0"}

POST /identify
Body:   {"annotated_image": "<base64 JPEG>"}
Return: {"result": "person"}   ← single common noun
```

---

## Edge Detection — How It Works

The pipeline is designed for **indoor scenes** (desk, room, people):

1. **Bilateral filter** — smooths fabric/wood texture noise while keeping hard edges
2. **Adaptive Canny** — thresholds auto-calibrate to the scene's brightness (no hand-tuning)
3. **Morphological close** — bridges gaps *inside* object outlines so each object becomes one filled shape
4. **Proximity merge** — bounding boxes within N px of each other are unioned (fixes fragmentation)
5. **Aspect-ratio filter** — drops thin-line noise (wires, shadows, furniture edges)

### Key flags

| Flag | Default | Effect |
|------|---------|--------|
| `--merge-distance 30` | 30 px | Increase for cluttered desks; decrease for well-separated objects |
| `--close-kernel 15` | 15 px | Larger = bridges bigger outline gaps (try 20–25 for clothing) |
| `--edge-min-area 500` | 500 px² | Raise to ignore small objects (phones, cups) |
| `--no-auto-canny` | off | Use `--canny-low/--canny-high` manually instead of auto |
| `--skip-frames 2` | 1 | Run detector every 2nd frame for better FPS |

### Indoor tuning presets

```bash
# Fine desk objects (small objects, close together)
python main.py --merge-distance 20 --edge-min-area 300 --close-kernel 10

# Room-scale scene (people, furniture)
python main.py --merge-distance 40 --edge-min-area 1000 --close-kernel 20

# Fast mode (CPU)
python main.py --skip-frames 2 --tracker centroid --merge-distance 30
```

---

## All Flags

### Input
```
--input 0              Camera index or video file path
--width 1280           Capture width
--height 720           Capture height
--grayscale            Force grayscale
```

### Edge Detector
```
--edge-min-area 500    Minimum contour area (px²)
--edge-max-area 100000 Maximum contour area
--merge-distance 30    Proximity merge radius (px); 0 = disabled
--close-kernel 15      Morphological close kernel size
--no-auto-canny        Disable adaptive thresholds
--canny-low 50         Manual Canny low (--no-auto-canny)
--canny-high 150       Manual Canny high (--no-auto-canny)
--use-bg-removal       rembg background removal (slow but cleaner edges)
--skip-frames 1        Detect every N frames
```

### Tracker
```
--tracker bytetrack    bytetrack (default) or centroid
```

### Identification
```
--remote-url URL       Remote server URL (omit to run without identification)
--remote-key KEY       Optional bearer token
--id-ttl 45            Cache TTL in seconds
--id-interval 5        Seconds between re-identification per track
```

### UI
```
--show-velocity        Show velocity indicator
--no-display           Headless mode (no window)
```

---

## Performance

| Config | Expected FPS |
|--------|-------------|
| CPU, ByteTrack, skip=1 | 20–30 |
| CPU, Centroid, skip=2 | 30–45 |
| With bg removal | 10–15 |

---

## Tests

```bash
pytest tests/ -v
```

All tests run offline with synthetic frames — no GPU, webcam, or internet needed.

---

## Privacy

- Frames are sent only to **your own** self-hosted server
- No third-party cloud APIs
- Use HTTPS (ngrok/cloudflared) for encrypted transport
- Add `--remote-key` for bearer-token auth

---

## License

MIT
