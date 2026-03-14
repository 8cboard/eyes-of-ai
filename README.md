# VisionTracker — Real-Time Object Detection, Tracking & LLM Identification

VisionTracker detects objects in real-time, tracks them across frames, and
optionally uses a self-hosted remote VLM to give each track a refined
identification label.

Two detection backends are available:

| Backend | Flag | Boxes | Class labels | Speed (CPU) |
|---------|------|-------|--------------|-------------|
| Edge (classic CV) | `--detector edge` | Good | "object" always | ~40–50 FPS |
| **YOLO (recommended)** | `--detector yolo` | **Excellent** | **Real names from frame 1** | ~25–35 FPS |

> **YOLO is the recommended backend.** It produces much tighter bounding boxes
> and gives you real class names (person, car, laptop …) immediately, before
> any LLM call.  The LLM identification layer is still available for finer
> descriptions when needed.

---

## Architecture

```
Camera → Detector ──────────────────────────────────────────────────────────┐
           │                                                                 │
           │  EdgeDetector  — Bilateral → Canny → morph-close → merge       │
           │  YOLODetector  — Ultralytics YOLO11/v8 inference               │
           │                                                                 │
           ▼                                                                 │
        Tracker  (ByteTrack or CentroidTracker)                              │
           │                                                                 │
           ▼                                                                 │
        IDService  (background thread, optional)                             │
           │  per-object: draw green box → POST /identify to remote LLM     │
           │  LLM refines the label (e.g. "person" → "person in blue jacket")│
           ▼                                                                 │
        UIOverlay  (OpenCV window / video writer) ◄───────────────────────── ┘
```

---

## Quick Start

### 1 — Install client dependencies

```bash
bash install.sh
```

### 2 — Run with YOLO (no server needed)

```bash
# Webcam — YOLO detects + labels everything from frame 1
python main.py --detector yolo

# Specific camera or video file
python main.py --detector yolo --input 1
python main.py --detector yolo --input video.mp4

# Record annotated video
python main.py --detector yolo --record-output session.mp4
```

### 3 — Add remote LLM identification (optional)

Open `remote_server/kaggle_setup.ipynb`, run all cells, copy the URL.

```bash
python main.py --detector yolo --remote-url https://abc123.ngrok.io
```

---

## Detector Backends

### YOLO  (`--detector yolo`)

Uses Ultralytics YOLO11 (or v8) for detection.  The model is downloaded
automatically on first run — no manual setup.

```bash
# Default: YOLO11 nano — 2.6 MB, ~30 FPS on laptop CPU
python main.py --detector yolo

# More accurate (slower):
python main.py --detector yolo --yolo-model yolo11s.pt

# Only detect people and cars (COCO class IDs 0 and 2):
python main.py --detector yolo --yolo-classes 0 2

# Lower confidence threshold to catch more objects:
python main.py --detector yolo --yolo-conf 0.25
```

**Why YOLO for detection but not identification?**

YOLO outputs one of 80 fixed COCO class names (person, car, dog …).  That is
excellent for *detection* but not for *identification* — it cannot tell you
which person, what breed of dog, or what model of laptop.  The remote LLM
does that refinement job on demand, per object.

### Edge  (`--detector edge`)

Classic CV pipeline: Bilateral filter → adaptive Canny → morphological close
→ proximity merge.  No extra dependencies, no model download.

```bash
python main.py --detector edge

# Tuning:
python main.py --detector edge --merge-distance 40 --edge-min-area 1000
```

---

## All Flags

### Input
```
--input 0              Camera index or video file path
--width 1280           Capture width
--height 720           Capture height
--grayscale            Force grayscale display (ID service always gets color)
```

### Detector
```
--detector edge|yolo   Detection backend (default: edge)
--skip-frames 2        Run detector every N frames (default 2, shared by both)
```

### YOLO options
```
--yolo-model yolo11n.pt   Model name or local path (auto-downloaded)
--yolo-conf 0.35          Confidence threshold (lower = more detections)
--yolo-iou 0.45           NMS IoU threshold
--yolo-imgsz 640          Inference image size (320=faster, 640=default)
--yolo-classes 0 2 7      Filter to specific COCO class IDs (omit for all 80)
--yolo-device cpu|cuda    Force device (auto-detected if omitted)
```

### Edge detector options
```
--edge-min-area 500    Minimum contour area (px²)
--edge-max-area 100000 Maximum contour area
--merge-distance 30    Proximity merge radius (px); 0 = disabled
--close-kernel 15      Morphological close kernel size
--no-auto-canny        Disable adaptive thresholds
--canny-low 50         Manual Canny low threshold
--canny-high 150       Manual Canny high threshold
--use-bg-removal       Enable rembg background removal (requires rembg)
```

### Tracker
```
--tracker bytetrack|centroid   Tracker backend (default: bytetrack)
```

### Identification (optional)
```
--remote-url URL       Remote server URL (omit to disable identification)
--remote-key KEY       Optional bearer token
--id-ttl 45            Cache TTL in seconds
--id-interval 5        Seconds between re-identification per track
```

### Output
```
--show-velocity        Show velocity indicator
--no-display           Headless mode (no window)
--record-output PATH   Save annotated video (e.g. output.mp4)
```

---

## Remote Server — Model

**LLaVA-1.6-Mistral-7B** (recommended)

| File | Size | Source |
|------|------|--------|
| `llava-1.6-mistral-7b.Q4_K_M.gguf` | ~4.4 GB | [cjpais/llava-1.6-mistral-7b-gguf](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf) |
| `mmproj-model-f16.gguf` | ~631 MB | same repo |

> ⚠️ The mmproj (vision encoder) file is required — without it the model is text-only.

```bash
cd remote_server
python server.py \
    --model-path  /path/to/llava-1.6-mistral-7b.Q4_K_M.gguf \
    --mmproj-path /path/to/mmproj-model-f16.gguf
```

---

## Performance

| Config | Expected FPS |
|--------|-------------|
| YOLO nano, ByteTrack, skip=2 | 25–35 |
| YOLO nano, Centroid, skip=2 | 30–40 |
| YOLO small, ByteTrack, skip=2 | 15–25 |
| Edge, ByteTrack, skip=2 | 35–50 |
| Edge, Centroid, skip=2 | 40–55 |

> `--skip-frames 2` (default) runs the detector on every other frame.
> The tracker interpolates bounding boxes on skipped frames, so there is no
> visible stutter at typical frame rates.

---

## COCO Class IDs (for --yolo-classes)

Common IDs: `0` person · `1` bicycle · `2` car · `3` motorcycle · `5` bus ·
`7` truck · `14` bird · `15` cat · `16` dog · `24` backpack · `26` handbag ·
`39` bottle · `41` cup · `62` tv · `63` laptop · `64` mouse · `66` keyboard ·
`67` phone · `73` book · `74` clock · `76` scissors

Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

---

## Tests

```bash
pytest tests/ -v
```

All tests run offline — no GPU, webcam, internet, or model download needed.
YOLO is tested via a lightweight mock so `ultralytics` does not need to be
installed in the test environment.

---

## Privacy

- Frames are sent only to **your own** self-hosted server
- No third-party cloud APIs
- Use HTTPS (ngrok/cloudflared) for encrypted transport
- Add `--remote-key` for bearer-token auth

---

## License

MIT
