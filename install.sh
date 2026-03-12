#!/usr/bin/env bash
# install.sh — VisionTracker install script
# Works on: Ubuntu/Debian, macOS (Homebrew), or any system with pip
set -euo pipefail

echo "============================================"
echo "  VisionTracker — Install Script"
echo "  Backend: OpenRouter / Nemotron Nano 12B VL"
echo "============================================"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
OS="$(uname -s)"
if [[ "$OS" == "Linux" ]]; then
    echo "[1/3] Installing system packages (apt)..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        python3-pip python3-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
        ffmpeg wget curl git
elif [[ "$OS" == "Darwin" ]]; then
    echo "[1/3] Installing system packages (brew)..."
    if ! command -v brew &>/dev/null; then
        echo "  Homebrew not found. Install from https://brew.sh first."
        exit 1
    fi
    brew install ffmpeg wget
else
    echo "[1/3] Skipping system packages (unknown OS: $OS)"
fi

# ── 2. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[2/3] Installing Python packages..."
pip install --upgrade pip wheel setuptools

# Detect GPU and install appropriate torch
if python3 -c "import subprocess; r=subprocess.run(['nvidia-smi'],capture_output=True); exit(0 if r.returncode==0 else 1)" 2>/dev/null; then
    echo "  → NVIDIA GPU detected. Installing torch with CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$OS" == "Darwin" ]] && python3 -c "import platform; exit(0 if 'arm' in platform.machine() else 1)" 2>/dev/null; then
    echo "  → Apple Silicon detected. Installing torch with MPS support..."
    pip install torch torchvision
else
    echo "  → No GPU detected. Installing CPU torch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

# ── 3. Pre-download YOLOv8n weights ──────────────────────────────────────────
echo ""
echo "[3/3] Pre-downloading YOLOv8n weights (~6 MB)..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('  ✓ YOLOv8n ready')
"

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  Set your free OpenRouter API key:"
echo "    export OPENROUTER_API_KEY=sk-or-v1-..."
echo "  Get a free key at: https://openrouter.ai"
echo "  (no credit card needed for :free models)"
echo ""
echo "  Run:"
echo "    python main.py"
echo "  or:"
echo "    python single_file_demo.py"
echo "============================================"
