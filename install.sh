#!/usr/bin/env bash
# install.sh — VisionTracker install script
# Installs all dependencies for the client (main.py) on Linux or macOS.
# For the remote server see: remote_server/README.md
set -euo pipefail

echo "============================================"
echo "  VisionTracker — Install Script"
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

# Detect GPU for torch
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

# ── 3. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying installation..."
python3 -c "
import cv2, numpy, scipy
print('  ✓ opencv   ', cv2.__version__)
print('  ✓ numpy    ', numpy.__version__)
print('  ✓ scipy loaded')
try:
    import supervision
    print('  ✓ supervision loaded (ByteTrack available)')
except ImportError:
    print('  ⚠ supervision not found — CentroidTracker will be used as fallback')
"

echo ""
echo "============================================"
echo "  Client installation complete!"
echo ""
echo "  Camera-only (no identification):"
echo "    python main.py"
echo ""
echo "  With remote LLM identification:"
echo "    python main.py --remote-url https://your-server.ngrok.io"
echo ""
echo "  Record annotated video:"
echo "    python main.py --record-output session.mp4"
echo ""
echo "  For remote server setup, see: remote_server/README.md"
echo "============================================"
