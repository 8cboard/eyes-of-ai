# install.ps1 — VisionTracker install script for Windows
# Run in PowerShell: .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VisionTracker — Install Script (Windows)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check Python installation ─────────────────────────────────────────────
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Found $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found. Please install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# ── 2. Create virtual environment (optional) ─────────────────────────────────
Write-Host ""
Write-Host "[2/4] Setting up virtual environment..." -ForegroundColor Yellow

$venvPath = ".venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
    Write-Host "  ✓ Created virtual environment at .venv" -ForegroundColor Green
} else {
    Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"
Write-Host "  ✓ Activated virtual environment" -ForegroundColor Green

# ── 3. Install Python packages ───────────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Installing Python packages..." -ForegroundColor Yellow

python -m pip install --upgrade pip wheel setuptools

# Detect CUDA for torch installation
try {
    $nvidiaSmi = nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  → NVIDIA GPU detected. Installing torch with CUDA 12.1..." -ForegroundColor Cyan
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    } else {
        throw "No NVIDIA GPU"
    }
} catch {
    Write-Host "  → No NVIDIA GPU detected. Installing CPU-only torch..." -ForegroundColor Cyan
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# Install requirements
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "  ✓ Installed requirements" -ForegroundColor Green
} else {
    Write-Host "  ✗ requirements.txt not found" -ForegroundColor Red
    exit 1
}

# ── 4. Optional: Download test video ─────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] Optional: Download test video..." -ForegroundColor Yellow

$testVideoUrl = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
$testVideoPath = "test_video.mp4"

if (-not (Test-Path $testVideoPath)) {
    $download = Read-Host "Download sample test video? (y/n)"
    if ($download -eq "y" -or $download -eq "Y") {
        try {
            Invoke-WebRequest -Uri $testVideoUrl -OutFile $testVideoPath -UseBasicParsing
            Write-Host "  ✓ Downloaded test_video.mp4" -ForegroundColor Green
        } catch {
            Write-Host "  ⚠ Failed to download test video" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  ✓ Test video already exists" -ForegroundColor Green
}

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To use VisionTracker:"
Write-Host ""
Write-Host "  1. Activate the virtual environment:" -ForegroundColor Cyan
Write-Host "     .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  2. Set up your remote LLM server (see remote_server/README.md)"
Write-Host ""
Write-Host "  3. Run VisionTracker:" -ForegroundColor Cyan
Write-Host "     python main.py --remote-url https://your-server-url"
Write-Host ""
Write-Host "  Or with test video:" -ForegroundColor Cyan
Write-Host "     python main.py --input test_video.mp4 --remote-url https://your-server-url"
Write-Host ""
Write-Host "Optional flags:" -ForegroundColor Cyan
Write-Host "     --use-bg-removal    Enable background removal (requires rembg)"
Write-Host "     --edge-min-area 500 Filter small contours"
Write-Host "     --id-interval 5     Identification interval in seconds"
Write-Host ""
