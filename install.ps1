# install.ps1 — VisionTracker install script for Windows
# Run with: PowerShell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VisionTracker — Windows Install Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check Python ─────────────────────────────────────────────────────────
Write-Host "[1/3] Checking Python installation..." -ForegroundColor Yellow

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found. Please install Python 3.9+ from https://python.org" -ForegroundColor Red
    exit 1
}

$pythonVersion = & $pythonCmd.Source --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# ── 2. Check for pip ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/3] Installing Python packages..." -ForegroundColor Yellow

& $pythonCmd.Source -m pip install --upgrade pip wheel setuptools

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upgrade pip" -ForegroundColor Red
    exit 1
}

# ── 3. Install requirements ─────────────────────────────────────────────────
$reqFile = Join-Path $PSScriptRoot "requirements.txt"
if (Test-Path $reqFile) {
    & $pythonCmd.Source -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "WARNING: requirements.txt not found, installing core packages..." -ForegroundColor Yellow
    & $pythonCmd.Source -m pip install opencv-python numpy scipy requests python-dotenv pytest
}

Write-Host "  ✓ Packages installed" -ForegroundColor Green

# ── 4. Check for video codecs ───────────────────────────────────────────────
Write-Host ""
Write-Host "[3/3] Checking video codec support..." -ForegroundColor Yellow

# Test OpenCV video capture capability
$testCode = @"
import cv2
print(f"OpenCV version: {cv2.__version__}")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Camera access available")
    cap.release()
else:
    print("⚠ No camera detected (this is OK if using video files)")
"@

& $pythonCmd.Source -c $testCode

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To use VisionTracker:"
Write-Host "  1. Run the remote server notebook in Colab or Kaggle"
Write-Host "  2. Set the remote URL:"
Write-Host "     `$env:REMOTE_LLM_URL = 'https://xxx.ngrok.io'"
Write-Host "  3. Run VisionTracker:"
Write-Host "     python main.py"
Write-Host ""
Write-Host "For a test with a video file:"
Write-Host "     python main.py --input video.mp4"
Write-Host "============================================" -ForegroundColor Cyan
