"""
conftest.py — Shared pytest fixtures for VisionTracker tests.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pytest

# Add project root to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def synthetic_frame_720p() -> np.ndarray:
    """Return a synthetic 720p BGR frame with a colored rectangle."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw a simple colored rectangle (simulated "object")
    frame[200:400, 300:600] = (0, 128, 255)  # orange rectangle
    frame[100:200, 800:1000] = (255, 0, 0)   # blue rectangle
    return frame


@pytest.fixture
def synthetic_crop() -> np.ndarray:
    """Return a small synthetic BGR crop (simulated object region)."""
    crop = np.zeros((100, 80, 3), dtype=np.uint8)
    crop[20:80, 10:70] = (100, 200, 50)
    return crop


@pytest.fixture
def synthetic_detections():
    """Return a fake DetectionResult with two synthetic boxes."""
    from detector import DetectionResult
    return DetectionResult(
        xyxy=np.array([[300, 200, 600, 400], [800, 100, 1000, 200]], dtype=np.float32),
        confidences=np.array([0.92, 0.75], dtype=np.float32),
        class_ids=np.array([0, 73], dtype=np.int32),  # person, book
        class_names=["person", "book"],
        frame_index=1,
        inference_ms=12.5,
    )
