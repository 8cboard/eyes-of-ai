"""
conftest.py — Shared pytest fixtures for VisionTracker tests.
"""
from __future__ import annotations
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def synthetic_frame_720p() -> np.ndarray:
    """720p BGR frame with two coloured rectangles (clear edges for detector)."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[200:400, 300:600] = (0, 128, 255)   # large orange rectangle
    frame[100:200, 800:1000] = (255, 50,  50)  # smaller blue rectangle
    return frame


@pytest.fixture
def cluttered_frame_720p() -> np.ndarray:
    """720p BGR frame with random noise + two solid objects.

    Simulates a textured scene that exposes the fragmentation problem.
    """
    rng = np.random.default_rng(42)
    frame = rng.integers(80, 120, (720, 1280, 3), dtype=np.uint8)
    # Two clearly-defined bright objects on a noisy background
    frame[150:350, 200:500] = 220    # object A
    frame[400:550, 700:900] = 220    # object B
    return frame


@pytest.fixture
def synthetic_detections():
    """DetectionResult with two synthetic non-overlapping boxes."""
    from edge_detector import DetectionResult
    return DetectionResult(
        xyxy=np.array([[300, 200, 600, 400], [800, 100, 1000, 200]], dtype=np.float32),
        confidences=np.array([0.92, 0.75], dtype=np.float32),
        class_ids=np.array([0, 0], dtype=np.int32),
        class_names=["object", "object"],
        frame_index=1,
        inference_ms=12.5,
    )
