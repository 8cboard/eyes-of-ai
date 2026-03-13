"""
edge_detector.py — Edge and contour-based object detector for VisionTracker.

Replaces YOLO with OpenCV edge/contour detection for the simplified architecture.
Detects "interesting" regions using Canny edge detection and contour finding,
then returns bounding boxes compatible with the tracker pipeline.

Key features:
  - Canny edge detection with configurable thresholds
  - Contour finding with minimum area filtering
  - Returns DetectionResult-compatible structure
  - No ML models required — pure OpenCV

Usage:
    detector = EdgeDetector(min_area=1000, canny_thresh1=50, canny_thresh2=150)
    result = detector.detect(frame)
    tracked = tracker.update(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Result from edge detection (compatible with tracker pipeline).

    Attributes
    ----------
    xyxy : np.ndarray, shape (N, 4), dtype float32
        Bounding boxes in [x1, y1, x2, y2] pixel coordinates.
    confidences : np.ndarray, shape (N,), dtype float32
        Detection confidence scores (normalized contour area / frame area).
    class_ids : np.ndarray, shape (N,), dtype int32
        Always 0 for edge-detected objects (no semantic classes).
    class_names : list[str]
        Always "object" for edge-detected regions.
    frame_index : int
        Index of the frame processed.
    inference_ms : float
        Wall-clock time (milliseconds) for edge detection.
    """

    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    confidences: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    class_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    class_names: list[str] = field(default_factory=list)
    frame_index: int = 0
    inference_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.xyxy)


# ─────────────────────────────────────────────────────────────────────────────
# Edge Detector class
# ─────────────────────────────────────────────────────────────────────────────

class EdgeDetector:
    """OpenCV-based edge and contour detector.

    Parameters
    ----------
    min_area : int
        Minimum contour area (pixels) to be considered an object.
        Smaller contours are filtered out as noise.
    max_area : int
        Maximum contour area (pixels) to be considered an object.
        Larger contours (e.g., full frame) are filtered out.
    canny_thresh1 : int
        Lower threshold for Canny edge detector.
    canny_thresh2 : int
        Upper threshold for Canny edge detector.
    dilate_iterations : int
        Number of dilation iterations to connect nearby edges.
    blur_kernel : tuple[int, int]
        Gaussian blur kernel size for noise reduction.
    skip_frames : int
        Run detection every N frames (return cached result on skipped frames).
    """

    def __init__(
        self,
        min_area: int = 1000,
        max_area: int = 500000,
        canny_thresh1: int = 50,
        canny_thresh2: int = 150,
        dilate_iterations: int = 2,
        blur_kernel: tuple[int, int] = (5, 5),
        skip_frames: int = 1,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.dilate_iterations = dilate_iterations
        self.blur_kernel = blur_kernel
        self.skip_frames = max(1, skip_frames)

        self._frame_counter: int = 0
        self._last_result: DetectionResult = DetectionResult()

        print(f"[EdgeDetector] min_area={min_area}, max_area={max_area}, "
              f"canny=({canny_thresh1},{canny_thresh2}), skip={skip_frames}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run edge detection on *frame* (or return cached result on skipped frames).

        Parameters
        ----------
        frame : np.ndarray
            BGR or grayscale uint8 image from OpenCV.

        Returns
        -------
        DetectionResult
            Always non-None. On skipped frames, ``frame_index`` of the cached
            result will differ from the current frame counter.
        """
        self._frame_counter += 1

        # Return cached result on skipped frames
        if (self._frame_counter % self.skip_frames) != 0:
            return self._last_result

        t0 = cv2.getTickCount()

        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_thresh1, self.canny_thresh2)

        # Dilate to connect nearby edges
        if self.dilate_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=self.dilate_iterations)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and convert to bounding boxes
        h, w = frame.shape[:2]
        frame_area = h * w

        boxes: list[list[float]] = []
        confidences: list[float] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
                # Confidence based on relative area (normalized)
                confidences.append(min(1.0, area / 50000.0))

        # Compute elapsed time
        t1 = cv2.getTickCount()
        inference_ms = ((t1 - t0) / cv2.getTickFrequency()) * 1000

        if len(boxes) == 0:
            det = DetectionResult(frame_index=self._frame_counter, inference_ms=inference_ms)
        else:
            xyxy = np.array(boxes, dtype=np.float32)
            confs = np.array(confidences, dtype=np.float32)
            class_ids = np.zeros(len(boxes), dtype=np.int32)
            class_names = ["object"] * len(boxes)

            det = DetectionResult(
                xyxy=xyxy,
                confidences=confs,
                class_ids=class_ids,
                class_names=class_names,
                frame_index=self._frame_counter,
                inference_ms=inference_ms,
            )

        self._last_result = det
        return det

    @property
    def frame_count(self) -> int:
        """Total number of frames processed (including skipped)."""
        return self._frame_counter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_display(frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Create a debug visualization showing edges overlaid on the frame.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame.
    edges : np.ndarray
        Binary edge map from Canny.

    Returns
    -------
    np.ndarray
        BGR frame with edges overlaid in green.
    """
    # Convert edges to 3-channel
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_bgr[:, :, 0] = 0  # Remove blue
    edges_bgr[:, :, 2] = 0  # Remove red

    # Blend with original
    blended = cv2.addWeighted(frame, 0.7, edges_bgr, 0.3, 0)
    return blended
