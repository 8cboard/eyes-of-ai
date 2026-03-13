"""
edge_detector.py — Edge/contour based object detector for VisionTracker.

Replaces YOLO with Canny edge detection + contour finding for a lightweight,
class-agnostic detection approach. All detected objects are assigned class_name="object".

Optional: Background removal using rembg library (LW option) to improve edge
detection by removing the background before processing.

Returns DetectionResult compatible with tracker.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Normalised result from edge detection.

    Attributes
    ----------
    xyxy : np.ndarray, shape (N, 4), dtype float32
        Bounding boxes in [x1, y1, x2, y2] pixel coordinates.
    confidences : np.ndarray, shape (N,), dtype float32
        Detection confidence scores (contour area normalized).
    class_ids : np.ndarray, shape (N,), dtype int32
        All zeros (single class "object").
    class_names : list[str]
        All entries are "object".
    frame_index : int
        Index of the frame processed.
    inference_ms : float
        Wall-clock time (milliseconds) for processing.
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


class EdgeDetector:
    """Edge-based object detector using Canny + contour finding.

    Parameters
    ----------
    min_area : int
        Minimum contour area in pixels to be considered an object.
    max_area : int
        Maximum contour area in pixels (filters out huge regions).
    canny_low : int
        Lower threshold for Canny edge detector.
    canny_high : int
        Upper threshold for Canny edge detector.
    dilate_iterations : int
        Number of dilation iterations to connect edges.
    use_bg_removal : bool
        Whether to use background removal before edge detection.
    skip_frames : int
        Run detection every N frames (returns cached result otherwise).
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 100000,
        canny_low: int = 50,
        canny_high: int = 150,
        dilate_iterations: int = 2,
        use_bg_removal: bool = False,
        skip_frames: int = 1,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.dilate_iterations = dilate_iterations
        self.use_bg_removal = use_bg_removal
        self.skip_frames = max(1, skip_frames)

        self._frame_counter: int = 0
        self._last_result: DetectionResult = DetectionResult()

        # Lazy import for rembg
        self._rembg_session = None
        if self.use_bg_removal:
            try:
                from rembg import new_session
                self._rembg_session = new_session(model_name="u2net")
                print("[EdgeDetector] Background removal enabled (rembg)")
            except ImportError:
                print("[EdgeDetector] Warning: rembg not installed, disabling background removal")
                self.use_bg_removal = False

        print(f"[EdgeDetector] min_area={min_area}, max_area={max_area}, "
              f"canny=({canny_low},{canny_high}), skip={skip_frames}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run edge detection on *frame* (or return cached result on skipped frames).

        Parameters
        ----------
        frame : np.ndarray
            BGR or grayscale uint8 image from OpenCV.

        Returns
        -------
        DetectionResult
            Detection results with all objects having class_name="object".
        """
        import time
        self._frame_counter += 1

        # Return cached result on skipped frames
        if (self._frame_counter % self.skip_frames) != 0:
            return self._last_result

        t0 = time.perf_counter()

        # Optional background removal
        if self.use_bg_removal and self._rembg_session is not None:
            from rembg import remove
            frame = remove(frame, session=self._rembg_session)
            # rembg returns RGBA, convert to BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Convert to grayscale
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Dilate to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=self.dilate_iterations)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and convert to bounding boxes
        boxes = []
        confidences = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x + w, y + h])
                # Confidence based on how "solid" the contour is
                # (area relative to bounding box)
                bbox_area = w * h
                solidity = area / bbox_area if bbox_area > 0 else 0
                confidences.append(min(1.0, solidity + 0.3))

        inference_ms = (time.perf_counter() - t0) * 1000

        if len(boxes) == 0:
            det = DetectionResult(frame_index=self._frame_counter, inference_ms=inference_ms)
        else:
            det = DetectionResult(
                xyxy=np.array(boxes, dtype=np.float32),
                confidences=np.array(confidences, dtype=np.float32),
                class_ids=np.zeros(len(boxes), dtype=np.int32),
                class_names=["object"] * len(boxes),
                frame_index=self._frame_counter,
                inference_ms=inference_ms,
            )

        self._last_result = det
        return det

    @property
    def class_names(self) -> dict[int, str]:
        """Return the {id: name} mapping (only one class: object)."""
        return {0: "object"}

    @property
    def frame_count(self) -> int:
        """Total number of frames processed (including skipped)."""
        return self._frame_counter
