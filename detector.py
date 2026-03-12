"""
detector.py — Object detector wrapper for VisionTracker.

Wraps Ultralytics YOLOv8 (nano/small/medium/large/x variants).
Key features:
  - Frame-skip: only runs inference every N frames to preserve FPS.
  - Device auto-detection: CUDA > MPS > CPU.
  - Grayscale-friendly: converts input to 3-channel if needed.
  - Returns a consistent DetectionResult dataclass on every call
    (using the previous result on skipped frames).
  - Configurable confidence and IoU thresholds.

No paid API keys required. YOLOv8 weights are downloaded automatically
from Ultralytics' public CDN on first use (~6 MB for 'n' variant).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Lazy import — allows the module to be imported even if ultralytics is not
# yet installed (e.g., during bare test collection).
try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ULTRALYTICS_AVAILABLE = False
    YOLO = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Normalised result from one detector call (or repeated on skipped frames).

    Attributes
    ----------
    xyxy : np.ndarray, shape (N, 4), dtype float32
        Bounding boxes in [x1, y1, x2, y2] pixel coordinates.
    confidences : np.ndarray, shape (N,), dtype float32
        Detection confidence scores in [0, 1].
    class_ids : np.ndarray, shape (N,), dtype int32
        Integer class indices (COCO 80-class by default).
    class_names : list[str]
        Human-readable class names corresponding to class_ids.
    frame_index : int
        Index of the frame on which inference actually ran.
    inference_ms : float
        Wall-clock time (milliseconds) for the YOLO forward pass.
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
# Detector class
# ─────────────────────────────────────────────────────────────────────────────

class Detector:
    """YOLOv8 detector with frame-skip support.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier.  Examples:
          'yolov8n.pt'  — nano,  ~6 MB,  fastest on CPU  (~12–18 FPS main loop)
          'yolov8s.pt'  — small, ~22 MB, good CPU/GPU balance
          'yolov8m.pt'  — medium, ~52 MB, prefers GPU
    skip_frames : int
        Run inference every ``skip_frames`` calls.  skip_frames=1 means every
        frame; skip_frames=2 means every other frame (doubles apparent FPS).
        The last DetectionResult is returned unchanged on skipped frames.
    conf_threshold : float
        Minimum confidence to include a detection.
    iou_threshold : float
        NMS IoU threshold.
    device : str | None
        'cpu', 'cuda', 'mps', or None for auto-detect.
    imgsz : int
        Inference image size (square side, px). Smaller = faster.
        Recommended: 640 (default), 320 (very fast CPU).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        skip_frames: int = 2,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        imgsz: int = 640,
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required: pip install ultralytics")

        self.skip_frames = max(1, skip_frames)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._frame_counter: int = 0
        self._last_result: DetectionResult = DetectionResult()

        # ── Device selection ──────────────────────────────────────────────
        self.device = device or _auto_select_device()
        print(f"[Detector] Loading {model_name} on device='{self.device}' "
              f"(skip_frames={skip_frames}, imgsz={imgsz})")

        # Weights are auto-downloaded on first call; no API key needed.
        self._model = YOLO(model_name)

        # Extract class name map for this model
        self._names: dict[int, str] = self._model.names  # {0: 'person', ...}

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on *frame* (or return cached result on skipped frames).

        Parameters
        ----------
        frame : np.ndarray
            BGR or grayscale uint8 image from OpenCV.

        Returns
        -------
        DetectionResult
            Always non-None.  On skipped frames, ``frame_index`` of the cached
            result will differ from the current frame counter.
        """
        self._frame_counter += 1

        # Return cached result on skipped frames
        if (self._frame_counter % self.skip_frames) != 0:
            return self._last_result

        # Ensure 3-channel input (handles grayscale)
        img = _ensure_bgr(frame)

        t0 = time.perf_counter()
        results = self._model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,  # suppress per-frame console spam
        )
        inference_ms = (time.perf_counter() - t0) * 1_000

        # Parse Ultralytics result object
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            det = DetectionResult(frame_index=self._frame_counter, inference_ms=inference_ms)
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy().astype(np.float32)
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
            names = [self._names.get(int(c), "unknown") for c in cls_ids]
            det = DetectionResult(
                xyxy=xyxy,
                confidences=confs,
                class_ids=cls_ids,
                class_names=names,
                frame_index=self._frame_counter,
                inference_ms=inference_ms,
            )

        self._last_result = det
        return det

    @property
    def class_names(self) -> dict[int, str]:
        """Return the {id: name} mapping for all classes in this model."""
        return self._names

    @property
    def frame_count(self) -> int:
        """Total number of frames processed (including skipped)."""
        return self._frame_counter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_select_device() -> str:
    """Return the best available device string for Ultralytics/PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            print("[Detector] CUDA detected — using GPU.")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Detector] MPS (Apple Silicon) detected — using MPS.")
            return "mps"
    except ImportError:
        pass
    print("[Detector] No GPU detected — using CPU.")
    return "cpu"


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert grayscale → BGR if necessary; leave BGR/RGB untouched."""
    if frame.ndim == 2:
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame
