"""
edge_detector.py — Detectors for VisionTracker.

Two detectors are provided, both returning DetectionResult objects that are
fully compatible with tracker.py:

  EdgeDetector  — Classic CV pipeline: Bilateral → Canny → morph-close →
                  proximity-merge.  No extra dependencies, works on any scene.
                  Use with:  python main.py --detector edge

  YOLODetector  — Ultralytics YOLOv8/v11 inference.  Much more accurate
                  bounding boxes with real class names (person, car, dog …)
                  and genuine confidence scores.  Class names flow straight
                  through to the UIOverlay so you see something meaningful
                  instantly, before any LLM identification call.
                  Use with:  python main.py --detector yolo
                  Model:     --yolo-model yolo11n.pt  (default, ~2.6 MB,
                             auto-downloaded on first run)

Both detectors honour skip_frames: detection always runs on frame 1, then
every N frames thereafter; intermediate frames return the cached result.

All classes in this module are importable for unit-testing without a live
camera or GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Normalised result from any detector.

    Attributes
    ----------
    xyxy : np.ndarray, shape (N, 4), dtype float32
        Bounding boxes in [x1, y1, x2, y2] pixel coordinates.
    confidences : np.ndarray, shape (N,), dtype float32
        Detection confidence scores.
    class_ids : np.ndarray, shape (N,), dtype int32
        Integer class indices.
    class_names : list[str]
        Human-readable class label for each detection.
    frame_index : int
        Frame counter at the time this result was produced.
    inference_ms : float
        Wall-clock time (milliseconds) for the detection step.
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
# Edge detector helpers  (kept module-level for testability)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_boxes_by_proximity(
    boxes: list[list[int]],
    expand_px: int = 30,
) -> list[list[int]]:
    """Union bounding boxes that are within *expand_px* pixels of each other.

    Algorithm
    ---------
    1. Expand every box by expand_px on all four sides.
    2. Union-find: two boxes are connected when their expanded rectangles
       overlap, meaning their original edges are within expand_px pixels.
    3. Each connected component collapses to the tight bounding box of its
       original (unexpanded) member boxes.
    """
    if not boxes:
        return []

    n = len(boxes)
    exp = [
        [x1 - expand_px, y1 - expand_px, x2 + expand_px, y2 + expand_px]
        for x1, y1, x2, y2 in boxes
    ]

    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            ex1, ey1, ex2, ey2 = exp[i]
            fx1, fy1, fx2, fy2 = exp[j]
            if ex1 < fx2 and ex2 > fx1 and ey1 < fy2 and ey2 > fy1:
                _union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        x1, y1, x2, y2 = boxes[i]
        if root not in groups:
            groups[root] = [x1, y1, x2, y2]
        else:
            g = groups[root]
            g[0] = min(g[0], x1)
            g[1] = min(g[1], y1)
            g[2] = max(g[2], x2)
            g[3] = max(g[3], y2)

    return list(groups.values())


def _auto_canny_thresholds(
    gray: np.ndarray,
    sigma: float = 0.33,
) -> tuple[int, int]:
    """Compute Canny thresholds from scene intensity percentiles.

    Uses the 10th–90th percentile midpoint as an anchor rather than the raw
    median, so very dark or overexposed scenes never produce a threshold
    collapse to (0, ~10) that would flood detection with noise.
    """
    p_lo   = float(np.percentile(gray, 10))
    p_hi   = float(np.percentile(gray, 90))
    anchor = (p_lo + p_hi) / 2.0
    anchor = max(30.0, min(anchor, 220.0))

    low  = max(0,   int((1.0 - sigma) * anchor))
    high = min(255, int((1.0 + sigma) * anchor))

    min_gap = 20
    if high - low < min_gap:
        half = min_gap // 2
        low  = max(0,   low  - half)
        high = min(255, high + half)

    return low, high


# ─────────────────────────────────────────────────────────────────────────────
# Shared skip-frame mixin
# ─────────────────────────────────────────────────────────────────────────────

class _SkipFrameMixin:
    """Adds skip-frame caching to any detector.

    Subclasses must call ``_sf_init(skip_frames)`` in __init__ and wrap
    their per-frame work with ``_sf_tick(frame)`` which returns either the
    cached result or calls ``_run(frame)`` for a fresh detection.
    """

    def _sf_init(self, skip_frames: int) -> None:
        self._skip_frames: int = max(1, skip_frames)
        self._frame_counter: int = 0
        self._last_result: DetectionResult = DetectionResult()

    def _sf_tick(self, frame: np.ndarray) -> DetectionResult:
        self._frame_counter += 1
        # Always run on frame 1; thereafter every skip_frames frames.
        if self._skip_frames > 1 and (self._frame_counter - 1) % self._skip_frames != 0:
            return self._last_result
        t0 = time.perf_counter()
        result = self._run(frame)  # type: ignore[attr-defined]
        result.inference_ms = (time.perf_counter() - t0) * 1000
        result.frame_index  = self._frame_counter
        self._last_result   = result
        return result

    @property
    def frame_count(self) -> int:
        return self._frame_counter


# ─────────────────────────────────────────────────────────────────────────────
# 1. EdgeDetector  (classic CV pipeline — unchanged behaviour)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeDetector(_SkipFrameMixin):
    """Edge-based object detector: Bilateral + Canny + morph-close + merge.

    Returns class-agnostic detections (class_name="object").  Fast on CPU
    with no extra dependencies beyond OpenCV.

    Parameters
    ----------
    min_area, max_area : int
        Contour area filter in pixels².
    canny_low, canny_high : int
        Manual Canny thresholds (only used when auto_canny=False).
    auto_canny : bool
        Adapt Canny thresholds to scene brightness via percentile anchoring.
    bilateral_d : int
        Bilateral filter diameter (0 = disabled).  5 is ~3× faster than 9.
    close_kernel_size : int
        Morphological-close kernel side length for bridging outline gaps.
    close_iterations, dilate_iterations : int
        Repetitions for morphological steps.
    merge_distance : int
        Proximity radius for bounding-box merging (0 = disabled).
    max_detections : int
        Hard cap on raw contour boxes before the O(n²) merge step.
    min_box_dim : int
        Minimum box width and height after merging.
    max_aspect : float
        Maximum long_side/short_side ratio (filters thin-line noise).
    use_bg_removal : bool
        Run rembg before edge detection (requires rembg package).
    skip_frames : int
        Run detection every N frames; return cached result otherwise.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 100_000,
        canny_low: int = 50,
        canny_high: int = 150,
        auto_canny: bool = True,
        bilateral_d: int = 5,
        close_kernel_size: int = 15,
        close_iterations: int = 2,
        dilate_iterations: int = 1,
        merge_distance: int = 30,
        max_detections: int = 80,
        min_box_dim: int = 20,
        max_aspect: float = 10.0,
        use_bg_removal: bool = False,
        skip_frames: int = 1,
    ) -> None:
        self._sf_init(skip_frames)

        self.min_area          = min_area
        self.max_area          = max_area
        self.canny_low         = canny_low
        self.canny_high        = canny_high
        self.auto_canny        = auto_canny
        self.bilateral_d       = bilateral_d
        self.close_kernel_size = close_kernel_size
        self.close_iterations  = close_iterations
        self.dilate_iterations = dilate_iterations
        self.merge_distance    = merge_distance
        self.max_detections    = max(1, max_detections)
        self.min_box_dim       = min_box_dim
        self.max_aspect        = max_aspect
        self.use_bg_removal    = use_bg_removal

        self._close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
            )
            if close_kernel_size > 0 else None
        )
        self._dilate_kernel = np.ones((5, 5), np.uint8)

        self._rembg_session = None
        if self.use_bg_removal:
            try:
                from rembg import new_session
                self._rembg_session = new_session(model_name="u2net")
                print("[EdgeDetector] Background removal enabled (rembg/u2net)")
            except ImportError:
                print("[EdgeDetector] Warning: rembg not installed — bg removal disabled")
                self.use_bg_removal = False

        print(
            f"[EdgeDetector] area=[{min_area},{max_area}] "
            f"canny={'auto(percentile)' if auto_canny else f'({canny_low},{canny_high})'} "
            f"bilateral_d={bilateral_d} "
            f"close={close_kernel_size}px×{close_iterations} "
            f"merge={merge_distance}px "
            f"max_det={self.max_detections} "
            f"skip={skip_frames}"
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run edge detection on *frame*, returning cached result on skipped frames."""
        return self._sf_tick(frame)

    def _run(self, frame: np.ndarray) -> DetectionResult:
        if self.use_bg_removal and self._rembg_session is not None:
            from rembg import remove
            frame = remove(frame, session=self._rembg_session)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3 else frame.copy()
        )
        processed = (
            cv2.bilateralFilter(gray, self.bilateral_d, 75, 75)
            if self.bilateral_d > 0 else gray
        )
        blurred = cv2.GaussianBlur(processed, (5, 5), 0)

        if self.auto_canny:
            low, high = _auto_canny_thresholds(blurred)
        else:
            low, high = self.canny_low, self.canny_high
        edges = cv2.Canny(blurred, low, high)

        if self._close_kernel is not None and self.close_iterations > 0:
            edges = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE,
                self._close_kernel, iterations=self.close_iterations,
            )
        if self.dilate_iterations > 0:
            edges = cv2.dilate(
                edges, self._dilate_kernel, iterations=self.dilate_iterations
            )

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h_frame, w_frame = frame.shape[:2]
        raw_boxes: list[list[int]] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < self.min_box_dim or h < self.min_box_dim:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect:
                continue
            raw_boxes.append([
                max(0, x), max(0, y),
                min(w_frame, x + w), min(h_frame, y + h),
            ])

        if not raw_boxes:
            return DetectionResult()

        if len(raw_boxes) > self.max_detections:
            raw_boxes.sort(
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
            )
            raw_boxes = raw_boxes[: self.max_detections]

        merged = (
            _merge_boxes_by_proximity(raw_boxes, self.merge_distance)
            if self.merge_distance > 0 else raw_boxes
        )

        frame_area = w_frame * h_frame
        max_post   = min(self.max_area * 4, int(frame_area * 0.90))

        final_boxes: list[list[int]] = []
        confidences: list[float]     = []

        for box in merged:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            if bw * bh < self.min_area or bw * bh > max_post:
                continue
            if bw < self.min_box_dim or bh < self.min_box_dim:
                continue
            final_boxes.append(box)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            conf   = max(0.3, min(1.0, 1.0 - (aspect - 1.0) / self.max_aspect))
            confidences.append(conf)

        if not final_boxes:
            return DetectionResult()

        return DetectionResult(
            xyxy=np.array(final_boxes, dtype=np.float32),
            confidences=np.array(confidences, dtype=np.float32),
            class_ids=np.zeros(len(final_boxes), dtype=np.int32),
            class_names=["object"] * len(final_boxes),
        )

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "object"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. YOLODetector  (Ultralytics YOLOv8 / YOLO11)
# ─────────────────────────────────────────────────────────────────────────────

class YOLODetector(_SkipFrameMixin):
    """YOLO-based detector using Ultralytics YOLOv8 / YOLO11.

    YOLO is used *only* for detection — it produces precise bounding boxes,
    real confidence scores, and human-readable class names (person, car, dog
    …).  Identification (what exactly is this specific object?) is still
    handled by the optional remote LLM via IDService, exactly as before.

    Why this matters
    ----------------
    EdgeDetector outputs class_name="object" for everything, so UIOverlay
    always shows "object" until the LLM responds.  YOLODetector outputs real
    class names immediately (e.g. "person", "laptop"), giving you useful
    labels from frame 1, with the LLM refining them later if enabled.

    Model selection
    ---------------
    YOLO11 nano (default):
      yolo11n.pt  — 2.6 MB, ~30 FPS on laptop CPU, 80 COCO classes
    YOLO11 small:
      yolo11s.pt  — 9.4 MB, ~20 FPS, noticeably more accurate
    YOLOv8 nano (older, widely tested):
      yolov8n.pt  — 6.2 MB, ~25 FPS

    Models are auto-downloaded from Ultralytics on first run.  Pass a local
    path to avoid re-downloading: --yolo-model /path/to/yolo11n.pt

    Parameters
    ----------
    model_path : str
        Ultralytics model name or path (default "yolo11n.pt").
    conf_threshold : float
        Minimum confidence to keep a detection (default 0.35).
    iou_threshold : float
        NMS IoU threshold (default 0.45).
    classes : list[int] | None
        Filter to specific COCO class IDs (None = all 80 classes).
        E.g. [0] = persons only, [0, 2, 7] = persons + cars + trucks.
    device : str
        Inference device: "cpu", "cuda", "mps", or "" (auto).
    imgsz : int
        Inference image size in pixels (longer side).  320 is fastest on
        CPU; 640 is standard quality.  Default 640.
    skip_frames : int
        Run YOLO every N frames; return cached result otherwise.
        Always runs on frame 1 regardless of this value.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Optional[list[int]] = None,
        device: str = "",
        imgsz: int = 640,
        skip_frames: int = 1,
    ) -> None:
        self._sf_init(skip_frames)

        self.model_path      = model_path
        self.conf_threshold  = conf_threshold
        self.iou_threshold   = iou_threshold
        self.classes         = classes
        self.imgsz           = imgsz

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for --detector yolo.\n"
                "Install it with:  pip install ultralytics"
            )

        self._model = YOLO(model_path)

        # Resolve device
        if device:
            self._device = device
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            except ImportError:
                self._device = "cpu"

        # Warm up on a blank frame so the first real frame is not slow
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        self._model.predict(
            dummy, imgsz=32, device=self._device,
            conf=self.conf_threshold, iou=self.iou_threshold,
            classes=self.classes, verbose=False,
        )

        # Cache the model's class name mapping
        self._class_map: dict[int, str] = self._model.names  # {0: "person", …}

        cls_str = (
            f"classes={classes}" if classes else "all 80 COCO classes"
        )
        print(
            f"[YOLODetector] model={model_path} device={self._device} "
            f"conf={conf_threshold} iou={iou_threshold} "
            f"imgsz={imgsz} {cls_str} skip={skip_frames}"
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLO inference on *frame*, returning cached result on skipped frames."""
        return self._sf_tick(frame)

    def _run(self, frame: np.ndarray) -> DetectionResult:
        results = self._model.predict(
            frame,
            imgsz=self.imgsz,
            device=self._device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
            return DetectionResult()

        boxes = results[0].boxes
        xyxy        = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = boxes.conf.cpu().numpy().astype(np.float32)
        class_ids   = boxes.cls.cpu().numpy().astype(np.int32)
        class_names = [self._class_map.get(int(c), "object") for c in class_ids]

        return DetectionResult(
            xyxy=xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names,
        )

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_map


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_detector(
    detector_type: str = "edge",
    **kwargs,
) -> "EdgeDetector | YOLODetector":
    """Instantiate the requested detector, forwarding relevant kwargs.

    Parameters
    ----------
    detector_type : "edge" | "yolo"
    **kwargs : forwarded to the selected detector's __init__.
               Unknown keys for the chosen detector are silently ignored so
               callers can pass the full argparse namespace without filtering.
    """
    if detector_type == "edge":
        _EDGE_KEYS = {
            "min_area", "max_area", "canny_low", "canny_high", "auto_canny",
            "bilateral_d", "close_kernel_size", "close_iterations",
            "dilate_iterations", "merge_distance", "max_detections",
            "min_box_dim", "max_aspect", "use_bg_removal", "skip_frames",
        }
        return EdgeDetector(**{k: v for k, v in kwargs.items() if k in _EDGE_KEYS})

    if detector_type == "yolo":
        _YOLO_KEYS = {
            "model_path", "conf_threshold", "iou_threshold",
            "classes", "device", "imgsz", "skip_frames",
        }
        return YOLODetector(**{k: v for k, v in kwargs.items() if k in _YOLO_KEYS})

    raise ValueError(
        f"Unknown detector type: {detector_type!r}. Choose 'edge' or 'yolo'."
    )
