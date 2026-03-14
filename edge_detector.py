"""
edge_detector.py — Edge/contour based object detector for VisionTracker.

Key improvements over v1:
  1. Adaptive Canny thresholds  — auto-calibrates to scene brightness so the
     detector works in both dark and bright environments without hand-tuning.
  2. Bilateral pre-filter       — smooths away texture noise while keeping
     crisp object-boundary edges (unlike Gaussian which blurs everything).
  3. Morphological closing      — bridges the gaps inside object outlines so
     findContours sees whole shapes instead of edge-fragments.
  4. Proximity-based box merge  — contour fragments that fall within
     `merge_distance` pixels of each other are unioned into one bounding box.
     This is the primary fix for "hundreds of tiny boxes per scene".
  5. Aspect-ratio + min-dim     — eliminates thin wire-like noise contours.

All detected objects are assigned class_name="object" (class-agnostic).
Returns DetectionResult compatible with tracker.py.
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
    """Normalised result from edge detection.

    Attributes
    ----------
    xyxy : np.ndarray, shape (N, 4), dtype float32
        Bounding boxes in [x1, y1, x2, y2] pixel coordinates.
    confidences : np.ndarray, shape (N,), dtype float32
        Detection confidence scores (aspect-ratio-based heuristic).
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


# ─────────────────────────────────────────────────────────────────────────────
# Proximity-based bounding-box merge  (module-level for testability)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_boxes_by_proximity(
    boxes: list[list[int]],
    expand_px: int = 30,
) -> list[list[int]]:
    """Union bounding boxes that are within *expand_px* pixels of each other.

    Algorithm:
      1. Expand every box by expand_px on all four sides.
      2. Build a union-find graph: two boxes are connected when their expanded
         versions overlap (meaning they are within expand_px px of each other).
      3. Each connected component collapses to the tight bounding box of its
         original (unexpanded) member boxes.

    Parameters
    ----------
    boxes : list of [x1, y1, x2, y2]
    expand_px : int
        Proximity radius in pixels.

    Returns
    -------
    list of [x1, y1, x2, y2] — merged bounding boxes.
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


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Canny helper  (module-level for testability)
# ─────────────────────────────────────────────────────────────────────────────

def _auto_canny_thresholds(
    gray: np.ndarray,
    sigma: float = 0.33,
) -> tuple[int, int]:
    """Compute Canny thresholds from scene intensity percentiles.

    FIX: uses the 10th-90th percentile span instead of the raw median.
    The old raw-median approach collapsed to (0, 0) on near-black frames
    (median=0 → low=-10→0, high=10) and flooded detection because almost
    every pixel crossed that tiny threshold.  Anchoring on the inner-80%
    intensity range gives a stable signal in all lighting conditions.

    Parameters
    ----------
    gray   : uint8 grayscale image
    sigma  : spread factor around the anchor value

    Returns
    -------
    (low, high) integer thresholds for cv2.Canny.
    """
    p_lo   = float(np.percentile(gray, 10))
    p_hi   = float(np.percentile(gray, 90))
    anchor = (p_lo + p_hi) / 2.0

    # Clamp anchor to a range that always produces meaningful edges.
    anchor = max(30.0, min(anchor, 220.0))

    low  = max(0,   int((1.0 - sigma) * anchor))
    high = min(255, int((1.0 + sigma) * anchor))

    # Guarantee a minimum gap so Canny is neither over- nor under-selective.
    min_gap = 20
    if high - low < min_gap:
        half = min_gap // 2
        low  = max(0,   low  - half)
        high = min(255, high + half)

    return low, high


# ─────────────────────────────────────────────────────────────────────────────
# Edge Detector
# ─────────────────────────────────────────────────────────────────────────────

class EdgeDetector:
    """Edge-based object detector: Bilateral + Canny + morph close + merge.

    Parameters
    ----------
    min_area : int
        Minimum contour area (pixels²) to be considered an object.
    max_area : int
        Maximum contour area (filters huge background regions).
    canny_low, canny_high : int
        Manual Canny thresholds — used only when ``auto_canny=False``.
    auto_canny : bool
        Auto-adapt Canny thresholds to the scene's brightness.
        Strongly recommended for real-world scenes.  Default: True.
    bilateral_d : int
        Diameter of the bilateral filter neighbourhood.  Larger = more
        texture smoothing but slower.  Default is 5 (CPU-friendly).
        Going from 9 to 5 is roughly 3x faster at the cost of slightly
        less texture suppression.  Set to 0 to disable.
    close_kernel_size : int
        Side length of the elliptical morphological-close kernel that
        bridges gaps in object outlines.  Larger = more gap bridging.
    close_iterations : int
        How many times to apply morphological closing.
    dilate_iterations : int
        Extra dilation passes after closing.
    merge_distance : int
        Pixel proximity threshold for bounding-box merging.  Any two boxes
        whose borders are closer than this value are merged into one.
        Set to 0 to disable merging entirely.
    max_detections : int
        Hard cap on pre-merge raw contour boxes (default 80).  Prevents
        O(n²) slowdowns in heavily cluttered scenes; the largest boxes
        (most likely to be real objects) are kept.
    min_box_dim : int
        Minimum width *and* height for a box after merging.
    max_aspect : float
        Maximum aspect ratio (long_side / short_side).  Filters thin lines.
    use_bg_removal : bool
        Use rembg background removal before edge detection.
    skip_frames : int
        Run detection every N frames; return cached result otherwise.
        Detection always runs on frame 1 regardless of this value.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 100_000,
        canny_low: int = 50,
        canny_high: int = 150,
        auto_canny: bool = True,
        bilateral_d: int = 5,           # was 9 — ~3x faster on CPU
        close_kernel_size: int = 15,
        close_iterations: int = 2,
        dilate_iterations: int = 1,
        merge_distance: int = 30,
        max_detections: int = 80,       # cap before O(n²) merge
        min_box_dim: int = 20,
        max_aspect: float = 10.0,
        use_bg_removal: bool = False,
        skip_frames: int = 1,
    ) -> None:
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
        self.skip_frames       = max(1, skip_frames)

        self._frame_counter: int = 0
        self._last_result: DetectionResult = DetectionResult()

        # Pre-build kernels
        self._close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
            )
            if close_kernel_size > 0 else None
        )
        self._dilate_kernel = np.ones((5, 5), np.uint8)

        # Optional rembg session
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

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run edge detection on *frame* (cached on skip frames).

        Parameters
        ----------
        frame : np.ndarray
            BGR or grayscale uint8 image from OpenCV.

        Returns
        -------
        DetectionResult
        """
        import time
        self._frame_counter += 1

        # FIX: use (counter-1) so detection runs on frame 1, 1+N, 1+2N ...
        # The old `counter % N != 0` skipped frame 1 entirely when N > 1,
        # giving an empty result on the very first displayed frame.
        if self.skip_frames > 1 and (self._frame_counter - 1) % self.skip_frames != 0:
            return self._last_result

        t0 = time.perf_counter()
        result = self._run(frame)
        result.inference_ms = (time.perf_counter() - t0) * 1000
        result.frame_index  = self._frame_counter
        self._last_result   = result
        return result

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _run(self, frame: np.ndarray) -> DetectionResult:
        # 1. Optional background removal
        if self.use_bg_removal and self._rembg_session is not None:
            from rembg import remove
            frame = remove(frame, session=self._rembg_session)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # 2. Convert to grayscale
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3 else frame.copy()
        )

        # 3. Bilateral filter — smooths textures, keeps object edges sharp
        processed = (
            cv2.bilateralFilter(gray, self.bilateral_d, 75, 75)
            if self.bilateral_d > 0 else gray
        )

        # 4. Gaussian blur removes any remaining high-frequency noise
        blurred = cv2.GaussianBlur(processed, (5, 5), 0)

        # 5. Canny edge detection
        if self.auto_canny:
            low, high = _auto_canny_thresholds(blurred)
        else:
            low, high = self.canny_low, self.canny_high
        edges = cv2.Canny(blurred, low, high)

        # 6. Morphological CLOSE (dilate then erode) to bridge gaps inside
        #    object outlines — the key step that produces whole-object shapes
        if self._close_kernel is not None and self.close_iterations > 0:
            edges = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE,
                self._close_kernel, iterations=self.close_iterations,
            )

        # 7. Optional extra dilation to thicken remaining edges
        if self.dilate_iterations > 0:
            edges = cv2.dilate(
                edges, self._dilate_kernel, iterations=self.dilate_iterations
            )

        # 8. Find external contours (one per connected edge region)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 9. Per-contour filtering
        h_frame, w_frame = frame.shape[:2]
        raw_boxes: list[list[int]] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w < self.min_box_dim or h < self.min_box_dim:
                continue

            # Aspect ratio guard: skip needle-like noise contours
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect:
                continue

            # Clamp to frame bounds
            raw_boxes.append([
                max(0, x),
                max(0, y),
                min(w_frame, x + w),
                min(h_frame, y + h),
            ])

        if not raw_boxes:
            return DetectionResult()

        # 10. Cap raw boxes before the O(n²) merge to protect CPU budgets.
        #     Sort descending by area so the most prominent objects survive.
        if len(raw_boxes) > self.max_detections:
            raw_boxes.sort(
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
            )
            raw_boxes = raw_boxes[: self.max_detections]

        # 11. Proximity merge — unites fragments of the same object
        merged = (
            _merge_boxes_by_proximity(raw_boxes, self.merge_distance)
            if self.merge_distance > 0 else raw_boxes
        )

        # 12. Post-merge quality filter
        frame_area = w_frame * h_frame
        max_post   = min(self.max_area * 4, int(frame_area * 0.90))

        final_boxes: list[list[int]] = []
        confidences: list[float]     = []

        for box in merged:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            bbox_area = bw * bh
            if bbox_area < self.min_area or bbox_area > max_post:
                continue
            if bw < self.min_box_dim or bh < self.min_box_dim:
                continue
            final_boxes.append(box)
            # Confidence: penalise extreme aspect ratios
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

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "object"}

    @property
    def frame_count(self) -> int:
        return self._frame_counter
