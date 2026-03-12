"""
stability.py — Stillness detection for VisionTracker.

A track is considered "still" when its centroid velocity and bounding-box
IoU variance are both below configurable thresholds for M consecutive frames.

Two implementations are provided:
  1. IoUVelocityStillnessDetector  — default, lightweight, works on any frame
  2. OpticalFlowStillnessDetector  — uses Farneback dense optical flow to
     detect camera/background motion, good for handheld or moving cameras.

Both expose the same API: call update(track_id, bbox) then is_still(track_id).

Tunable parameters (all exposed as constructor args):
  history_len (N) — sliding window of centroid/IoU samples
  still_frames (M) — consecutive "below threshold" frames required
  velocity_thresh  — max centroid pixel-per-frame movement
  iou_thresh       — min IoU between consecutive boxes to count as stable
"""

from __future__ import annotations

import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StillnessState:
    """Internal state maintained per track."""
    centroids: deque = field(default_factory=lambda: deque(maxlen=30))
    boxes: deque = field(default_factory=lambda: deque(maxlen=30))
    consecutive_still: int = 0
    is_stable: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. IoU + Velocity (default)
# ─────────────────────────────────────────────────────────────────────────────

class IoUVelocityStillnessDetector:
    """Per-track stillness detector using centroid velocity + IoU stability.

    Algorithm (per update call):
      1. Append new centroid and bbox to the track's history deque.
      2. If history_len samples have been collected:
         a. Compute mean centroid velocity (pixel/frame) over the window.
         b. Compute mean pairwise consecutive IoU over the window.
         c. If velocity < velocity_thresh AND mean_iou > iou_thresh:
              increment consecutive_still counter
            else:
              reset consecutive_still to 0
      3. A track reports is_still=True when consecutive_still >= still_frames.

    Parameters
    ----------
    history_len : int
        N — sliding window length for velocity/IoU computation.
    still_frames : int
        M — required consecutive "below threshold" frames.
    velocity_thresh : float
        Maximum centroid movement in pixels/frame to count as still.
        Typical values: 3–8 px for 720p at 30 FPS.
    iou_thresh : float
        Minimum consecutive-IoU to count as geometrically stable.
        Values near 1.0 mean very little box size change.
    """

    def __init__(
        self,
        history_len: int = 15,
        still_frames: int = 10,
        velocity_thresh: float = 5.0,
        iou_thresh: float = 0.85,
    ) -> None:
        self.history_len = history_len
        self.still_frames = still_frames
        self.velocity_thresh = velocity_thresh
        self.iou_thresh = iou_thresh
        # {track_id: StillnessState}
        self._states: dict[int, StillnessState] = defaultdict(
            lambda: StillnessState(
                centroids=deque(maxlen=history_len),
                boxes=deque(maxlen=history_len),
            )
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, track_id: int, bbox: np.ndarray) -> None:
        """Record a new observation for *track_id*.

        Parameters
        ----------
        track_id : int
        bbox : np.ndarray, shape (4,)
            [x1, y1, x2, y2] in pixels.
        """
        state = self._states[track_id]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        state.centroids.append((cx, cy))
        state.boxes.append(bbox.copy())

        if len(state.centroids) < 2:
            return  # Not enough data yet

        # ── Centroid velocity ─────────────────────────────────────────────
        cx_arr = np.array([c[0] for c in state.centroids])
        cy_arr = np.array([c[1] for c in state.centroids])
        dx = np.diff(cx_arr)
        dy = np.diff(cy_arr)
        velocities = np.sqrt(dx ** 2 + dy ** 2)
        mean_velocity = float(np.mean(velocities))

        # ── Consecutive IoU ───────────────────────────────────────────────
        box_arr = np.array(state.boxes)  # (T, 4)
        iou_vals = [
            _box_iou(box_arr[i], box_arr[i + 1])
            for i in range(len(box_arr) - 1)
        ]
        mean_iou = float(np.mean(iou_vals)) if iou_vals else 0.0

        # ── Decide stillness ──────────────────────────────────────────────
        currently_still = (
            mean_velocity < self.velocity_thresh and mean_iou > self.iou_thresh
        )
        if currently_still:
            state.consecutive_still += 1
        else:
            state.consecutive_still = 0

        state.is_stable = state.consecutive_still >= self.still_frames

    def is_still(self, track_id: int) -> bool:
        """Return True if the track is currently considered still."""
        return self._states[track_id].is_stable

    def reset(self, track_id: int) -> None:
        """Clear history for a track (call after identification completes)."""
        if track_id in self._states:
            s = self._states[track_id]
            s.consecutive_still = 0
            s.is_stable = False

    def remove_track(self, track_id: int) -> None:
        """Remove all state for a track (call when tracker drops the track)."""
        self._states.pop(track_id, None)

    def get_velocity(self, track_id: int) -> float:
        """Return the last computed mean velocity for debugging / UI."""
        state = self._states.get(track_id)
        if state is None or len(state.centroids) < 2:
            return 0.0
        cx_arr = np.array([c[0] for c in state.centroids])
        cy_arr = np.array([c[1] for c in state.centroids])
        dx = np.diff(cx_arr)
        dy = np.diff(cy_arr)
        return float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Optical-Flow Stillness (for moving cameras)
# ─────────────────────────────────────────────────────────────────────────────

class OpticalFlowStillnessDetector:
    """Stillness detector that accounts for global camera motion via dense
    optical flow (Gunnar Farneback algorithm, available in OpenCV on CPU).

    Useful when the camera itself is panning/tilting — a naive centroid
    velocity check would flag all objects as moving even if they are still
    relative to the scene.

    Algorithm:
      1. Compute dense optical flow between the previous and current frame.
      2. Estimate the global/background flow field (median of flow in the
         non-box regions of the image).
      3. Subtract background flow from each tracked object's region.
      4. The residual motion magnitude determines per-track stillness.

    Parameters
    ----------
    history_len : int
        N frames of residual-motion history.
    still_frames : int
        M consecutive still frames required.
    residual_thresh : float
        Maximum mean residual optical flow magnitude (px/frame) per object.
    """

    def __init__(
        self,
        history_len: int = 10,
        still_frames: int = 8,
        residual_thresh: float = 2.0,
    ) -> None:
        self.history_len = history_len
        self.still_frames = still_frames
        self.residual_thresh = residual_thresh

        self._prev_gray: Optional[np.ndarray] = None
        self._flow: Optional[np.ndarray] = None  # Last computed flow field

        # {track_id: deque of residual magnitudes}
        self._residuals: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        self._consecutive: dict[int, int] = defaultdict(int)
        self._stable: dict[int, bool] = defaultdict(bool)

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_frame(self, gray_frame: np.ndarray) -> None:
        """Compute optical flow from the previous to the current gray frame.

        Call this once per frame *before* calling update() for any tracks.

        Parameters
        ----------
        gray_frame : np.ndarray
            Single-channel (grayscale) uint8 frame.
        """
        import cv2
        if self._prev_gray is not None:
            self._flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray_frame,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        self._prev_gray = gray_frame.copy()

    def update(self, track_id: int, bbox: np.ndarray) -> None:
        """Update stillness state for *track_id* using the latest flow field.

        Parameters
        ----------
        track_id : int
        bbox : np.ndarray, shape (4,)  — [x1, y1, x2, y2] in pixels.
        """
        if self._flow is None:
            return  # No flow computed yet

        x1, y1, x2, y2 = bbox.astype(int)
        h, w = self._flow.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return

        # Flow in object region
        obj_flow = self._flow[y1:y2, x1:x2]  # (H, W, 2)

        # Background flow estimate — median of the entire frame flow
        bg_fx = float(np.median(self._flow[..., 0]))
        bg_fy = float(np.median(self._flow[..., 1]))

        # Residual: subtract global camera motion
        residual = obj_flow - np.array([bg_fx, bg_fy])
        mag = float(np.mean(np.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2)))

        self._residuals[track_id].append(mag)
        mean_mag = float(np.mean(self._residuals[track_id]))

        if mean_mag < self.residual_thresh:
            self._consecutive[track_id] += 1
        else:
            self._consecutive[track_id] = 0

        self._stable[track_id] = self._consecutive[track_id] >= self.still_frames

    def is_still(self, track_id: int) -> bool:
        return self._stable.get(track_id, False)

    def reset(self, track_id: int) -> None:
        self._consecutive[track_id] = 0
        self._stable[track_id] = False

    def remove_track(self, track_id: int) -> None:
        self._residuals.pop(track_id, None)
        self._consecutive.pop(track_id, None)
        self._stable.pop(track_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
