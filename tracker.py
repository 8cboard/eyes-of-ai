"""
tracker.py — Multi-object tracker for VisionTracker.

Two implementations:
  1. ByteTrackWrapper  — wraps supervision.ByteTrack (recommended).
     ByteTrack is a high-performance tracker from the paper
     "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
     (Zhang et al., 2022).  The ``supervision`` library ships a CPU-friendly
     implementation under the MIT license — no GPU required.

  2. CentroidTracker   — pure-Python fallback using IoU + centroid distance
     matching via the Hungarian algorithm.  Zero extra dependencies beyond
     scipy.  Best for very constrained CPU machines or when supervision is
     unavailable.

Switching trackers:
  Change the ``tracker_type`` argument in ``build_tracker()``:
    build_tracker('bytetrack')   → ByteTrackWrapper
    build_tracker('centroid')    → CentroidTracker

  To plug in a different tracker (e.g., BoT-SORT, OC-SORT):
    1. Implement the Tracker ABC defined below.
    2. Return your class from build_tracker().

Both trackers accept a DetectionResult and return a list[TrackedObject].
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# supervision is MIT-licensed and ships ByteTrack
try:
    import supervision as sv
    _SUPERVISION_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SUPERVISION_AVAILABLE = False
    sv = None  # type: ignore

from edge_detector import DetectionResult


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    """A detection associated with a persistent track ID.

    Attributes
    ----------
    track_id : int
        Unique, monotonically increasing integer assigned by the tracker.
    xyxy : np.ndarray, shape (4,), dtype float32
        Bounding box [x1, y1, x2, y2] in pixels.
    class_id : int
        COCO class index.
    class_name : str
        Human-readable class label.
    confidence : float
        Detection confidence from the underlying detector.
    age : int
        Number of frames this track has been alive.
    """

    track_id: int
    xyxy: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    age: int = 0

    @property
    def centroid(self) -> tuple[float, float]:
        """Return (cx, cy) centroid of the bounding box."""
        x1, y1, x2, y2 = self.xyxy
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return float(max(0, x2 - x1) * max(0, y2 - y1))


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface — implement this to plug in any tracker
# ─────────────────────────────────────────────────────────────────────────────

class Tracker(ABC):
    """Abstract tracker interface.

    Implement ``update(detection_result)`` to return a list of TrackedObjects
    for the current frame.  Trackers are stateful — call update() once per
    frame in order.
    """

    @abstractmethod
    def update(self, det: DetectionResult) -> list[TrackedObject]:
        """Update internal state and return tracked objects for this frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable tracker name for logging."""


# ─────────────────────────────────────────────────────────────────────────────
# 1. ByteTrack (via supervision)
# ─────────────────────────────────────────────────────────────────────────────

class ByteTrackWrapper(Tracker):
    """Wraps supervision.ByteTrack.

    ByteTrack uses two confidence thresholds:
      - high-conf detections → immediate track association
      - low-conf detections  → used to rescue "lost" tracks

    Parameters
    ----------
    track_activation_threshold : float
        Min confidence for a detection to activate a new track.
    lost_track_buffer : int
        Frames to keep a track alive after its last detection.
    minimum_matching_threshold : float
        IoU threshold for the primary association step.
    frame_rate : int
        Expected camera frame rate (used to scale the lost buffer).
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        if not _SUPERVISION_AVAILABLE:
            raise ImportError(
                "supervision is required for ByteTrack: pip install supervision"
            )
        # New supervision (0.25+) uses ByteTrack with different API
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        print("[Tracker] Using ByteTrack (supervision)")

    @property
    def name(self) -> str:
        return "ByteTrack"

    def update(self, det: DetectionResult) -> list[TrackedObject]:
        if det.count == 0:
            # Still need to call tracker.update to age/drop lost tracks
            try:
                self._tracker.update_with_detections(sv.Detections.empty())
            except Exception:
                pass
            return []

        sv_det = sv.Detections(
            xyxy=det.xyxy,
            confidence=det.confidences,
            class_id=det.class_ids,
        )
        try:
            tracked = self._tracker.update_with_detections(sv_det)
        except Exception as e:
            # Graceful degradation: return empty if tracker crashes
            print(f"[Tracker] ByteTrack update error: {e}")
            return []

        objects: list[TrackedObject] = []
        for i in range(len(tracked)):
            if tracked.tracker_id is None:
                continue
            tid = tracked.tracker_id[i]
            if tid is None:
                continue
            # Map tracked index back to detection class_name
            # (supervision may reorder detections)
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            cls_name = det.class_names[i] if i < len(det.class_names) else "unknown"
            objects.append(TrackedObject(
                track_id=int(tid),
                xyxy=tracked.xyxy[i].astype(np.float32),
                class_id=cls_id,
                class_name=cls_name,
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
            ))
        return objects


# ─────────────────────────────────────────────────────────────────────────────
# 2. Centroid Tracker (pure-Python fallback)
# ─────────────────────────────────────────────────────────────────────────────

class CentroidTracker(Tracker):
    """Lightweight pure-Python centroid tracker using IoU + centroid distance.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) for
    optimal assignment.  No neural network — works on any hardware.

    Parameters
    ----------
    max_disappeared : int
        Frames a track can go unmatched before being deregistered.
    iou_weight : float
        Weight for IoU cost vs centroid-distance cost (0–1).
    max_distance : float
        Maximum pixel distance between centroids to allow a match.
    """

    def __init__(
        self,
        max_disappeared: int = 20,
        iou_weight: float = 0.7,
        max_distance: float = 150.0,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.iou_weight = iou_weight
        self.max_distance = max_distance

        self._next_id: int = 1
        # {track_id: TrackedObject}
        self._tracks: OrderedDict[int, TrackedObject] = OrderedDict()
        # {track_id: consecutive missed frames}
        self._disappeared: dict[int, int] = {}

        print("[Tracker] Using CentroidTracker (CPU fallback)")

    @property
    def name(self) -> str:
        return "CentroidTracker"

    def update(self, det: DetectionResult) -> list[TrackedObject]:
        if det.count == 0:
            # Mark all existing tracks as disappeared
            for tid in list(self._tracks.keys()):
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
            return list(self._tracks.values())

        if not self._tracks:
            # First frame — register all detections as new tracks
            for i in range(det.count):
                self._register(det, i)
            return list(self._tracks.values())

        # ── Build cost matrix ────────────────────────────────────────────
        track_ids = list(self._tracks.keys())
        track_boxes = np.array([self._tracks[t].xyxy for t in track_ids])
        det_boxes = det.xyxy

        cost = self._build_cost_matrix(track_boxes, det_boxes)

        # ── Hungarian assignment ─────────────────────────────────────────
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] > 1.0:
                # Cost too high — treat as no match
                continue
            tid = track_ids[r]
            self._update_track(tid, det, c)
            matched_tracks.add(tid)
            matched_dets.add(c)

        # Mark unmatched tracks as disappeared
        for r, tid in enumerate(track_ids):
            if tid not in matched_tracks:
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        # Register unmatched detections as new tracks
        for c in range(det.count):
            if c not in matched_dets:
                self._register(det, c)

        return list(self._tracks.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _register(self, det: DetectionResult, idx: int) -> None:
        tid = self._next_id
        self._next_id += 1
        name = det.class_names[idx] if idx < len(det.class_names) else "unknown"
        self._tracks[tid] = TrackedObject(
            track_id=tid,
            xyxy=det.xyxy[idx].copy(),
            class_id=int(det.class_ids[idx]),
            class_name=name,
            confidence=float(det.confidences[idx]),
            age=1,
        )
        self._disappeared[tid] = 0

    def _deregister(self, track_id: int) -> None:
        del self._tracks[track_id]
        del self._disappeared[track_id]

    def _update_track(self, track_id: int, det: DetectionResult, det_idx: int) -> None:
        t = self._tracks[track_id]
        name = det.class_names[det_idx] if det_idx < len(det.class_names) else t.class_name
        self._tracks[track_id] = TrackedObject(
            track_id=track_id,
            xyxy=det.xyxy[det_idx].copy(),
            class_id=int(det.class_ids[det_idx]),
            class_name=name,
            confidence=float(det.confidences[det_idx]),
            age=t.age + 1,
        )
        self._disappeared[track_id] = 0

    def _build_cost_matrix(
        self, track_boxes: np.ndarray, det_boxes: np.ndarray
    ) -> np.ndarray:
        """Combined IoU + centroid-distance cost matrix, range [0, 2]."""
        n_tracks = len(track_boxes)
        n_dets = len(det_boxes)
        cost = np.full((n_tracks, n_dets), 2.0, dtype=np.float32)

        iou_mat = _batch_iou(track_boxes, det_boxes)          # [n_tracks, n_dets]
        dist_mat = _batch_centroid_distance(track_boxes, det_boxes)  # [n_tracks, n_dets]
        # Normalise distance cost to [0, 1] using max_distance
        dist_mat_norm = np.clip(dist_mat / self.max_distance, 0.0, 1.0)

        # High IoU → low cost; large distance → high cost
        cost = (1.0 - iou_mat) * self.iou_weight + dist_mat_norm * (1.0 - self.iou_weight)
        return cost


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two arrays of boxes [N,4] and [M,4]."""
    # Expand dims for broadcasting: [N, 1, 4] vs [1, M, 4]
    a = boxes_a[:, np.newaxis, :]  # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union_area = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def _batch_centroid_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean centroid distance between two box arrays."""
    cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) / 2  # (N,)
    cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) / 2
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2  # (M,)
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2

    dx = cx_a[:, np.newaxis] - cx_b[np.newaxis, :]  # (N, M)
    dy = cy_a[:, np.newaxis] - cy_b[np.newaxis, :]
    return np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_tracker(
    tracker_type: Literal["bytetrack", "centroid"] = "bytetrack",
    **kwargs,
) -> Tracker:
    """Return a Tracker instance by name.

    Parameters
    ----------
    tracker_type : str
        'bytetrack' (default) or 'centroid'.
    **kwargs
        Passed to the tracker constructor.

    To plug in a custom tracker:
        1. Subclass ``Tracker``.
        2. Add a new case here.
    """
    if tracker_type == "bytetrack":
        if _SUPERVISION_AVAILABLE:
            return ByteTrackWrapper(**kwargs)
        else:
            print("[Tracker] supervision not available — falling back to CentroidTracker")
            return CentroidTracker(**kwargs)
    elif tracker_type == "centroid":
        return CentroidTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type!r}. Choose 'bytetrack' or 'centroid'.")
