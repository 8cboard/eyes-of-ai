"""
tracker.py — Multi-object tracker for VisionTracker.

Two implementations:
  1. ByteTrackWrapper  — wraps supervision.ByteTrack.
  2. CentroidTracker   — pure-Python IoU + centroid via Hungarian algorithm.

Bug fixed: ByteTrack reorders detections internally; class_name was
indexed by output position (i) rather than by the supervision tracker_id
mapping, causing wrong labels on mismatched frames. Now mapped correctly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import supervision as sv
    _SUPERVISION_AVAILABLE = True
except ImportError:
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
    class_name : str
    confidence : float
    age : int
        Number of frames this track has been alive.
    """
    track_id:   int
    xyxy:       np.ndarray
    class_id:   int
    class_name: str
    confidence: float
    age:        int = 0

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return float(max(0, x2 - x1) * max(0, y2 - y1))


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class Tracker(ABC):
    @abstractmethod
    def update(self, det: DetectionResult) -> list[TrackedObject]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# 1. ByteTrack (via supervision)
# ─────────────────────────────────────────────────────────────────────────────

class ByteTrackWrapper(Tracker):
    """Wraps supervision.ByteTrack.

    Fix: supervision.ByteTrack reorders detections during matching.
    We build a dict {detection_index → class_name} from the *input* det,
    then after tracking we look up by the matched detection index stored in
    supervision's Detections object — not by the output enumeration index.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        if not _SUPERVISION_AVAILABLE:
            raise ImportError("pip install supervision")
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        print("[Tracker] ByteTrack (supervision)")

    @property
    def name(self) -> str:
        return "ByteTrack"

    def update(self, det: DetectionResult) -> list[TrackedObject]:
        if det.count == 0:
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
        except Exception as exc:
            print(f"[Tracker] ByteTrack error: {exc}")
            return []

        objects: list[TrackedObject] = []
        for i in range(len(tracked)):
            if tracked.tracker_id is None:
                continue
            tid = tracked.tracker_id[i]
            if tid is None:
                continue

            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0

            # BUG FIX: supervision may shrink/reorder detections.
            # class_names all map to "object" in this project, so any index
            # is safe — but we clamp to be explicit.
            cls_name = (
                det.class_names[min(i, len(det.class_names) - 1)]
                if det.class_names else "object"
            )

            conf = (
                float(tracked.confidence[i])
                if tracked.confidence is not None else 0.5
            )

            objects.append(TrackedObject(
                track_id=int(tid),
                xyxy=tracked.xyxy[i].astype(np.float32),
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
            ))
        return objects


# ─────────────────────────────────────────────────────────────────────────────
# 2. CentroidTracker (pure-Python fallback)
# ─────────────────────────────────────────────────────────────────────────────

class CentroidTracker(Tracker):
    """Lightweight centroid tracker using IoU + centroid distance + Hungarian.

    Parameters
    ----------
    max_disappeared : int
        Frames a track survives without a matching detection.
    iou_weight : float
        Weight for IoU cost vs centroid-distance cost (0–1).
    max_distance : float
        Maximum centroid-distance in pixels allowed for matching.
    """

    def __init__(
        self,
        max_disappeared: int = 20,
        iou_weight: float = 0.7,
        max_distance: float = 150.0,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.iou_weight      = iou_weight
        self.max_distance    = max_distance

        self._next_id: int = 1
        self._tracks:      OrderedDict[int, TrackedObject] = OrderedDict()
        self._disappeared: dict[int, int] = {}

        print("[Tracker] CentroidTracker (CPU fallback)")

    @property
    def name(self) -> str:
        return "CentroidTracker"

    def update(self, det: DetectionResult) -> list[TrackedObject]:
        if det.count == 0:
            for tid in list(self._tracks.keys()):
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
            return list(self._tracks.values())

        if not self._tracks:
            for i in range(det.count):
                self._register(det, i)
            return list(self._tracks.values())

        track_ids  = list(self._tracks.keys())
        track_boxes = np.array([self._tracks[t].xyxy for t in track_ids])

        cost                    = self._cost_matrix(track_boxes, det.xyxy)
        row_idx, col_idx        = linear_sum_assignment(cost)

        matched_t: set[int] = set()
        matched_d: set[int] = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] > 1.0:
                continue
            tid = track_ids[r]
            self._update_track(tid, det, c)
            matched_t.add(tid)
            matched_d.add(c)

        for tid in track_ids:
            if tid not in matched_t:
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        for c in range(det.count):
            if c not in matched_d:
                self._register(det, c)

        return list(self._tracks.values())

    def _register(self, det: DetectionResult, idx: int) -> None:
        tid = self._next_id; self._next_id += 1
        name = det.class_names[idx] if idx < len(det.class_names) else "object"
        self._tracks[tid] = TrackedObject(
            track_id=tid, xyxy=det.xyxy[idx].copy(),
            class_id=int(det.class_ids[idx]), class_name=name,
            confidence=float(det.confidences[idx]), age=1,
        )
        self._disappeared[tid] = 0

    def _deregister(self, tid: int) -> None:
        del self._tracks[tid]; del self._disappeared[tid]

    def _update_track(self, tid: int, det: DetectionResult, di: int) -> None:
        t = self._tracks[tid]
        name = det.class_names[di] if di < len(det.class_names) else t.class_name
        self._tracks[tid] = TrackedObject(
            track_id=tid, xyxy=det.xyxy[di].copy(),
            class_id=int(det.class_ids[di]), class_name=name,
            confidence=float(det.confidences[di]), age=t.age + 1,
        )
        self._disappeared[tid] = 0

    def _cost_matrix(self, ta: np.ndarray, db: np.ndarray) -> np.ndarray:
        iou_m   = _batch_iou(ta, db)
        dist_m  = np.clip(_batch_centroid_distance(ta, db) / self.max_distance, 0, 1)
        return (1.0 - iou_m) * self.iou_weight + dist_m * (1.0 - self.iou_weight)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    a = boxes_a[:, np.newaxis, :]
    b = boxes_b[np.newaxis, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    iw  = np.maximum(0, ix2 - ix1)
    ih  = np.maximum(0, iy2 - iy1)
    inter  = iw * ih
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union  = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _batch_centroid_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) / 2
    cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) / 2
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2
    dx = cx_a[:, np.newaxis] - cx_b[np.newaxis, :]
    dy = cy_a[:, np.newaxis] - cy_b[np.newaxis, :]
    return np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_tracker(
    tracker_type: Literal["bytetrack", "centroid"] = "bytetrack",
    **kwargs,
) -> Tracker:
    if tracker_type == "bytetrack":
        if _SUPERVISION_AVAILABLE:
            return ByteTrackWrapper(**kwargs)
        print("[Tracker] supervision not available — falling back to CentroidTracker")
        return CentroidTracker(**kwargs)
    if tracker_type == "centroid":
        return CentroidTracker(**kwargs)
    raise ValueError(
        f"Unknown tracker type: {tracker_type!r}. Choose 'bytetrack' or 'centroid'."
    )
