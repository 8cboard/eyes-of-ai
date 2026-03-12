#!/usr/bin/env python3
"""
single_file_demo.py — VisionTracker all-in-one runnable demo.

Complete pipeline in a single file:
  720p webcam → YOLOv8n detection → ByteTrack/centroid tracking →
  IoU+velocity stillness detection → multi-object batched identification
  via OpenRouter/Nemotron Nano 12B VL → progress overlay → cached labels

Identification backend: OpenRouter free tier (no credit card needed)
  Model: nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free
  Get a free API key at: https://openrouter.ai

Quick start:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-or-v1-...
    python single_file_demo.py

Optional flags:
    --openrouter-key KEY   Pass key directly instead of env var
    --tracker centroid     Use centroid tracker (no supervision dependency)
    --grayscale            Reduce load on very slow CPUs
    --skip-frames 3        Detect every 3rd frame
    --batch-size 4         Crops per API call (1–4)

ASSUMPTIONS:
  1. Python 3.9+
  2. OpenCV with GUI support: pip install opencv-python
  3. YOLOv8n weights auto-downloaded on first run (~6 MB)
  4. Internet connection required for OpenRouter identification
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import base64
import os
import queue
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict, deque
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Optional imports (lazy, with graceful fallbacks)
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    _YOLO = None

try:
    import supervision as sv
    _SV_OK = True
except ImportError:
    _SV_OK = False
    sv = None

try:
    import requests as _requests_mod
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    _requests_mod = None


# ══════════════════════════════════════════════════════════════════════════════
# ① DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class DetectionResult:
    """Output of one detector forward pass (or the cached result on skipped frames)."""
    __slots__ = ('xyxy', 'confidences', 'class_ids', 'class_names', 'frame_index', 'inference_ms')

    def __init__(
        self,
        xyxy: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
        frame_index: int = 0,
        inference_ms: float = 0.0,
    ) -> None:
        self.xyxy         = xyxy         if xyxy         is not None else np.empty((0, 4), dtype=np.float32)
        self.confidences  = confidences  if confidences  is not None else np.empty((0,),   dtype=np.float32)
        self.class_ids    = class_ids    if class_ids    is not None else np.empty((0,),   dtype=np.int32)
        self.class_names  = class_names  if class_names  is not None else []
        self.frame_index  = frame_index
        self.inference_ms = inference_ms

    @property
    def count(self) -> int:
        return len(self.xyxy)


class TrackedObject:
    """A detection linked to a persistent track ID."""
    __slots__ = ('track_id', 'xyxy', 'class_id', 'class_name', 'confidence', 'age')

    def __init__(
        self,
        track_id: int,
        xyxy: np.ndarray,
        class_id: int,
        class_name: str,
        confidence: float,
        age: int = 0,
    ) -> None:
        self.track_id   = track_id
        self.xyxy       = xyxy
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.age        = age

    @property
    def centroid(self) -> tuple:
        x1, y1, x2, y2 = self.xyxy
        return float((x1 + x2) / 2), float((y1 + y2) / 2)


# ══════════════════════════════════════════════════════════════════════════════
# ② DETECTOR  (YOLOv8n with frame-skip)
# ══════════════════════════════════════════════════════════════════════════════

class Detector:
    """
    YOLOv8 detector with frame-skip to maintain high FPS on CPU.

    Performance guide:
      skip_frames=1, imgsz=640  → all frames, ~12 FPS on i7 CPU
      skip_frames=2, imgsz=640  → every other frame, ~22 FPS
      skip_frames=3, imgsz=320  → every 3rd frame at small res, ~35 FPS
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        skip_frames: int = 2,
        conf: float = 0.35,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        if not _YOLO_OK:
            raise ImportError("pip install ultralytics")
        self.skip_frames = max(1, skip_frames)
        self._count = 0
        self._last: DetectionResult = DetectionResult()
        self.device = device or _best_device()
        print(f"[Detector] {model_name} | device={self.device} | skip={skip_frames} | imgsz={imgsz}")
        self._model = _YOLO(model_name)
        self._conf = conf
        self._imgsz = imgsz
        self._names: dict[int, str] = self._model.names

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self._count += 1
        if self._count % self.skip_frames != 0:
            return self._last

        # Ensure BGR (handles grayscale input)
        img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        t0 = time.perf_counter()
        res = self._model.predict(
            source=img, conf=self._conf, imgsz=self._imgsz,
            device=self.device, verbose=False,
        )[0]
        ms = (time.perf_counter() - t0) * 1000

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            self._last = DetectionResult(frame_index=self._count, inference_ms=ms)
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
            self._last = DetectionResult(
                xyxy=xyxy,
                confidences=boxes.conf.cpu().numpy().astype(np.float32),
                class_ids=cls_ids,
                class_names=[self._names.get(int(c), "?") for c in cls_ids],
                frame_index=self._count,
                inference_ms=ms,
            )
        return self._last


def _best_device() -> str:
    """Auto-detect best compute device. Works with or without torch installed."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# ③ TRACKER
#   Primary:  ByteTrackWrapper (supervision)
#   Fallback: CentroidTracker  (pure Python + scipy)
# ══════════════════════════════════════════════════════════════════════════════

class ByteTrackWrapper:
    """
    Wraps supervision.ByteTracker.
    Handles API differences across supervision versions (0.18 – 0.25+).
    """
    name = "ByteTrack"

    def __init__(self) -> None:
        if not _SV_OK:
            raise ImportError("pip install supervision")
        self._t = _make_sv_bytetracker()
        print("[Tracker] ByteTrack (supervision)")

    def update(self, det: DetectionResult) -> list[TrackedObject]:
        if det.count == 0:
            try:
                self._t.update_with_detections(sv.Detections.empty())
            except Exception:
                pass
            return []
        sv_det = sv.Detections(
            xyxy=det.xyxy, confidence=det.confidences, class_id=det.class_ids
        )
        try:
            tracked = self._t.update_with_detections(sv_det)
        except Exception as e:
            print(f"[Tracker] ByteTrack update error: {e}")
            return []
        result = []
        for i in range(len(tracked)):
            if tracked.tracker_id is None: continue
            tid = tracked.tracker_id[i]
            if tid is None: continue
            result.append(TrackedObject(
                track_id=int(tid),
                xyxy=tracked.xyxy[i].astype(np.float32),
                class_id=int(tracked.class_id[i]) if tracked.class_id is not None else 0,
                class_name=det.class_names[i] if i < len(det.class_names) else "?",
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
            ))
        return result


def _make_sv_bytetracker():
    """Create sv.ByteTracker handling constructor API changes across versions."""
    import inspect
    params = set(inspect.signature(sv.ByteTracker.__init__).parameters.keys())
    if "track_high_thresh" in params:
        return sv.ByteTracker(track_high_thresh=0.25, track_low_thresh=0.1,
                              new_track_thresh=0.25, track_buffer=30,
                              match_thresh=0.8, frame_rate=30)
    elif "track_activation_threshold" in params:
        return sv.ByteTracker(track_activation_threshold=0.25, lost_track_buffer=30,
                              minimum_matching_threshold=0.8, frame_rate=30)
    else:
        try:
            return sv.ByteTracker(0.25, 30, 0.8, 30)
        except TypeError:
            return sv.ByteTracker()


class CentroidTracker:
    """
    Pure-Python IoU + centroid-distance tracker using the Hungarian algorithm.

    No neural network, minimal dependencies (just scipy).
    Recommended for CPU-only machines where even supervision feels heavy.

    Expected performance: same as ByteTrack in controlled scenes;
    may swap IDs through occlusions (ByteTrack handles this better).
    """
    name = "Centroid"

    def __init__(self, max_disappeared: int = 20) -> None:
        self._next_id = 1
        self._tracks: OrderedDict[int, TrackedObject] = OrderedDict()
        self._gone: dict[int, int] = {}
        self._max_gone = max_disappeared
        print("[Tracker] CentroidTracker (CPU fallback)")

    def update(self, det: DetectionResult) -> list[TrackedObject]:
        if det.count == 0:
            for tid in list(self._tracks):
                self._gone[tid] += 1
                if self._gone[tid] > self._max_gone:
                    self._drop(tid)
            return list(self._tracks.values())

        if not self._tracks:
            for i in range(det.count):
                self._add(det, i)
            return list(self._tracks.values())

        tids = list(self._tracks.keys())
        tboxes = np.array([self._tracks[t].xyxy for t in tids])
        dboxes = det.xyxy
        cost = _cost_matrix(tboxes, dboxes)
        rows, cols = linear_sum_assignment(cost)

        matched_t, matched_d = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] < 1.5:
                self._update(tids[r], det, c)
                matched_t.add(tids[r])
                matched_d.add(c)

        for r, tid in enumerate(tids):
            if tid not in matched_t:
                self._gone[tid] += 1
                if self._gone[tid] > self._max_gone:
                    self._drop(tid)

        for c in range(det.count):
            if c not in matched_d:
                self._add(det, c)

        return list(self._tracks.values())

    def _add(self, det: DetectionResult, i: int) -> None:
        tid = self._next_id; self._next_id += 1
        self._tracks[tid] = TrackedObject(
            track_id=tid, xyxy=det.xyxy[i].copy(),
            class_id=int(det.class_ids[i]),
            class_name=det.class_names[i] if i < len(det.class_names) else "?",
            confidence=float(det.confidences[i]), age=1,
        )
        self._gone[tid] = 0

    def _update(self, tid: int, det: DetectionResult, i: int) -> None:
        t = self._tracks[tid]
        self._tracks[tid] = TrackedObject(
            track_id=tid, xyxy=det.xyxy[i].copy(),
            class_id=int(det.class_ids[i]),
            class_name=det.class_names[i] if i < len(det.class_names) else t.class_name,
            confidence=float(det.confidences[i]), age=t.age + 1,
        )
        self._gone[tid] = 0

    def _drop(self, tid: int) -> None:
        self._tracks.pop(tid, None); self._gone.pop(tid, None)


def _cost_matrix(ta: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Combined IoU + normalised centroid distance cost matrix."""
    a = ta[:, None]; b = db[None, :]
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (ta[:, 2] - ta[:, 0]) * (ta[:, 3] - ta[:, 1])
    area_b = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    cx_a = (ta[:, 0] + ta[:, 2]) / 2; cy_a = (ta[:, 1] + ta[:, 3]) / 2
    cx_b = (db[:, 0] + db[:, 2]) / 2; cy_b = (db[:, 1] + db[:, 3]) / 2
    dist = np.sqrt((cx_a[:, None] - cx_b[None, :]) ** 2 + (cy_a[:, None] - cy_b[None, :]) ** 2)
    dist_norm = np.clip(dist / 200.0, 0, 1)

    return ((1 - iou) * 0.7 + dist_norm * 0.3).astype(np.float32)


def build_tracker(name: str) -> ByteTrackWrapper | CentroidTracker:
    if name == "bytetrack":
        if _SV_OK:
            return ByteTrackWrapper()
        print("[Tracker] supervision not found, using CentroidTracker")
    return CentroidTracker()


# ══════════════════════════════════════════════════════════════════════════════
# ④ STILLNESS DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class StillnessDetector:
    """
    Per-track stillness policy using centroid velocity + consecutive IoU.

    Parameters
    ----------
    history_len : int  (N)
        Sliding window of observations (centroids + boxes).
    still_frames : int  (M)
        Consecutive frames below threshold required to flag as still.
    velocity_thresh : float
        Max mean centroid displacement per frame (pixels) to count as still.
    iou_thresh : float
        Min mean pairwise IoU between consecutive boxes to count as stable.

    Alternative: set use_optical_flow=True for camera-motion robustness
    (see stability.py OpticalFlowStillnessDetector for the full implementation).
    """

    def __init__(
        self,
        history_len: int = 15,
        still_frames: int = 10,
        velocity_thresh: float = 5.0,
        iou_thresh: float = 0.85,
    ) -> None:
        self.N = history_len
        self.M = still_frames
        self.vt = velocity_thresh
        self.it = iou_thresh

        # {track_id: {"cx": deque, "cy": deque, "boxes": deque, "streak": int, "stable": bool}}
        self._s: dict[int, dict] = defaultdict(lambda: {
            "cx": deque(maxlen=history_len),
            "cy": deque(maxlen=history_len),
            "boxes": deque(maxlen=history_len),
            "streak": 0, "stable": False,
        })

    def update(self, tid: int, bbox: np.ndarray) -> None:
        s = self._s[tid]
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        s["cx"].append(cx); s["cy"].append(cy); s["boxes"].append(bbox.copy())

        if len(s["cx"]) < 2:
            return

        cxs, cys = np.array(s["cx"]), np.array(s["cy"])
        vel = float(np.mean(np.sqrt(np.diff(cxs) ** 2 + np.diff(cys) ** 2)))

        boxes = np.array(s["boxes"])
        ious = [_box_iou_1d(boxes[i], boxes[i + 1]) for i in range(len(boxes) - 1)]
        mean_iou = float(np.mean(ious)) if ious else 0.0

        if vel < self.vt and mean_iou > self.it:
            s["streak"] += 1
        else:
            s["streak"] = 0
        s["stable"] = s["streak"] >= self.M

    def is_still(self, tid: int) -> bool:
        return self._s[tid]["stable"]

    def reset(self, tid: int) -> None:
        self._s[tid]["streak"] = 0; self._s[tid]["stable"] = False

    def remove(self, tid: int) -> None:
        self._s.pop(tid, None)

    def velocity(self, tid: int) -> float:
        s = self._s.get(tid)
        if not s or len(s["cx"]) < 2:
            return 0.0
        cxs, cys = np.array(s["cx"]), np.array(s["cy"])
        return float(np.mean(np.sqrt(np.diff(cxs) ** 2 + np.diff(cys) ** 2)))


def _box_iou_1d(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    denom = ua + ub - inter
    return float(inter / denom) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ IDENTIFICATION SERVICE
#   Backends: blip (local) | huggingface (free API)
#   Runs in a background ThreadPoolExecutor
#   Writes results to a thread-safe progress dict
# ══════════════════════════════════════════════════════════════════════════════

# ── OpenRouter / Nemotron Nano 12B VL constants ───────────────────────────────
OR_API_URL     = "https://openrouter.ai/api/v1/chat/completions"
OR_MODEL       = "nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free"
OR_REFILL_RATE = 0.22    # tokens/sec → 13.2 RPM (under 20 RPM free limit)
OR_BUCKET_CAP  = 3       # burst capacity
OR_BATCH_SIZE  = 4       # crops per API call
OR_BATCH_WAIT  = 1.5     # seconds to wait for a fuller batch
OR_JPEG_Q      = 72
OR_MAX_PX      = 336

# CLI equivalent (replace <B64> with base64-encoded JPEG, no billing required):
# curl -X POST https://openrouter.ai/api/v1/chat/completions \
#   -H "Authorization: Bearer $OPENROUTER_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{"model":"nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free",
#        "messages":[{"role":"user","content":[
#          {"type":"text","text":"Describe this object in one sentence."},
#          {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<B64>"}}
#        ]}],"max_tokens":80}'


class _TokenBucket:
    """Thread-safe token bucket for rate limiting."""
    def __init__(self, rate=OR_REFILL_RATE, cap=OR_BUCKET_CAP):
        self._rate = rate; self._cap = float(cap)
        self._tokens = float(cap); self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._cap, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.5)

    def peek(self) -> float:
        with self._lock:
            now = time.monotonic()
            return min(self._cap, self._tokens + (now - self._last) * self._rate)


class _IDCache:
    """Thread-safe per-track TTL cache."""
    def __init__(self, ttl: float = 45.0):
        self._ttl = ttl; self._d: dict[int, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, tid: int) -> Optional[str]:
        with self._lock:
            e = self._d.get(tid)
            if e is None or time.monotonic() - e[1] > self._ttl:
                self._d.pop(tid, None); return None
            return e[0]

    def set(self, tid: int, label: str):
        with self._lock: self._d[tid] = (label, time.monotonic())


class _Pending:
    """A crop waiting for identification. Sortable by priority (lower = served first)."""
    __slots__ = ('priority', 'track_id', 'crop', 'cls_name', 'cls_id')

    def __init__(self, priority: float, track_id: int, crop: np.ndarray,
                 cls_name: str, cls_id: int) -> None:
        self.priority  = priority
        self.track_id  = track_id
        self.crop      = crop
        self.cls_name  = cls_name
        self.cls_id    = cls_id

    def __lt__(self, other: '_Pending') -> bool:
        return self.priority < other.priority

    def __le__(self, other: '_Pending') -> bool:
        return self.priority <= other.priority


class IDService:
    """
    Multi-object identification via OpenRouter / Nemotron Nano 12B VL (free).

    RATE LIMIT STRATEGY — three layers:
      1. Batch dispatch: up to OR_BATCH_SIZE crops per API call
         → 1 request identifies multiple objects simultaneously
      2. Token-bucket rate limiter: ~13 RPM, stays under 20 RPM free limit
      3. HTTP 429 back-off: re-queues items at front, sleeps Retry-After

    A SINGLE background dispatcher thread owns all HTTP traffic.
    The main thread only calls submit() — never blocks.

    API key: get free at https://openrouter.ai (no credit card for :free models)
    Set: export OPENROUTER_API_KEY=sk-or-v1-...
    """

    def __init__(
        self,
        api_key:   str   = "",
        cache_ttl: float = 45.0,
        batch_size: int  = OR_BATCH_SIZE,
        batch_wait: float = OR_BATCH_WAIT,
    ) -> None:
        self._key       = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._key:
            raise ValueError(
                "OpenRouter API key required.\n"
                "  Get a FREE key at: https://openrouter.ai\n"
                "  export OPENROUTER_API_KEY=sk-or-v1-..."
            )
        self._bucket    = _TokenBucket()
        self._cache     = _IDCache(cache_ttl)
        self._bsz       = batch_size
        self._bwait     = batch_wait
        self.progress: dict[int, dict] = {}
        self._plock     = threading.Lock()
        self._q: queue.PriorityQueue[_Pending] = queue.PriorityQueue()
        self._inflight: set[int] = set()
        self._ifl_lock  = threading.Lock()
        self._alive     = True

        if _REQUESTS_OK:
            self._session = _requests_mod.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self._key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/visiontracker",
                "X-Title":       "VisionTracker",
            })
        else:
            self._session = None

        self._thread = threading.Thread(target=self._loop, name="id-disp", daemon=True)
        self._thread.start()

        eff = OR_REFILL_RATE * batch_size * 60
        print(f"[IDService] OpenRouter / {OR_MODEL}")
        print(f"[IDService] ~{OR_REFILL_RATE*60:.0f} RPM × batch {batch_size} = ~{eff:.0f} IDs/min")

    def cached(self, tid: int) -> Optional[str]:
        return self._cache.get(tid)

    def submit(self, tid: int, crop: np.ndarray, cls_name: str, cls_id: int) -> bool:
        if self._cache.get(tid): return False
        with self._ifl_lock:
            if tid in self._inflight: return False
            self._inflight.add(tid)
        self._q.put(_Pending(priority=time.monotonic(), track_id=tid,
                             crop=crop.copy(), cls_name=cls_name, cls_id=cls_id))
        self._set(tid, "queued", 0.0)
        return True

    def inject_cached(self, tid: int):
        lbl = self._cache.get(tid)
        if lbl: self._set(tid, "done", 1.0, label=lbl)

    def queue_depth(self) -> int: return self._q.qsize()

    def shutdown(self):
        self._alive = False
        self._q.put(_Pending(float("inf"), -1, np.zeros((1,1,3),np.uint8), "__stop__", -1))
        self._thread.join(timeout=2.0)

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _loop(self):
        while self._alive:
            try: first = self._q.get(timeout=1.0)
            except queue.Empty: continue
            if first.track_id == -1: break

            batch = [first]
            deadline = time.monotonic() + self._bwait
            while len(batch) < self._bsz:
                left = deadline - time.monotonic()
                if left <= 0: break
                try:
                    item = self._q.get(timeout=left)
                except queue.Empty: break
                if item.track_id == -1: self._alive = False; break
                if self._cache.get(item.track_id):
                    self._set(item.track_id, "done", 1.0, label=self._cache.get(item.track_id))
                    with self._ifl_lock: self._inflight.discard(item.track_id)
                    continue
                batch.append(item)

            if not batch: continue
            for item in batch: self._set(item.track_id, "identifying", 0.2)

            self._bucket.acquire()     # blocks if rate limit reached
            self._fire(batch)

    def _fire(self, batch: list[_Pending]):
        for item in batch: self._set(item.track_id, "identifying", 0.5)
        try:
            results = self._call_api(batch)
            for item in batch:
                lbl = results.get(item.track_id, item.cls_name)
                self._cache.set(item.track_id, lbl)
                self._set(item.track_id, "done", 1.0, label=lbl)
                print(f"[IDService] ✓ #{item.track_id} ({item.cls_name}) → \"{lbl[:60]}\"")
        except _RateLimit as e:
            print(f"[IDService] ⚠️ Rate limited — sleeping {e.wait:.0f}s, re-queuing {len(batch)}")
            for item in batch:
                self._set(item.track_id, "queued", 0.0,
                          label=f"Rate limited, retry in {e.wait:.0f}s…")
            time.sleep(e.wait)
            for item in batch:
                item.priority = time.monotonic() - 1000
                self._q.put(item)
            return
        except Exception as exc:
            msg = str(exc)[:80]
            print(f"[IDService] ✗ {msg}")
            for item in batch:
                self._cache.set(item.track_id, item.cls_name)
                self._set(item.track_id, "error", 1.0, label=item.cls_name, error=msg)
        finally:
            with self._ifl_lock:
                for item in batch: self._inflight.discard(item.track_id)

    def _call_api(self, batch: list[_Pending]) -> dict[int, str]:
        if not _REQUESTS_OK or not self._session:
            raise RuntimeError("pip install requests")
        n = len(batch)
        if n == 1:
            prompt = (f"The detector says this is '{batch[0].cls_name}'. "
                      "Describe it in ONE sentence ≤ 12 words: colour, shape, key features.")
        else:
            lines = "\n".join(f"{i+1}. <description>" for i in range(n))
            prompt = (f"I'll show you {n} object crops. "
                      "One sentence each (≤ 12 words: colour, shape, features).\n"
                      f"Reply ONLY:\n{lines}")

        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, item in enumerate(batch):
            if n > 1:
                content.append({"type": "text", "text": f"Image {i+1} — {item.cls_name}:"})
            b64 = _enc_crop(item.crop)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        payload = {"model": OR_MODEL, "messages": [{"role":"user","content":content}],
                   "max_tokens": 100*n, "temperature": 0.2}
        import json as _json
        resp = self._session.post(OR_API_URL, data=_json.dumps(payload), timeout=20)
        if resp.status_code == 429:
            raise _RateLimit(float(resp.headers.get("Retry-After", 30)))
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_or(raw, batch)

    def _set(self, tid, status, progress, label=None, error=None):
        with self._plock:
            prev = self.progress.get(tid, {})
            self.progress[tid] = {"status": status, "progress": progress,
                                  "label": label if label is not None else prev.get("label"),
                                  "error": error}


class _RateLimit(Exception):
    def __init__(self, wait=30.0): self.wait = wait


def _enc_crop(crop: np.ndarray, mx=OR_MAX_PX, q=OR_JPEG_Q) -> str:
    h, w = crop.shape[:2]
    if max(h, w) > mx:
        s = mx / max(h, w)
        crop = cv2.resize(crop, (max(1,int(w*s)), max(1,int(h*s))), cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def _parse_or(text: str, batch: list[_Pending]) -> dict[int, str]:
    import re as _re
    n = len(batch)
    if n == 1:
        return {batch[0].track_id: _re.sub(r"^1[.)]\s*","",text.strip()).strip() or batch[0].cls_name}
    pat = _re.compile(r"^\s*(\d+)[.)]\s*(.+)$")
    parsed = {}
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n: parsed[idx] = m.group(2).strip()
    fallbacks = [l.strip() for l in text.splitlines() if l.strip()]
    return {item.track_id: parsed.get(i, fallbacks[i] if i < len(fallbacks) else item.cls_name)
            for i, item in enumerate(batch)}


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ UI OVERLAY  (OpenCV drawing)
# ══════════════════════════════════════════════════════════════════════════════

_PALETTE = [
    (86, 180, 233), (230, 159, 0), (0, 158, 115), (204, 121, 167),
    (0, 114, 178), (213, 94, 0), (240, 228, 66), (0, 204, 153),
    (255, 102, 102), (153, 102, 255), (255, 178, 102), (102, 255, 178),
]

def _clr(tid: int) -> tuple[int, int, int]:
    return _PALETTE[tid % len(_PALETTE)]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_frame(
    frame: np.ndarray,
    tracked: list[TrackedObject],
    progress: dict,
    fps: float,
    mode: str = "local",
    backend: str = "blip",
) -> np.ndarray:
    """
    Render all overlays onto a copy of *frame*.

    Visual elements per object:
      • Coloured bounding box (colour cycles by track_id)
      • Label block: "#ID  class_name  conf%"
      • Identification result or "Identifying… XX%"
      • Progress bar along the bottom edge of the box
      • Green/orange dot: done / in-progress
    HUD: FPS | track count | mode
    """
    out = frame.copy()

    for obj in tracked:
        c = _clr(obj.track_id)
        x1, y1, x2, y2 = obj.xyxy.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

        pg = progress.get(obj.track_id, {})
        status = pg.get("status", "idle")
        lbl = pg.get("label")
        pct = pg.get("progress", 0.0)

        header = f"#{obj.track_id} {obj.class_name} {obj.confidence:.0%}"
        id_line: Optional[str] = None
        if status == "done" and lbl:
            id_line = lbl[:42]
        elif status in ("identifying", "queued"):
            partial = (lbl or "")[:20]
            id_line = f"Identifying… {int(pct*100)}%  {partial}"

        # Label block background + text
        lines = [header] + ([id_line] if id_line else [])
        fs = 0.45; lh = 17; pad = 3
        bh = lh * len(lines) + pad * 2
        bw = max(cv2.getTextSize(l, FONT, fs, 1)[0][0] for l in lines) + pad * 2
        lx, ly = max(0, x1), max(bh, y1) - bh
        sub = out[ly:ly+bh, lx:lx+bw]
        if sub.size > 0:
            ov = sub.copy()
            cv2.rectangle(ov, (0, 0), (bw, bh), c, -1)
            cv2.addWeighted(ov, 0.4, sub, 0.6, 0, sub)
            out[ly:ly+bh, lx:lx+bw] = sub
        for i, line in enumerate(lines):
            cv2.putText(out, line, (lx + pad, ly + pad + (i + 1) * lh - 3),
                        FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)

        # Progress bar
        if status in ("identifying", "queued") and (x2 - x1) > 20:
            cv2.rectangle(out, (x1, y2), (x2, y2 + 5), (40, 40, 40), -1)
            w_fill = int((x2 - x1) * min(1.0, pct))
            if w_fill > 0:
                cv2.rectangle(out, (x1, y2), (x1 + w_fill, y2 + 5), c, -1)

        # Status dot
        dot_c = (0, 220, 0) if status == "done" else (0, 165, 255) if status in ("identifying", "queued") else None
        if dot_c:
            cv2.circle(out, (x1 + 6, y1 + 6), 4, dot_c, -1)

    # HUD bar
    h, w = out.shape[:2]
    hud = out[0:22, 0:w].copy()
    cv2.rectangle(hud, (0, 0), (w, 22), (15, 15, 15), -1)
    cv2.addWeighted(hud, 0.65, out[0:22, 0:w], 0.35, 0, out[0:22, 0:w])
    cv2.putText(out,
        f"FPS: {fps:.1f}  |  Tracks: {len(tracked)}  |  {mode}/{backend}  |  Q=quit",
        (8, 15), FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ HELPERS & FPS COUNTER
# ══════════════════════════════════════════════════════════════════════════════

class FPS:
    def __init__(self, alpha: float = 0.1) -> None:
        self._fps = 0.0; self._t: Optional[float] = None; self._a = alpha

    def tick(self) -> float:
        now = time.perf_counter()
        if self._t:
            inst = 1.0 / max(1e-6, now - self._t)
            self._fps = self._a * inst + (1 - self._a) * self._fps
        self._t = now
        return self._fps


def crop_pad(frame: np.ndarray, xyxy: np.ndarray, pad: int = 8) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(xyxy[0]) - pad), max(0, int(xyxy[1]) - pad)
    x2, y2 = min(w, int(xyxy[2]) + pad), min(h, int(xyxy[3]) + pad)
    return frame[y1:y2, x1:x2].copy()


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionTracker single-file demo (OpenRouter/Nemotron)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default="0", help="Webcam index or video file path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--grayscale", action="store_true",
                   help="Grayscale input (faster on CPU)")
    p.add_argument("--detector", default="yolov8n.pt",
                   choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    p.add_argument("--skip-frames", type=int, default=2,
                   help="Detector runs every N frames")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "centroid"])
    p.add_argument("--still-frames", type=int, default=10,
                   help="M: consecutive below-threshold frames to trigger ID")
    p.add_argument("--still-window", type=int, default=15,
                   help="N: centroid/IoU history window")
    p.add_argument("--id-ttl", type=float, default=45.0,
                   help="Seconds before re-ID of a cached track")
    p.add_argument("--openrouter-key", default=None,
                   help="OpenRouter API key (free). "
                        "Get at https://openrouter.ai — no credit card for :free models. "
                        "Also reads OPENROUTER_API_KEY env var.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Max crops per OpenRouter API call (1–4)")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress OpenCV window")
    p.add_argument("--device", default=None, help="cpu | cuda | mps")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("\n" + "=" * 55)
    print("  VisionTracker single-file demo")
    print("  ID backend: OpenRouter / Nemotron Nano 12B VL (free)")
    print(f"  Tracker: {args.tracker}  |  Detector: {args.detector}")
    print("=" * 55 + "\n")

    # ── Build components ─────────────────────────────────────────────────────
    detector = Detector(
        model_name=args.detector,
        skip_frames=args.skip_frames,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    tracker = build_tracker(args.tracker)
    stillness = StillnessDetector(
        history_len=args.still_window,
        still_frames=args.still_frames,
    )
    id_svc = IDService(
        api_key    = args.openrouter_key,
        cache_ttl  = args.id_ttl,
        batch_size = args.batch_size,
    )
    fps_counter = FPS()

    # ── Open capture ─────────────────────────────────────────────────────────
    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {args.input}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Main] Capture: {aw}×{ah} | Press Q to quit\n")

    active_ids: set[int] = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Main] Stream ended.")
                break

            disp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if args.grayscale else frame
            if args.grayscale:
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

            # ── Detect ───────────────────────────────────────────────────────
            det = detector.detect(disp)

            # ── Track ────────────────────────────────────────────────────────
            tracked = tracker.update(det)
            cur_ids = {obj.track_id for obj in tracked}

            for tid in active_ids - cur_ids:
                stillness.remove(tid)
            active_ids = cur_ids

            # ── Stillness + multi-object ID submission ────────────────────────
            for obj in tracked:
                stillness.update(obj.track_id, obj.xyxy)
                if stillness.is_still(obj.track_id):
                    if id_svc.cached(obj.track_id) is None:
                        crop = crop_pad(frame, obj.xyxy)
                        if crop.size > 0:
                            submitted = id_svc.submit(
                                obj.track_id, crop, obj.class_name, obj.class_id
                            )
                            if submitted:
                                print(f"[Main] → queued #{obj.track_id} "
                                      f"({obj.class_name}) | q:{id_svc.queue_depth()}")
                            stillness.reset(obj.track_id)
                    else:
                        id_svc.inject_cached(obj.track_id)

            # ── Render ───────────────────────────────────────────────────────
            fps = fps_counter.tick()
            annotated = draw_frame(
                disp, tracked, id_svc.progress, fps,
                mode="openrouter",
                backend=f"nemotron q:{id_svc.queue_depth()}",
            )

            if not args.no_display:
                cv2.imshow("VisionTracker", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[Main] Quit.")
                    break

    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        id_svc.shutdown()
        print("[Main] Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
