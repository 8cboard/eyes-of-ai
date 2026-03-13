"""
id_service.py — Remote LLM Identification Service (Annotated Frame Mode)

Sends full annotated frames (not crops) to a remote LLM server for identification.
The LLM receives a frame with colored bounding boxes and returns a single common
noun describing each object based on its color.

Architecture:
  - Local: Edge detection → Tracking → Draw colored boxes → Send annotated frame
  - Remote: LLM identifies objects by color → Returns {track_id: noun}

API Endpoint (remote server):
  POST /identify
  {
    "annotated_image": "base64_jpeg",
    "color_map": {"red": 1, "blue": 2, "green": 3, ...}
  }
  
  Response:
  {
    "results": [
      {"track_id": 1, "description": "person"},
      {"track_id": 2, "description": "car"}
    ]
  }

Prompt engineering ensures single common noun responses (person, car, dog, chair, etc.)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    import requests as _req
    _REQ_OK = True
except ImportError:  # pragma: no cover
    _REQ_OK = False
    _req = None  # type: ignore

log = logging.getLogger("id_service")

# Configuration constants
DEFAULT_REMOTE_URL = ""
DEFAULT_BATCH_SIZE = 8
DEFAULT_BATCH_WAIT_MS = 1_000
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0

# Image encoding
JPEG_QUALITY = 85
MAX_FRAME_SIZE = 1280  # Longest edge for annotated frame


# Color palette must match ui_overlay.py exactly
_COLOR_NAMES = [
    "sky blue", "orange", "green", "pink", "blue", "vermillion",
    "yellow", "teal", "red", "purple", "peach", "mint",
    "lime", "light blue", "magenta", "light yellow", "cyan",
    "olive", "dark cyan", "dark magenta"
]

_PALETTE = [
    (86, 180, 233),   # sky blue
    (230, 159, 0),    # orange
    (0, 158, 115),    # green
    (204, 121, 167),  # pink
    (0, 114, 178),    # blue
    (213, 94, 0),     # vermillion
    (240, 228, 66),   # yellow
    (0, 204, 153),    # teal
    (255, 102, 102),  # red
    (153, 102, 255),  # purple
    (255, 178, 102),  # peach
    (102, 255, 178),  # mint
    (178, 255, 102),  # lime
    (102, 178, 255),  # light blue
    (255, 102, 178),  # magenta
    (255, 255, 102),  # light yellow
    (102, 255, 255),  # cyan
    (204, 204, 0),    # olive
    (0, 204, 204),    # dark cyan
    (204, 0, 204),    # dark magenta
]


def _color_name(track_id: int) -> str:
    """Get the color name for a track_id (matches UI palette)."""
    return _COLOR_NAMES[track_id % len(_COLOR_NAMES)]


def _color_bgr(track_id: int) -> tuple[int, int, int]:
    """Get the BGR color tuple for a track_id (matches UI palette)."""
    return _PALETTE[track_id % len(_PALETTE)]


@dataclass(order=True)
class PendingItem:
    """One frame awaiting identification."""
    priority:     float       = field()
    track_ids:    list[int]   = field(compare=False)
    frame:        np.ndarray  = field(compare=False)
    submitted_at: float       = field(compare=False, default_factory=time.monotonic)


class IdentificationCache:
    """Thread-safe per-track label cache with configurable TTL."""

    def __init__(self, ttl_seconds: float = 45.0) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._store: dict[int, tuple[str, float]] = {}

    def get(self, track_id: int) -> Optional[str]:
        """Return cached label or None if absent/expired."""
        with self._lock:
            entry = self._store.get(track_id)
            if entry is None:
                return None
            label, ts = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[track_id]
                return None
            return label

    def set(self, track_id: int, label: str) -> None:
        with self._lock:
            self._store[track_id] = (label, time.monotonic())

    def invalidate(self, track_id: int) -> None:
        with self._lock:
            self._store.pop(track_id, None)

    def evict_expired(self) -> int:
        """Lazily evict expired entries; return count removed."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, (_, ts) in self._store.items() if now - ts > self._ttl]
            for k in stale:
                del self._store[k]
        return len(stale)


class RemoteLLMClient:
    """HTTP client for the remote LLM identification server.

    Parameters
    ----------
    base_url : str
        The public URL of the remote server
    api_key : str, optional
        Optional API key for server authentication
    timeout : float
        Request timeout in seconds
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT_S,
    ) -> None:
        if not base_url:
            raise ValueError(
                "Remote server URL is required.\n"
                "  1. Run the remote server (see remote_server/README.md)\n"
                "  2. Copy the public URL\n"
                "  3. python main.py --remote-url https://xxx.ngrok.io"
            )
        if not _REQ_OK:
            raise ImportError("pip install requests")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._session = _req.Session()

        # Health check on init
        self._check_health()

    def _check_health(self) -> None:
        """Verify server is reachable and healthy."""
        try:
            resp = self._session.get(
                f"{self._base_url}/health",
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"[IDService] Connected to server: {data.get('model', 'unknown')}")
        except Exception as exc:
            print(f"[IDService] Health check warning: {exc}")
            print(f"[IDService] Server may still be starting up...")

    def _headers(self) -> dict[str, str]:
        """Build request headers with optional auth."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def identify_frame(
        self,
        frame: np.ndarray,
        track_ids: list[int]
    ) -> dict[int, str]:
        """Identify objects in an annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            Annotated frame with colored bounding boxes
        track_ids : list[int]
            List of track IDs present in the frame

        Returns
        -------
        dict[int, str]
            {track_id: description_string} for each track
        """
        # Encode frame to base64 JPEG
        img_b64 = _encode_frame(frame)

        # Build color map
        color_map = {_color_name(tid): tid for tid in track_ids}

        payload = {
            "annotated_image": img_b64,
            "color_map": color_map,
        }

        last_exception: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.post(
                    f"{self._base_url}/identify",
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self._timeout,
                )
                resp.raise_for_status()

                data = resp.json()
                results = {}
                for result in data.get("results", []):
                    tid = result.get("track_id")
                    desc = result.get("description", "")
                    if tid is not None:
                        results[tid] = desc

                # Fill in any missing results with "object"
                for tid in track_ids:
                    if tid not in results:
                        results[tid] = "object"

                return results

            except _req.exceptions.RequestException as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                    print(f"[IDService] Request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    break

        raise last_exception or RuntimeError("Unknown error in identify_frame")


class IdentificationService:
    """Non-blocking identification using remote LLM with annotated frames.

    One background thread owns all HTTP traffic.
    The main thread calls ``submit()`` with an annotated frame.
    The dispatcher sends frames to the remote LLM server.

    Parameters
    ----------
    remote_url : str
        The public URL of the remote server
    api_key : str, optional
        Optional API key for server authentication
    cache_ttl : float
        Seconds a cached result stays valid.
    batch_size : int
        Max track IDs per API call.
    batch_wait_ms : int
        How long (ms) the dispatcher waits for a fuller batch.
    """

    def __init__(
        self,
        remote_url: str = "",
        api_key: Optional[str] = None,
        cache_ttl: float = 45.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_wait_ms: int = DEFAULT_BATCH_WAIT_MS,
    ) -> None:
        remote_url = remote_url or os.environ.get("REMOTE_URL", "")
        api_key = api_key or os.environ.get("REMOTE_API_KEY")

        self._client = RemoteLLMClient(base_url=remote_url, api_key=api_key)
        self._cache = IdentificationCache(ttl_seconds=cache_ttl)
        self._batch_sz = max(1, min(batch_size, 16))
        self._batch_wait = batch_wait_ms / 1_000.0

        # Shared progress dict
        self.progress: dict[int, dict] = {}
        self._prog_lock = threading.Lock()

        # Priority queue
        self._q: queue.PriorityQueue[PendingItem] = queue.PriorityQueue()

        # Track which track_ids are currently queued or in-flight
        self._inflight: set[int] = set()
        self._ifl_lock = threading.Lock()

        # Counters
        self.stats = dict(requests=0, identified=0, errors=0, retries=0)

        # Start dispatcher thread
        self._alive = True
        self._thread = threading.Thread(
            target=self._dispatch_loop, name="id-dispatcher", daemon=True
        )
        self._thread.start()

        print(f"[IDService] Remote LLM identification")
        print(f"[IDService] Server: {remote_url}")
        print(f"[IDService] Batch size: {self._batch_sz} | Batch wait: {batch_wait_ms}ms")
        print(f"[IDService] Cache TTL: {cache_ttl}s")

    def get_cached(self, track_id: int) -> Optional[str]:
        """Return cached label or None if not yet identified / expired."""
        return self._cache.get(track_id)

    def submit(
        self,
        track_ids: list[int],
        annotated_frame: np.ndarray,
        priority: Optional[float] = None,
    ) -> bool:
        """Enqueue an annotated frame for background identification.

        Returns True if enqueued, False if all tracks already in-flight or cached.
        """
        # Filter out cached and in-flight tracks
        new_tracks = []
        with self._ifl_lock:
            for tid in track_ids:
                if self._cache.get(tid) is None and tid not in self._inflight:
                    new_tracks.append(tid)
                    self._inflight.add(tid)

        if not new_tracks:
            return False

        item = PendingItem(
            priority=priority if priority is not None else time.monotonic(),
            track_ids=new_tracks,
            frame=annotated_frame.copy(),
        )
        self._q.put(item)

        for tid in new_tracks:
            self._set_prog(tid, "queued", 0.0)

        return True

    def queue_depth(self) -> int:
        return self._q.qsize()

    def inject_cached(self, track_id: int) -> None:
        """Push a cached label into progress dict."""
        label = self._cache.get(track_id)
        if label is not None:
            self._set_prog(track_id, "done", 1.0, label=label)

    def shutdown(self) -> None:
        """Stop the dispatcher thread gracefully."""
        self._alive = False
        self._q.put(PendingItem(
            priority=float("inf"), track_ids=[],
            frame=np.zeros((1, 1, 3), np.uint8),
        ))
        self._thread.join(timeout=3.0)

    def _dispatch_loop(self) -> None:
        """Collect batches from the queue and send to remote server."""
        while self._alive:
            try:
                first = self._q.get(timeout=1.0)
            except queue.Empty:
                self._cache.evict_expired()
                continue

            if not first.track_ids:  # stop sentinel
                break

            batch: list[PendingItem] = [first]
            all_track_ids = set(first.track_ids)

            # Fill batch
            deadline = time.monotonic() + self._batch_wait
            while len(all_track_ids) < self._batch_sz:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    item = self._q.get(timeout=left)
                except queue.Empty:
                    break

                if not item.track_ids:
                    self._alive = False
                    break

                # Skip if already identified
                skip = False
                for tid in item.track_ids:
                    if self._cache.get(tid) is not None:
                        self._set_prog(tid, "done", 1.0, label=self._cache.get(tid))
                        with self._ifl_lock:
                            self._inflight.discard(tid)
                    elif tid in all_track_ids:
                        skip = True
                    else:
                        all_track_ids.add(tid)

                if not skip:
                    batch.append(item)

            if not batch:
                continue

            for tid in all_track_ids:
                self._set_prog(tid, "identifying", 0.2)

            # Use the most recent frame for identification
            latest_frame = max(batch, key=lambda x: x.submitted_at).frame

            self._fire(latest_frame, list(all_track_ids))

    def _fire(self, frame: np.ndarray, track_ids: list[int]) -> None:
        """Execute API call and distribute results."""
        for tid in track_ids:
            self._set_prog(tid, "identifying", 0.5)

        try:
            self.stats["requests"] += 1
            results = self._client.identify_frame(frame, track_ids)

            for tid in track_ids:
                label = results.get(tid, "object")
                self._cache.set(tid, label)
                self._set_prog(tid, "done", 1.0, label=label)
                self.stats["identified"] += 1
                print(f"[IDService] ✓ #{tid} → \"{label}\"")

        except Exception as exc:
            self.stats["errors"] += 1
            msg = _trunc(str(exc), 80)
            print(f"[IDService] ✗ Error: {msg}")
            for tid in track_ids:
                self._cache.set(tid, "object")
                self._set_prog(tid, "error", 1.0, label="object", error=msg)

        finally:
            with self._ifl_lock:
                for tid in track_ids:
                    self._inflight.discard(tid)

    def _set_prog(
        self,
        track_id: int,
        status: str,
        progress: float,
        label: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._prog_lock:
            prev = self.progress.get(track_id, {})
            self.progress[track_id] = {
                "status": status,
                "progress": progress,
                "label": label if label is not None else prev.get("label"),
                "error": error,
            }


def _encode_frame(
    frame: np.ndarray,
    max_px: int = MAX_FRAME_SIZE,
    quality: int = JPEG_QUALITY,
) -> str:
    """Resize and JPEG-encode a frame, return base64 string."""
    h, w = frame.shape[:2]
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"
