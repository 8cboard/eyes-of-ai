"""
id_service.py — Remote LLM Identification Service (Single Object Mode)

Sends full frames with a SINGLE bounding box drawn to a remote LLM server for 
identification. Each tracked object gets its own request with the box drawn on 
a copy of the full frame.

Architecture:
  - Local: Edge detection → Tracking → For each object: Draw single box → Send frame
  - Remote: LLM identifies the single boxed object → Returns single common noun

API Endpoint (remote server):
  POST /identify
  {
    "annotated_image": "base64_jpeg"
  }
  
  Response:
  {
    "result": "person"  // single common noun
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
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0

# Image encoding
JPEG_QUALITY = 85
MAX_FRAME_SIZE = 1280  # Longest edge for annotated frame


@dataclass(order=True)
class PendingItem:
    """One frame with single box awaiting identification."""
    priority:     float       = field()
    track_id:     int         = field(compare=False)
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
        track_id: int
    ) -> str:
        """Identify the single object in an annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame with single bounding box drawn
        track_id : int
            Track ID of the object in the box

        Returns
        -------
        str
            Single common noun describing the object
        """
        # Encode frame to base64 JPEG
        img_b64 = _encode_frame(frame)

        payload = {
            "annotated_image": img_b64,
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
                result = data.get("result", "object")

                # Validate: should be a single common noun
                if not result or not isinstance(result, str):
                    result = "object"

                return result.strip().lower()

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
    """Non-blocking identification using remote LLM with single-box frames.

    One background thread owns all HTTP traffic.
    The main thread calls ``submit()`` with a frame containing a single box.
    The dispatcher sends individual frames to the remote LLM server.

    Parameters
    ----------
    remote_url : str
        The public URL of the remote server
    api_key : str, optional
        Optional API key for server authentication
    cache_ttl : float
        Seconds a cached result stays valid.
    """

    def __init__(
        self,
        remote_url: str = "",
        api_key: Optional[str] = None,
        cache_ttl: float = 45.0,
        **kwargs,  # Accept but ignore batch_size, batch_wait_ms for compatibility
    ) -> None:
        remote_url = remote_url or os.environ.get("REMOTE_URL", "")
        api_key = api_key or os.environ.get("REMOTE_API_KEY")

        self._client = RemoteLLMClient(base_url=remote_url, api_key=api_key)
        self._cache = IdentificationCache(ttl_seconds=cache_ttl)

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

        print(f"[IDService] Remote LLM identification (single object mode)")
        print(f"[IDService] Server: {remote_url}")
        print(f"[IDService] Cache TTL: {cache_ttl}s")

    def get_cached(self, track_id: int) -> Optional[str]:
        """Return cached label or None if not yet identified / expired."""
        return self._cache.get(track_id)

    def submit(
        self,
        track_id: int,
        frame_with_box: np.ndarray,
        priority: Optional[float] = None,
    ) -> bool:
        """Enqueue a single-object frame for background identification.

        Parameters
        ----------
        track_id : int
            The track ID of the object in the frame.
        frame_with_box : np.ndarray
            Full frame with only this object's bounding box drawn.
        priority : float, optional
            Priority for queue ordering (lower = earlier).

        Returns True if enqueued, False if track already in-flight or cached.
        """
        # Check if already cached or in-flight
        with self._ifl_lock:
            if self._cache.get(track_id) is not None:
                return False
            if track_id in self._inflight:
                return False
            self._inflight.add(track_id)

        item = PendingItem(
            priority=priority if priority is not None else time.monotonic(),
            track_id=track_id,
            frame=frame_with_box.copy(),
        )
        self._q.put(item)

        self._set_prog(track_id, "queued", 0.0)

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
            priority=float("inf"), track_id=-1,
            frame=np.zeros((1, 1, 3), np.uint8),
        ))
        self._thread.join(timeout=3.0)

    def _dispatch_loop(self) -> None:
        """Process items from the queue and send to remote server."""
        while self._alive:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                self._cache.evict_expired()
                continue

            if item.track_id < 0:  # stop sentinel
                break

            # Skip if already identified while waiting
            if self._cache.get(item.track_id) is not None:
                with self._ifl_lock:
                    self._inflight.discard(item.track_id)
                continue

            self._fire(item)

    def _fire(self, item: PendingItem) -> None:
        """Execute API call for a single object."""
        track_id = item.track_id
        self._set_prog(track_id, "identifying", 0.5)

        try:
            self.stats["requests"] += 1
            label = self._client.identify_frame(item.frame, track_id)

            self._cache.set(track_id, label)
            self._set_prog(track_id, "done", 1.0, label=label)
            self.stats["identified"] += 1
            print(f"[IDService] ✓ #{track_id} → \"{label}\"")

        except Exception as exc:
            self.stats["errors"] += 1
            msg = _trunc(str(exc), 80)
            print(f"[IDService] ✗ Error: {msg}")
            self._cache.set(track_id, "object")
            self._set_prog(track_id, "error", 1.0, label="object", error=msg)

        finally:
            with self._ifl_lock:
                self._inflight.discard(track_id)

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
