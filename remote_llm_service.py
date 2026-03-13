"""
remote_llm_service.py — Remote LLM Identification Service

Sends full annotated frames (with colored bounding boxes) to a remote
Kaggle/Colab server running a vision-language model for identification.

The remote server supports models <10GB in either:
  - GGUF format (via llama-cpp-python)
  - safetensors format (via transformers)

Unlike the old architecture that sent individual crops, this service sends
the full annotated image and receives descriptions for all tracked objects.

Usage:
    service = RemoteLLMService(remote_url="https://xxx.ngrok.io")
    service.submit(track_id, annotated_frame, track_ids_in_frame)
    label = service.get_cached(track_id)
"""

from __future__ import annotations

import base64
import io
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

log = logging.getLogger("remote_llm")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 1           # One annotated frame per request
DEFAULT_BATCH_WAIT_MS = 500      # Short wait — frames come continuously
REQUEST_TIMEOUT_S = 60           # Longer timeout for local model inference
MAX_RETRIES = 3                  # Retry failed requests with backoff
RETRY_BASE_DELAY_S = 2.0         # Base delay for exponential backoff

# Image encoding
JPEG_QUALITY = 75                # Balance quality vs bandwidth
MAX_FRAME_PX = 1280              # Resize if frame is larger


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class PendingFrame:
    """One annotated frame awaiting identification."""
    priority:     float       = field()          # lower = served sooner
    track_id:     int         = field(compare=False)  # Primary track being focused
    frame:        np.ndarray  = field(compare=False)  # Annotated frame (with boxes)
    track_ids:    list[int]   = field(compare=False)  # All track IDs in frame
    submitted_at: float       = field(compare=False, default_factory=time.monotonic)


class IdentificationCache:
    """Thread-safe per-track label cache with configurable TTL.

    Parameters
    ----------
    ttl_seconds : float
        How long an identified label stays valid.
    """

    def __init__(self, ttl_seconds: float = 45.0) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # {track_id: (label, inserted_at)}
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


# ─────────────────────────────────────────────────────────────────────────────
# Remote HTTP client
# ─────────────────────────────────────────────────────────────────────────────

class RemoteLLMClient:
    """HTTP client for the remote LLM server.

    Communicates with a FastAPI server running on Colab/Kaggle via ngrok tunnel.
    Sends full annotated frames and receives identification results.

    Parameters
    ----------
    base_url : str
        The public URL of the ngrok tunnel (e.g., https://abc123.ngrok.io)
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
                "Remote LLM server URL is required.\n"
                "  1. Run the Colab/Kaggle notebook from remote_server/\n"
                "  2. Copy the public URL (e.g., https://xxx.ngrok.io)\n"
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
            print(f"[RemoteLLM] Connected to server: {data.get('model', 'unknown')}")
        except Exception as exc:
            print(f"[RemoteLLM] ⚠️  Health check failed: {exc}")
            print(f"[RemoteLLM] Server may still be starting up...")

    def _headers(self) -> dict[str, str]:
        """Build request headers with optional auth."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def identify_frame(self, frame_b64: str, track_ids: list[int]) -> dict[int, str]:
        """Send an annotated frame to the remote server for identification.

        Parameters
        ----------
        frame_b64 : str
            Base64-encoded JPEG of the annotated frame
        track_ids : list[int]
            List of track IDs visible in the frame

        Returns
        -------
        dict[int, str]
            {track_id: description_string} for each track

        Raises
        ------
        requests.RequestException
            If the request fails after retries
        """
        payload = {
            "annotated_image": frame_b64,
            "track_ids": track_ids,
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
                results: dict[int, str] = {}

                # Parse results from server
                for result in data.get("results", []):
                    tid = result.get("track_id")
                    desc = result.get("description", "")
                    if tid is not None:
                        results[tid] = desc

                # Fill in any missing results
                for tid in track_ids:
                    if tid not in results:
                        results[tid] = f"object #{tid}"

                return results

            except _req.exceptions.RequestException as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                    print(f"[RemoteLLM] Request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    break

        # All retries exhausted
        raise last_exception or RuntimeError("Unknown error in identify_frame")


# ─────────────────────────────────────────────────────────────────────────────
# Main service — background dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class RemoteLLMService:
    """Non-blocking identification using remote LLM server.

    One background thread (*dispatcher*) owns all HTTP traffic.
    The main thread calls ``submit()`` which puts frames into a
    ``PriorityQueue``. The dispatcher sends annotated frames to the
    remote server and writes results into the shared ``progress`` dict.

    Parameters
    ----------
    remote_url : str
        The public URL of the remote server (ngrok tunnel)
    api_key : str, optional
        Optional API key for server authentication
    cache_ttl : float
        Seconds a cached result stays valid.
    batch_size : int
        Always 1 in this architecture (one frame per request)
    batch_wait_ms : int
        How long (ms) the dispatcher waits before sending a frame.
    """

    def __init__(
        self,
        remote_url: str,
        api_key: Optional[str] = None,
        cache_ttl: float = 45.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_wait_ms: int = DEFAULT_BATCH_WAIT_MS,
    ) -> None:
        self._client = RemoteLLMClient(base_url=remote_url, api_key=api_key)
        self._cache = IdentificationCache(ttl_seconds=cache_ttl)
        self._batch_sz = 1  # Always 1 frame per request
        self._batch_wait = batch_wait_ms / 1000.0

        # Shared progress dict — read by ui_overlay at frame rate
        self.progress: dict[int, dict] = {}
        self._prog_lock = threading.Lock()

        # Priority queue
        self._q: queue.PriorityQueue[PendingFrame] = queue.PriorityQueue()

        # Track which track_ids are currently queued or in-flight
        self._inflight: set[int] = set()
        self._ifl_lock = threading.Lock()

        # Counters for debugging / UI
        self.stats = dict(requests=0, identified=0, errors=0, retries=0)

        # Start the dispatcher thread
        self._alive = True
        self._thread = threading.Thread(
            target=self._dispatch_loop, name="remote-llm-dispatcher", daemon=True
        )
        self._thread.start()

        print(f"[RemoteLLM] Remote LLM identification service")
        print(f"[RemoteLLM] Server: {remote_url}")
        print(f"[RemoteLLM] Cache TTL: {cache_ttl}s")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_cached(self, track_id: int) -> Optional[str]:
        """Return cached label or None if not yet identified / expired."""
        return self._cache.get(track_id)

    def submit(
        self,
        track_id: int,
        annotated_frame: np.ndarray,
        track_ids: list[int],
        priority: Optional[float] = None,
    ) -> bool:
        """Enqueue an annotated frame for background identification.

        Returns True if enqueued, False if already in-flight or cached.
        Pass a lower *priority* value to jump ahead in the queue.
        """
        if self._cache.get(track_id) is not None:
            return False
        with self._ifl_lock:
            if track_id in self._inflight:
                return False
            self._inflight.add(track_id)

        item = PendingFrame(
            priority=priority if priority is not None else time.monotonic(),
            track_id=track_id,
            frame=annotated_frame.copy(),
            track_ids=track_ids,
        )
        self._q.put(item)
        self._set_prog(track_id, "queued", 0.0)
        return True

    def queue_depth(self) -> int:
        return self._q.qsize()

    def inject_cached(self, track_id: int) -> None:
        """Push a cached label into progress dict (called from main loop)."""
        label = self._cache.get(track_id)
        if label is not None:
            self._set_prog(track_id, "done", 1.0, label=label)

    def shutdown(self) -> None:
        """Stop the dispatcher thread gracefully."""
        self._alive = False
        # Sentinel item to unblock queue.get()
        self._q.put(PendingFrame(
            priority=float("inf"),
            track_id=-1,
            frame=np.zeros((10, 10, 3), np.uint8),
            track_ids=[],
        ))
        self._thread.join(timeout=3.0)

    # ── Dispatcher (background thread) ─────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Collect frames from the queue and send to remote server."""
        while self._alive:
            # Block until at least one item arrives
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                self._cache.evict_expired()
                continue

            if item.track_id == -1:  # stop sentinel
                break

            # Skip if already identified while waiting
            if self._cache.get(item.track_id) is not None:
                self._set_prog(item.track_id, "done", 1.0,
                               label=self._cache.get(item.track_id))
                with self._ifl_lock:
                    self._inflight.discard(item.track_id)
                continue

            self._set_prog(item.track_id, "identifying", 0.2)
            self._fire(item)

    def _fire(self, item: PendingFrame) -> None:
        """Execute one API call and distribute results."""
        self._set_prog(item.track_id, "identifying", 0.5)

        try:
            # Encode frame to base64 JPEG
            frame_b64 = _encode_frame(item.frame)

            self.stats["requests"] += 1
            results = self._client.identify_frame(frame_b64, item.track_ids)

            # Update cache and progress for all tracks in the frame
            for tid in item.track_ids:
                label = results.get(tid, f"object #{tid}")
                self._cache.set(tid, label)
                self._set_prog(tid, "done", 1.0, label=label)
                self.stats["identified"] += 1
                if tid == item.track_id:
                    print(f"[RemoteLLM] ✓ #{tid} → \"{_trunc(label, 60)}\"")

        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            msg = _trunc(str(exc), 80)
            print(f"[RemoteLLM] ✗ Error: {msg}")
            # Fallback: set generic labels
            for tid in item.track_ids:
                self._cache.set(tid, f"object #{tid}")
                self._set_prog(tid, "error", 1.0,
                               label=f"object #{tid}", error=msg)

        finally:
            with self._ifl_lock:
                self._inflight.discard(item.track_id)

    # ── Internal ───────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _encode_frame(
    frame: np.ndarray,
    max_px: int = MAX_FRAME_PX,
    quality: int = JPEG_QUALITY,
) -> str:
    """Resize and JPEG-encode a frame, return base64 string.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame from OpenCV.
    max_px : int
        Maximum dimension (width or height) — larger frames are resized.
    quality : int
        JPEG quality (0-100).

    Returns
    -------
    str
        Base64-encoded JPEG data URI.
    """
    h, w = frame.shape[:2]

    # Resize if frame is too large (saves bandwidth)
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Encode to JPEG
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"
