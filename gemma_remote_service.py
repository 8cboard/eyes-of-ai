"""
gemma_remote_service.py — Self-Hosted Gemma 3 4B HTTP Client

Backend: User-hosted Gemma 3 4B model on Google Colab or Kaggle via FastAPI

Setup:
  1. Run the Colab or Kaggle notebook from remote_server/
  2. Copy the ngrok public URL
  3. python main.py --use-remote-gemma --remote-gemma-url https://xxx.ngrok.io

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DESIGN DIFFERENCES FROM OpenRouter (id_service.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Unlike OpenRouter's free tier with strict 20 RPM limits, the self-hosted
server has no enforced rate limit — the user controls capacity via their
Colab/Kaggle instance. Key differences:

  ✗ No TokenBucket rate limiter (user controls server capacity)
  ✓ Higher default batch size (8 vs 4) — Colab/Kaggle can handle more
  ✓ Longer timeout (60s vs 20s) — local inference may be slower
  ✓ Health check endpoint — verifies server before submitting
  ✓ Retry with exponential backoff for transient failures
  ✗ No 429 handling (server doesn't rate limit)

Progress dict format (same as id_service.py for UI compatibility):
  {
    track_id (int): {
        "status":   "idle" | "queued" | "identifying" | "done" | "error",
        "progress": float 0.0 – 1.0,
        "label":    str | None,
        "error":    str | None,
    }
  }

Privacy:
  Crops sent over HTTPS to your self-hosted server.
  Images never touch third-party APIs.
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

log = logging.getLogger("gemma_remote")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants (all overridable in GemmaRemoteService.__init__)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REMOTE_URL = ""
DEFAULT_BATCH_SIZE = 8           # Higher than OpenRouter — user controls capacity
DEFAULT_BATCH_WAIT_MS = 1_000    # Slightly faster dispatch (less queuing delay)
REQUEST_TIMEOUT_S = 60           # Longer timeout for local model inference
MAX_RETRIES = 3                  # Retry failed requests with backoff
RETRY_BASE_DELAY_S = 2.0         # Base delay for exponential backoff

# Image encoding
JPEG_QUALITY = 72                # Same as id_service.py
MAX_CROP_PX = 336                # Same as id_service.py


# ─────────────────────────────────────────────────────────────────────────────
# Data structures (reused from id_service.py for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class PendingItem:
    """One crop awaiting identification. Sortable by priority for the queue."""
    priority:     float       = field()          # lower = served sooner (FIFO default)
    track_id:     int         = field(compare=False)
    crop:         np.ndarray  = field(compare=False)
    class_name:   str         = field(compare=False)
    class_id:     int         = field(compare=False)
    submitted_at: float       = field(compare=False, default_factory=time.monotonic)


class IdentificationCache:
    """Thread-safe per-track label cache with configurable TTL.

    Copied from id_service.py for self-containment.

    Parameters
    ----------
    ttl_seconds : float
        How long an identified label stays valid.
    """

    def __init__(self, ttl_seconds: float = 45.0) -> None:
        self._ttl   = ttl_seconds
        self._lock  = threading.Lock()
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

class GemmaHTTPClient:
    """HTTP client for the self-hosted Gemma remote server.

    Communicates with a FastAPI server running on Colab/Kaggle via ngrok tunnel.
    Supports batch identification of multiple objects in a single request.

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
        api_key:  Optional[str] = None,
        timeout:  float = REQUEST_TIMEOUT_S,
    ) -> None:
        if not base_url:
            raise ValueError(
                "Remote Gemma server URL is required.\n"
                "  1. Run the Colab notebook: remote_server/colab_setup.ipynb\n"
                "  2. Copy the public URL (e.g., https://xxx.ngrok.io)\n"
                "  3. python main.py --use-remote-gemma --remote-gemma-url https://xxx.ngrok.io"
            )
        if not _REQ_OK:
            raise ImportError("pip install requests")

        self._base_url = base_url.rstrip("/")
        self._api_key  = api_key
        self._timeout  = timeout
        self._session  = _req.Session()

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
            print(f"[GemmaRemote] Connected to server: {data.get('model', 'unknown')}")
        except Exception as exc:
            print(f"[GemmaRemote] ⚠️  Health check failed: {exc}")
            print(f"[GemmaRemote] Server may still be starting up...")

    def _headers(self) -> dict[str, str]:
        """Build request headers with optional auth."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def identify_batch(self, items: list[PendingItem]) -> dict[int, str]:
        """Identify a batch of objects via the remote server.

        Parameters
        ----------
        items : list[PendingItem]
            List of crops to identify

        Returns
        -------
        dict[int, str]
            {track_id: description_string} for each item

        Raises
        ------
        requests.RequestException
            If the request fails after retries
        """
        n = len(items)

        # Encode images
        images_b64 = [_encode_crop(item.crop) for item in items]
        class_names = [item.class_name for item in items]
        track_ids = [item.track_id for item in items]

        payload = {
            "images": images_b64,
            "class_names": class_names,
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
                results = {}
                for result in data.get("results", []):
                    tid = result.get("track_id")
                    desc = result.get("description", "")
                    if tid is not None:
                        results[tid] = desc

                # Fill in any missing results with class_name
                for item in items:
                    if item.track_id not in results:
                        results[item.track_id] = item.class_name

                return results

            except _req.exceptions.RequestException as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY_S * (2 ** attempt)  # Exponential backoff
                    print(f"[GemmaRemote] Request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    break

        # All retries exhausted
        raise last_exception or RuntimeError("Unknown error in identify_batch")


# ─────────────────────────────────────────────────────────────────────────────
# Main service — single dispatcher, batching, no rate limiting
# ─────────────────────────────────────────────────────────────────────────────

class GemmaRemoteService:
    """Non-blocking multi-object identification using self-hosted Gemma 3 4B.

    One background thread (*dispatcher*) owns all HTTP traffic.
    The main thread calls ``submit()`` which puts items into a
    ``PriorityQueue``.  The dispatcher assembles batches and fires
    HTTP requests to the remote server.  No rate limiting — the user
    controls server capacity via their Colab/Kaggle instance.

    Parameters
    ----------
    remote_url : str
        The public URL of the remote server (ngrok tunnel)
    api_key : str, optional
        Optional API key for server authentication
    cache_ttl : float
        Seconds a cached result stays valid.
    batch_size : int
        Max crops per API call (1–16 recommended; larger = more latency)
    batch_wait_ms : int
        How long (ms) the dispatcher waits for a fuller batch.
        Higher = fewer requests but slightly more ID latency.
    """

    def __init__(
        self,
        remote_url:    str,
        api_key:       Optional[str] = None,
        cache_ttl:     float = 45.0,
        batch_size:    int   = DEFAULT_BATCH_SIZE,
        batch_wait_ms: int   = DEFAULT_BATCH_WAIT_MS,
    ) -> None:
        self._client     = GemmaHTTPClient(base_url=remote_url, api_key=api_key)
        self._cache      = IdentificationCache(ttl_seconds=cache_ttl)
        self._batch_sz   = max(1, min(batch_size, 16))  # Cap at 16 for safety
        self._batch_wait = batch_wait_ms / 1_000.0  # ms → s

        # Shared progress dict — read by ui_overlay at frame rate
        self.progress: dict[int, dict] = {}
        self._prog_lock  = threading.Lock()

        # Priority queue (smallest priority value dispatched first)
        self._q: queue.PriorityQueue[PendingItem] = queue.PriorityQueue()

        # Track which track_ids are currently queued or in-flight
        self._inflight: set[int] = set()
        self._ifl_lock  = threading.Lock()

        # Counters for debugging / UI
        self.stats = dict(requests=0, identified=0, errors=0, retries=0)

        # Start the single dispatcher thread
        self._alive = True
        self._thread = threading.Thread(
            target=self._dispatch_loop, name="gemma-remote-dispatcher", daemon=True
        )
        self._thread.start()

        print(f"[GemmaRemote] Self-hosted Gemma 3 4B")
        print(f"[GemmaRemote] Server: {remote_url}")
        print(f"[GemmaRemote] Batch size: {self._batch_sz} | Batch wait: {batch_wait_ms}ms")
        print(f"[GemmaRemote] Cache TTL: {cache_ttl}s")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_cached(self, track_id: int) -> Optional[str]:
        """Return cached label or None if not yet identified / expired."""
        return self._cache.get(track_id)

    def submit(
        self,
        track_id:   int,
        crop:       np.ndarray,
        class_name: str,
        class_id:   int,
        priority:   Optional[float] = None,
    ) -> bool:
        """Enqueue *crop* for background identification.

        Returns True if enqueued, False if already in-flight or cached.
        Pass a lower *priority* value to jump ahead in the queue.
        """
        if self._cache.get(track_id) is not None:
            return False
        with self._ifl_lock:
            if track_id in self._inflight:
                return False
            self._inflight.add(track_id)

        item = PendingItem(
            priority   = priority if priority is not None else time.monotonic(),
            track_id   = track_id,
            crop       = crop.copy(),
            class_name = class_name,
            class_id   = class_id,
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
        self._q.put(PendingItem(
            priority=float("inf"), track_id=-1,
            crop=np.zeros((1, 1, 3), np.uint8),
            class_name="__stop__", class_id=-1,
        ))
        self._thread.join(timeout=3.0)

    # ── Dispatcher (background thread — single, serialised) ────────────────────

    def _dispatch_loop(self) -> None:
        """Collect batches from the queue and send to remote server."""
        while self._alive:
            # Block until at least one item arrives
            try:
                first = self._q.get(timeout=1.0)
            except queue.Empty:
                self._cache.evict_expired()   # housekeeping while idle
                continue

            if first.track_id == -1:          # stop sentinel
                break

            batch: list[PendingItem] = [first]

            # ── Fill batch (wait up to batch_wait for more items) ─────────
            deadline = time.monotonic() + self._batch_wait
            while len(batch) < self._batch_sz:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    item = self._q.get(timeout=left)
                except queue.Empty:
                    break

                if item.track_id == -1:
                    self._alive = False
                    break

                # Skip if already identified while we were waiting
                if self._cache.get(item.track_id) is not None:
                    self._set_prog(item.track_id, "done", 1.0,
                                   label=self._cache.get(item.track_id))
                    with self._ifl_lock:
                        self._inflight.discard(item.track_id)
                    continue

                batch.append(item)

            if not batch:
                continue

            for item in batch:
                self._set_prog(item.track_id, "identifying", 0.2)

            # ── Fire batch (no rate limiting — user controls server) ───────
            self._fire(batch)

    def _fire(self, batch: list[PendingItem]) -> None:
        """Execute one batched API call and distribute results."""
        for item in batch:
            self._set_prog(item.track_id, "identifying", 0.5)

        try:
            self.stats["requests"] += 1
            results = self._client.identify_batch(batch)

            for item in batch:
                label = results.get(item.track_id, item.class_name)
                self._cache.set(item.track_id, label)
                self._set_prog(item.track_id, "done", 1.0, label=label)
                self.stats["identified"] += 1
                print(f"[GemmaRemote] ✓ #{item.track_id} ({item.class_name}) "
                      f"→ \"{_trunc(label, 60)}\"")

        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            msg = _trunc(str(exc), 80)
            print(f"[GemmaRemote] ✗ Error: {msg}")
            for item in batch:
                self._cache.set(item.track_id, item.class_name)
                self._set_prog(item.track_id, "error", 1.0,
                               label=item.class_name, error=msg)

        finally:
            with self._ifl_lock:
                for item in batch:
                    self._inflight.discard(item.track_id)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _set_prog(
        self,
        track_id: int,
        status:   str,
        progress: float,
        label:    Optional[str] = None,
        error:    Optional[str] = None,
    ) -> None:
        with self._prog_lock:
            prev = self.progress.get(track_id, {})
            self.progress[track_id] = {
                "status":   status,
                "progress": progress,
                "label":    label if label is not None else prev.get("label"),
                "error":    error,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (same as id_service.py)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_crop(
    crop: np.ndarray,
    max_px:  int = MAX_CROP_PX,
    quality: int = JPEG_QUALITY,
) -> str:
    """Resize and JPEG-encode a crop, return base64 string."""
    h, w = crop.shape[:2]
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        crop = cv2.resize(
            crop,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"
