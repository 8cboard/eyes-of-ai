"""
id_service.py — OpenRouter Vision-LLM Identification Service (Nemotron Nano 12B VL)

Backend: OpenRouter free tier → nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free
         (OpenAI-compatible API, no billing required — free model, free account)

Obtain a free API key at: https://openrouter.ai  (no credit card needed for :free models)
Set via:  export OPENROUTER_API_KEY=sk-or-v1-...
  or pass --openrouter-key on the CLI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RATE LIMIT STRATEGY  (the hard problem with free endpoints)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OpenRouter free models allow ~20 requests/minute (RPM).
A naive 1-request-per-object strategy with 5+ objects on screen
would exhaust that in seconds.  Three-layer defence:

  Layer 1 — Batch dispatch (multiplies throughput × N per request)
    ┌─────────────────────────────────────────────────────────┐
    │  Up to BATCH_SIZE (default 4) crops are packed into a   │
    │  SINGLE OpenRouter call with N images in one message.   │
    │  The model returns one description per image.           │
    │  → 1 API hit identifies up to 4 objects.                │
    │  At 13 RPM × batch 4 = ~52 effective IDs/min.           │
    └─────────────────────────────────────────────────────────┘

  Layer 2 — Token-bucket rate limiter
    A background thread owns the HTTP client exclusively.
    Before every request it acquires a token from a bucket
    refilling at REFILL_RATE (default 0.22 tok/s = 13.2 RPM,
    safely under the 20 RPM limit).  Burst capacity = 3 tokens.

  Layer 3 — Adaptive 429 back-off + re-queue
    HTTP 429 → sleep Retry-After seconds, return items to the
    front of the priority queue, and retry.  No items are lost.

  Plus — TTL cache + stillness gating (enforced in main.py)
    Objects only submitted when still AND cache expired,
    so static background objects are never re-submitted.

Progress dict format (shared with ui_overlay.py):
  {
    track_id (int): {
        "status":   "idle" | "queued" | "identifying" | "done" | "error",
        "progress": float 0.0 – 1.0,
        "label":    str | None,
        "error":    str | None,
    }
  }

Privacy:
  Crops sent as base64 JPEG to https://openrouter.ai/api/v1.
  Do not use with sensitive footage.
  OpenRouter privacy policy: https://openrouter.ai/privacy
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import re
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants (all overridable in IdentificationService.__init__)
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free Nemotron Nano 12B Vision-Language model on OpenRouter.
# Verify / update at: https://openrouter.ai/models?q=nemotron
NEMOTRON_MODEL     = "nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free"

# Rate limiting — stay safely under 20 RPM free limit
REFILL_RATE        = 0.22    # tokens/second  →  13.2 requests/minute
BUCKET_CAPACITY    = 3       # max burst (tokens)

# Batching
BATCH_SIZE         = 4       # max crops per single API call
BATCH_WAIT_MS      = 1_500   # ms to wait collecting a fuller batch before firing

# HTTP
REQUEST_TIMEOUT_S  = 20      # seconds per API call

# Image encoding
JPEG_QUALITY       = 72      # lower = smaller payload = faster
MAX_CROP_PX        = 336     # longest-edge resize before encoding (model recommended)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class PendingItem:
    """One crop awaiting identification.  Sortable by priority for the queue."""
    priority:     float       = field()          # lower = served sooner (FIFO default)
    track_id:     int         = field(compare=False)
    crop:         np.ndarray  = field(compare=False)
    class_name:   str         = field(compare=False)
    class_id:     int         = field(compare=False)
    submitted_at: float       = field(compare=False, default_factory=time.monotonic)


class RateLimitError(Exception):
    """Raised when OpenRouter returns HTTP 429."""
    def __init__(self, retry_after: float = 30.0) -> None:
        super().__init__(f"Rate limited — retry after {retry_after:.0f}s")
        self.retry_after = retry_after


# ─────────────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class TokenBucket:
    """Thread-safe leaky/token bucket for HTTP rate limiting.

    Tokens refill continuously at ``rate`` tokens/second up to ``capacity``.
    ``acquire()`` blocks until a token is available.

    Parameters
    ----------
    rate : float
        Refill rate in tokens per second.
    capacity : float
        Maximum bucket size (burst limit).
    """

    def __init__(self, rate: float = REFILL_RATE, capacity: float = BUCKET_CAPACITY) -> None:
        self._rate        = rate
        self._capacity    = float(capacity)
        self._tokens      = float(capacity)
        self._last_refill = time.monotonic()
        self._lock        = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        """Block until n tokens are available, then consume them."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
            # Sleep until approximately one token is ready
            wait = (n - self._tokens) / self._rate
            time.sleep(min(wait, 0.5))

    def peek(self) -> float:
        """Return current token count (approximate, for display)."""
        with self._lock:
            self._refill()
            return round(self._tokens, 2)

    def _refill(self) -> None:
        """Add tokens for elapsed time (must be called under self._lock)."""
        now    = time.monotonic()
        added  = (now - self._last_refill) * self._rate
        self._tokens      = min(self._capacity, self._tokens + added)
        self._last_refill = now


# ─────────────────────────────────────────────────────────────────────────────
# Per-track TTL cache
# ─────────────────────────────────────────────────────────────────────────────

class IdentificationCache:
    """Thread-safe per-track label cache with configurable TTL.

    Parameters
    ----------
    ttl_seconds : float
        How long an identified label stays valid.  After expiry the track
        becomes eligible for re-identification if it becomes still again.
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
# OpenRouter HTTP client
# ─────────────────────────────────────────────────────────────────────────────

class OpenRouterClient:
    """Thin OpenAI-compatible client targeting OpenRouter/Nemotron vision.

    Supports multi-image batching: up to BATCH_SIZE base64 crops in one call.

    Free key signup: https://openrouter.ai  (no credit card for :free models)

    Example CLI equivalent (single crop, no auth required for free models):
        curl -X POST https://openrouter.ai/api/v1/chat/completions \\
          -H "Authorization: Bearer $OPENROUTER_API_KEY" \\
          -H "Content-Type: application/json" \\
          -d '{
            "model": "nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free",
            "messages": [{
              "role": "user",
              "content": [
                {"type":"text","text":"Describe this object in one sentence."},
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<B64>"}}
              ]
            }],
            "max_tokens": 80
          }'
    """

    def __init__(
        self,
        api_key:  str,
        model:    str   = NEMOTRON_MODEL,
        timeout:  float = REQUEST_TIMEOUT_S,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required (it's free).\n"
                "  1. Go to https://openrouter.ai and create a free account.\n"
                "  2. Generate a key (no credit card needed for :free models).\n"
                "  3. export OPENROUTER_API_KEY=sk-or-v1-...\n"
                "     or use --openrouter-key on the CLI."
            )
        if not _REQ_OK:
            raise ImportError("pip install requests")

        self._model   = model
        self._timeout = timeout
        self._session = _req.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/visiontracker",   # polite routing hint
            "X-Title":       "VisionTracker",
        })

    def identify_batch(self, items: list[PendingItem]) -> dict[int, str]:
        """Identify up to BATCH_SIZE objects in a single API call.

        Packs all crops as image_url content blocks in one user message.
        Asks the model to respond with a numbered list (one line per image).

        Returns
        -------
        dict[int, str]
            {track_id: description_string} for every item.
            Falls back to item.class_name on any parse failure.
        """
        n       = len(items)
        content = _build_message_content(items)

        payload = {
            "model":       self._model,
            "messages":    [{"role": "user", "content": content}],
            "max_tokens":  100 * n,   # ~10 words × ~10 tokens × n objects
            "temperature": 0.2,       # low = consistent, factual descriptions
        }

        resp = self._session.post(
            OPENROUTER_API_URL,
            data=json.dumps(payload),
            timeout=self._timeout,
        )

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 30))
            raise RateLimitError(retry_after)

        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        log.debug("Raw model output: %r", raw)
        return _parse_batch_response(raw, items)


# ─────────────────────────────────────────────────────────────────────────────
# Main service — single dispatcher, batching, rate limiting
# ─────────────────────────────────────────────────────────────────────────────

class IdentificationService:
    """Non-blocking multi-object identification using OpenRouter Nemotron VL.

    One background thread (*dispatcher*) owns all HTTP traffic.
    The main thread calls ``submit()`` which puts items into a
    ``PriorityQueue``.  The dispatcher assembles batches, throttles via
    ``TokenBucket``, fires batched API calls, and writes results into
    the shared ``progress`` dict.

    Parameters
    ----------
    api_key : str
        OpenRouter API key.  Falls back to OPENROUTER_API_KEY env var.
    model : str
        Model slug on OpenRouter.
    cache_ttl : float
        Seconds a cached result stays valid.
    batch_size : int
        Max crops per API call (1–4 recommended; larger = more latency).
    batch_wait_ms : int
        How long (ms) the dispatcher waits for a fuller batch.
        Higher = fewer requests but slightly more ID latency.
        For ≤10 s total ID time: keep at 1500–2000 ms.
    """

    def __init__(
        self,
        api_key:      str   = "",
        model:        str   = NEMOTRON_MODEL,
        cache_ttl:    float = 45.0,
        batch_size:   int   = BATCH_SIZE,
        batch_wait_ms: int  = BATCH_WAIT_MS,
    ) -> None:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client     = OpenRouterClient(api_key=api_key, model=model)
        self._bucket     = TokenBucket(rate=REFILL_RATE, capacity=BUCKET_CAPACITY)
        self._cache      = IdentificationCache(ttl_seconds=cache_ttl)
        self._batch_sz   = max(1, min(batch_size, 8))
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
        self.stats = dict(requests=0, identified=0, rate_limit_hits=0, errors=0)

        # Start the single dispatcher thread
        self._alive = True
        self._thread = threading.Thread(
            target=self._dispatch_loop, name="id-dispatcher", daemon=True
        )
        self._thread.start()

        eff = REFILL_RATE * self._batch_sz * 60
        print(f"[IDService] OpenRouter / {model}")
        print(f"[IDService] Rate limiter: {REFILL_RATE:.2f} tok/s "
              f"(~{REFILL_RATE*60:.0f} RPM) × batch {self._batch_sz} "
              f"= ~{eff:.0f} effective IDs/min")
        print(f"[IDService] Cache TTL: {cache_ttl}s | Batch wait: {batch_wait_ms}ms")

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

    def bucket_tokens(self) -> float:
        """Current token bucket level (0.0 – BUCKET_CAPACITY). For HUD display."""
        return self._bucket.peek()

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
        """Collect batches from the queue and send to OpenRouter."""
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

            # ── Rate-limit: block until token available ───────────────────
            log.debug("Bucket level: %.2f — waiting for token…", self._bucket.peek())
            self._bucket.acquire()            # may block up to a few seconds

            # ── Fire batch ────────────────────────────────────────────────
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
                print(f"[IDService] ✓ #{item.track_id} ({item.class_name}) "
                      f"→ \"{_trunc(label, 60)}\"")

        except RateLimitError as exc:
            self.stats["rate_limit_hits"] += 1
            wait = exc.retry_after
            print(f"[IDService] ⚠️  Rate limit — sleeping {wait:.0f}s, re-queuing {len(batch)} items")
            for item in batch:
                self._set_prog(item.track_id, "queued", 0.05,
                               label=f"Rate limited — retrying in {wait:.0f}s…")
            time.sleep(wait)
            # Re-queue at front (negative priority = highest priority)
            for item in batch:
                item.priority = time.monotonic() - 1_000
                self._q.put(item)
            return  # keep inflight — items re-queued

        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            msg = _trunc(str(exc), 80)
            print(f"[IDService] ✗ Error: {msg}")
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
# Message construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_message_content(items: list[PendingItem]) -> list[dict]:
    """Build the OpenAI-compatible multimodal content list for a batch."""
    n = len(items)
    if n == 1:
        prompt = (
            f"The object detector classified this crop as '{items[0].class_name}'. "
            "Describe it in ONE concise sentence (≤ 12 words): colour, shape, key features."
        )
    else:
        numbered = "\n".join(f"{i+1}. <description>" for i in range(n))
        prompt = (
            f"I will show you {n} object crops from a video frame. "
            "The detector classes are shown before each image.\n"
            "Write ONE short sentence per object (≤ 12 words: colour, shape, features).\n"
            f"Reply ONLY in this exact format:\n{numbered}"
        )

    content: list[dict] = [{"type": "text", "text": prompt}]

    for i, item in enumerate(items):
        if n > 1:
            content.append({
                "type": "text",
                "text": f"Image {i + 1} — detected class: {item.class_name}",
            })
        b64 = _encode_crop(item.crop)
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    return content


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


def _parse_batch_response(text: str, items: list[PendingItem]) -> dict[int, str]:
    """Parse model's numbered-list reply into {track_id: description}.

    Expected format (n > 1):
        1. A red ceramic mug with a curved handle
        2. A silver MacBook with stickers on the lid
        ...

    Falls back gracefully if the model doesn't follow the format exactly.
    """
    n = len(items)
    results: dict[int, str] = {}

    if n == 1:
        clean = re.sub(r"^1[.)]\s*", "", text.strip()).strip()
        results[items[0].track_id] = clean or items[0].class_name
        return results

    # Try numbered list pattern
    numbered: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)[.)]\s*(.+)$")
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                numbered[idx] = m.group(2).strip()

    for i, item in enumerate(items):
        if i in numbered:
            results[item.track_id] = numbered[i]
        else:
            # Heuristic fallback: grab non-empty lines in order
            fallback_lines = [l.strip() for l in text.splitlines() if l.strip()]
            results[item.track_id] = (
                fallback_lines[i] if i < len(fallback_lines) else item.class_name
            )

    return results


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"
