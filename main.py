"""
main.py — VisionTracker entry point.

Orchestrates the full pipeline:
  Camera → EdgeDetector → Tracker → IDService → UIOverlay

For each tracked object, sends the full frame with ONLY that object's box
drawn to the remote LLM server for identification.

Run:
  python main.py --remote-url https://xxx.ngrok.io
  python main.py --remote-url https://xxx.ngrok.io --use-bg-removal
  python main.py  # (no URL — runs without identification)

See README.md for full flag documentation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VisionTracker — real-time edge detection, tracking, and LLM identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    p.add_argument("--input",  default="0",
                   help="Camera index (0) or path to video file")
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--grayscale", action="store_true",
                   help="Force grayscale (faster on CPU)")

    # Edge Detector
    p.add_argument("--edge-min-area",   type=int,   default=500)
    p.add_argument("--edge-max-area",   type=int,   default=100_000)
    p.add_argument("--canny-low",       type=int,   default=50,
                   help="Canny lower threshold (used when --no-auto-canny)")
    p.add_argument("--canny-high",      type=int,   default=150,
                   help="Canny upper threshold (used when --no-auto-canny)")
    p.add_argument("--no-auto-canny",   action="store_true",
                   help="Disable adaptive Canny thresholds and use --canny-low/--canny-high")
    p.add_argument("--merge-distance",  type=int,   default=30,
                   help="Pixel distance for proximity-based bounding-box merging "
                        "(0 = disable; higher = more aggressive merging)")
    p.add_argument("--close-kernel",    type=int,   default=15,
                   help="Morphological close kernel size (larger bridges bigger gaps)")
    p.add_argument("--use-bg-removal",  action="store_true",
                   help="Enable background removal before edge detection (requires rembg)")
    p.add_argument("--skip-frames",     type=int,   default=1,
                   help="Run detector every N frames (1=every frame)")

    # Tracker
    p.add_argument("--tracker", default="bytetrack",
                   choices=["bytetrack", "centroid"])

    # ID Service
    p.add_argument("--remote-url", default=None,
                   help="Remote LLM server URL (e.g., https://xxx.ngrok.io). "
                        "If omitted, identification is disabled.")
    p.add_argument("--remote-key", default=None,
                   help="API key for remote server (optional)")
    p.add_argument("--id-ttl",      type=float, default=45.0)
    p.add_argument("--id-interval", type=float, default=5.0,
                   help="Seconds between identification attempts per track")
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--batch-wait",  type=int,   default=1000)

    # UI
    p.add_argument("--show-velocity", action="store_true")
    p.add_argument("--no-display",    action="store_true",
                   help="Headless mode — no OpenCV window")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# FPS counter
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Exponential moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._fps: float = 0.0
        self._last_t: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            inst = 1.0 / dt if dt > 0 else 0.0
            self._fps = inst if self._fps == 0.0 else \
                self._alpha * inst + (1 - self._alpha) * self._fps
        self._last_t = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ─────────────────────────────────────────────────────────────────────────────
# Null ID service (graceful no-op when no remote URL is configured)
# ─────────────────────────────────────────────────────────────────────────────

class NullIdentificationService:
    """Drop-in replacement for IdentificationService when no URL is set.

    All methods are no-ops or return harmless defaults so the rest of the
    pipeline runs normally — just without LLM identification.
    """

    def __init__(self) -> None:
        self.progress: dict[int, dict] = {}
        self.stats = dict(requests=0, identified=0, errors=0)
        print("[IDService] No remote URL configured — identification disabled.")
        print("[IDService] Pass --remote-url https://... to enable.")

    def get_cached(self, track_id: int) -> Optional[str]:
        return None

    def submit(self, track_id: int, frame_with_box: np.ndarray, **_) -> bool:
        return False

    def queue_depth(self) -> int:
        return 0

    def inject_cached(self, track_id: int) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Single-box frame annotator
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame_with_single_box(
    frame: np.ndarray,
    xyxy: np.ndarray,
    box_color: tuple[int, int, int] = (0, 255, 0),
    box_thickness: int = 3,
) -> np.ndarray:
    """Return a copy of *frame* with a single bounding box drawn on it.

    The server prompt says 'green bounding box', so the default colour must
    stay green (0, 255, 0).  Do not change this without updating the prompt.
    """
    out = frame.copy()
    x1, y1, x2, y2 = xyxy.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, box_thickness)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    from edge_detector import EdgeDetector
    from tracker import build_tracker
    from ui_overlay import UIOverlay

    # ── Resolve remote URL ────────────────────────────────────────────────────
    remote_url = args.remote_url or os.environ.get("REMOTE_URL", "").strip()

    # ── Build ID service ──────────────────────────────────────────────────────
    if remote_url:
        from id_service import IdentificationService
        id_service = IdentificationService(
            remote_url=remote_url,
            api_key=args.remote_key or os.environ.get("REMOTE_API_KEY"),
            cache_ttl=args.id_ttl,
            batch_size=args.batch_size,
            batch_wait_ms=args.batch_wait,
        )
    else:
        id_service = NullIdentificationService()

    # ── Build pipeline components ─────────────────────────────────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  VisionTracker  —  starting")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    detector = EdgeDetector(
        min_area=args.edge_min_area,
        max_area=args.edge_max_area,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        auto_canny=not args.no_auto_canny,
        merge_distance=args.merge_distance,
        close_kernel_size=args.close_kernel,
        use_bg_removal=args.use_bg_removal,
        skip_frames=args.skip_frames,
    )

    tracker    = build_tracker(args.tracker)
    overlay    = UIOverlay(show_velocity=args.show_velocity)
    fps_ctr    = FPSCounter()

    # ── Open capture ──────────────────────────────────────────────────────────
    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Main] ERROR: Cannot open input: {args.input}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[Main] Capture: {int(cap.get(3))}×{int(cap.get(4))}")
    print(f"[Main] Tracker: {tracker.name}")
    print(f"[Main] ID interval: {args.id_interval}s")
    if not args.no_display:
        print("[Main] Press Q to quit.\n")

    active_ids:   set[int]         = set()
    last_id_time: dict[int, float] = {}

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] End of stream.")
                break

            display = frame
            if args.grayscale:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

            # Detect + track
            det_result = detector.detect(display)
            tracked    = tracker.update(det_result)
            current    = {o.track_id for o in tracked}

            # Cleanup stale state
            for tid in active_ids - current:
                last_id_time.pop(tid, None)
            active_ids = current

            # Enqueue new identifications
            now = time.monotonic()
            for obj in tracked:
                tid = obj.track_id
                last_t = last_id_time.get(tid, 0.0)
                if (now - last_t > args.id_interval
                        and id_service.get_cached(tid) is None):
                    annotated = draw_frame_with_single_box(display, obj.xyxy)
                    if id_service.submit(track_id=tid, frame_with_box=annotated):
                        last_id_time[tid] = now
                        print(f"[Main] → queued #{tid} | q={id_service.queue_depth()}")

            # Push cached labels into overlay progress
            for obj in tracked:
                id_service.inject_cached(obj.track_id)

            # Render
            fps        = fps_ctr.tick()
            backend_s  = f"remote|q:{id_service.queue_depth()}" if remote_url else "disabled"
            annotated_frame = overlay.draw(
                frame=display,
                tracked_objects=tracked,
                progress_dict=id_service.progress,
                fps=fps,
                mode="edge-llm",
                backend=backend_s,
            )

            if not args.no_display:
                cv2.imshow("VisionTracker", annotated_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    print("[Main] Quit requested.")
                    break

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        print("[Main] Cleaning up…")
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        id_service.shutdown()
        print("[Main] Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
