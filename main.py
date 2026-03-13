"""
main.py — VisionTracker entry point.

Orchestrates the full pipeline:
  Camera → EdgeDetector → Tracker → For each object: Draw Frame with Single Box → IDService → UIOverlay

For each tracked object, sends the full frame with ONLY that object's bounding box drawn.
Multiple requests per frame - one per object.

Run:
  python main.py --remote-url https://xxx.ngrok.io
  python main.py --remote-url https://xxx.ngrok.io --use-bg-removal

See README.md for full flag documentation and performance notes.
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
    p.add_argument("--input", default="0",
                   help="Camera index (0) or path to video file")
    p.add_argument("--width", type=int, default=1280, help="Capture width in pixels")
    p.add_argument("--height", type=int, default=720, help="Capture height in pixels")
    p.add_argument("--grayscale", action="store_true",
                   help="Force grayscale (faster on CPU)")

    # Edge Detector
    p.add_argument("--edge-min-area", type=int, default=500,
                   help="Minimum contour area to detect as object")
    p.add_argument("--edge-max-area", type=int, default=100000,
                   help="Maximum contour area (filters huge regions)")
    p.add_argument("--canny-low", type=int, default=50,
                   help="Canny edge detector low threshold")
    p.add_argument("--canny-high", type=int, default=150,
                   help="Canny edge detector high threshold")
    p.add_argument("--use-bg-removal", action="store_true",
                   help="Use background removal before edge detection (requires rembg)")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Run detector every N frames (1=every frame)")

    # Tracker
    p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "centroid"],
                   help="Tracker type (centroid is faster on very slow CPUs)")

    # ID Service
    p.add_argument("--remote-url", default=None,
                   help="Remote LLM server URL (e.g., https://xxx.ngrok.io). "
                        "Can also be set via REMOTE_URL env var.")
    p.add_argument("--remote-key", default=None,
                   help="Optional API key for remote server. "
                        "Can also be set via REMOTE_API_KEY env var.")
    p.add_argument("--id-ttl", type=float, default=45.0,
                   help="Seconds before re-identification of a cached track")
    p.add_argument("--id-interval", type=float, default=5.0,
                   help="Seconds between identification attempts for each track")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Max tracks per API call")
    p.add_argument("--batch-wait", type=int, default=1000,
                   help="ms to wait for a fuller batch before dispatching")

    # UI
    p.add_argument("--show-velocity", action="store_true",
                   help="Show velocity indicator on bounding boxes")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress OpenCV window (useful for testing)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# FPS tracker
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Exponential moving average FPS counter."""
    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._fps: float = 0.0
        self._last_t: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            if self._fps == 0.0:
                self._fps = inst_fps
            else:
                self._fps = self._alpha * inst_fps + (1 - self._alpha) * self._fps
        self._last_t = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ─────────────────────────────────────────────────────────────────────────────
# Color palette (must match ui_overlay.py and id_service.py)
# ─────────────────────────────────────────────────────────────────────────────

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


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# Single-box frame drawer for per-object identification
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame_with_single_box(
    frame: np.ndarray,
    xyxy: np.ndarray,
    box_thickness: int = 3,
    box_color: tuple[int, int, int] = (0, 255, 0),  # Green box
) -> np.ndarray:
    """Draw a single bounding box on a copy of the frame.

    Parameters
    ----------
    frame : np.ndarray
        Original frame (BGR).
    xyxy : np.ndarray
        Bounding box [x1, y1, x2, y2].
    box_thickness : int
        Thickness of the bounding box line.
    box_color : tuple[int, int, int]
        BGR color for the box (default green).

    Returns
    -------
    np.ndarray
        Frame copy with single bounding box drawn.
    """
    annotated = frame.copy()
    x1, y1, x2, y2 = xyxy.astype(int)

    # Draw bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── Imports (deferred to allow --help without heavy deps) ────────────────
    from edge_detector import EdgeDetector
    from tracker import build_tracker
    from ui_overlay import UIOverlay
    from id_service import IdentificationService

    # ── Determine ID service backend ─────────────────────────────────────────
    remote_url = args.remote_url or os.environ.get("REMOTE_URL", "")

    if not remote_url:
        print("[Main] WARNING: No --remote-url provided. Identification will not work.")
        print("[Main] Set REMOTE_URL env var or pass --remote-url")

    id_service_kwargs = dict(
        remote_url=remote_url,
        api_key=args.remote_key or os.environ.get("REMOTE_API_KEY"),
        cache_ttl=args.id_ttl,
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait,
    )

    # ── Build components ─────────────────────────────────────────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  VisionTracker starting")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    detector = EdgeDetector(
        min_area=args.edge_min_area,
        max_area=args.edge_max_area,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        use_bg_removal=args.use_bg_removal,
        skip_frames=args.skip_frames,
    )

    tracker = build_tracker(args.tracker)

    id_service = IdentificationService(**id_service_kwargs)

    overlay = UIOverlay(show_velocity=args.show_velocity)
    fps_counter = FPSCounter()

    # ── Open camera / video ──────────────────────────────────────────────────
    input_src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(input_src)
    if not cap.isOpened():
        print(f"[Main] ERROR: Cannot open input: {args.input}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Main] Capture: {actual_w}×{actual_h}")
    print(f"[Main] Detector: Edge/Canny | skip={args.skip_frames}")
    print(f"[Main] Tracker: {tracker.name}")
    print(f"[Main] ID interval: {args.id_interval}s")
    if not args.no_display:
        print("[Main] Press Q to quit.\n")

    active_track_ids: set[int] = set()
    last_id_time: dict[int, float] = {}  # track_id -> last identification time

    # ── Main loop ────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] End of stream.")
                break

            if args.grayscale:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = frame

            # ── Detect ──────────────────────────────────────────────────────
            det_result = detector.detect(display_frame)

            # ── Track ───────────────────────────────────────────────────────
            tracked = tracker.update(det_result)
            current_ids = {obj.track_id for obj in tracked}

            # Cleanup disappeared tracks
            for tid in active_track_ids - current_ids:
                last_id_time.pop(tid, None)
            active_track_ids = current_ids

            # ── Periodic identification submission ──────────────────────────
            # For each tracked object, send a full frame with ONLY that object's box
            now = time.monotonic()
            tracks_to_id = []

            for obj in tracked:
                tid = obj.track_id
                last_time = last_id_time.get(tid, 0)

                # Check if enough time has passed and not already cached
                if (now - last_time > args.id_interval and
                        id_service.get_cached(tid) is None):
                    tracks_to_id.append(tid)
                    last_id_time[tid] = now

            # Submit one frame per object (each with only that object's box)
            for obj in tracked:
                if obj.track_id in tracks_to_id:
                    # Create frame with only this object's bounding box
                    frame_with_single_box = draw_frame_with_single_box(
                        display_frame, obj.xyxy
                    )

                    submitted = id_service.submit(
                        track_id=obj.track_id,
                        frame_with_box=frame_with_single_box,
                    )
                    if submitted:
                        q_depth = id_service.queue_depth()
                        print(f"[Main] → queued track #{obj.track_id} | queue depth: {q_depth}")

            # Inject cached labels into progress
            for obj in tracked:
                id_service.inject_cached(obj.track_id)

            # ── Render ──────────────────────────────────────────────────────
            fps = fps_counter.tick()
            backend_str = f"remote | q:{id_service.queue_depth()}"
            mode_str = "edge-llm"

            annotated = overlay.draw(
                frame=display_frame,
                tracked_objects=tracked,
                progress_dict=id_service.progress,
                fps=fps,
                mode=mode_str,
                backend=backend_str,
                velocities=None,
            )

            if not args.no_display:
                cv2.imshow("VisionTracker", annotated)
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
        print("[Main] Done. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
