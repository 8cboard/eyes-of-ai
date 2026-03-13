"""
main.py — VisionTracker simplified entry point.

New simplified architecture:
  Camera → EdgeDetector → CentroidTracker → [draw boxes] → RemoteLLMService → UIOverlay

Key changes:
  - Uses OpenCV edge/contour detection instead of YOLO
  - No stillness detection — identifies continuously
  - Sends full annotated frames (with colored boxes) to remote LLM
  - Only CentroidTracker (ByteTrack hardcoded disabled)
  - No OpenRouter support — only remote Kaggle/Colab LLM server

Run:
  python main.py --remote-url https://xxx.ngrok.io

See README.md for full documentation.
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
        description="VisionTracker — edge detection, tracking, and remote LLM identification",
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
    p.add_argument("--min-area", type=int, default=1000,
                   help="Minimum contour area to detect (pixels)")
    p.add_argument("--max-area", type=int, default=500000,
                   help="Maximum contour area to detect (pixels)")
    p.add_argument("--canny-low", type=int, default=50,
                   help="Canny edge detection lower threshold")
    p.add_argument("--canny-high", type=int, default=150,
                   help="Canny edge detection upper threshold")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Run detector every N frames (1=every frame)")

    # Tracker (Centroid only)
    p.add_argument("--max-disappeared", type=int, default=20,
                   help="Frames before track is dropped")

    # Remote LLM Service
    p.add_argument("--remote-url", default=None,
                   help="URL of the remote LLM server (e.g., https://xxx.ngrok.io). "
                        "Can also be set via REMOTE_LLM_URL env var.")
    p.add_argument("--remote-key", default=None,
                   help="Optional API key for remote server authentication. "
                        "Can also be set via REMOTE_LLM_KEY env var.")
    p.add_argument("--id-ttl", type=float, default=45.0,
                   help="Seconds before re-identification of a cached track")
    p.add_argument("--batch-wait", type=int, default=500,
                   help="ms to wait before dispatching frame (default 500)")

    # UI
    p.add_argument("--no-display", action="store_true",
                   help="Suppress OpenCV window (useful for testing)")
    p.add_argument("--save-video", type=str, default=None,
                   help="Save output to video file (e.g., output.mp4)")

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
# Color generator for bounding boxes
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
# Drawing helper for annotated frames
# ─────────────────────────────────────────────────────────────────────────────

def draw_annotated_frame(
    frame: np.ndarray,
    tracked_objects: list,
    progress_dict: dict,
    fps: float,
    queue_depth: int = 0,
) -> np.ndarray:
    """Draw bounding boxes and labels onto frame for display and sending to LLM.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame.
    tracked_objects : list
        List of TrackedObject from tracker.
    progress_dict : dict
        Identification progress from RemoteLLMService.
    fps : float
        Current FPS.
    queue_depth : int
        Current queue depth for display.

    Returns
    -------
    np.ndarray
        Annotated frame with boxes and labels.
    """
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    for obj in tracked_objects:
        color = _color(obj.track_id)
        x1, y1, x2, y2 = obj.xyxy.astype(int)

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Get identification progress
        prog = progress_dict.get(obj.track_id, {})
        status = prog.get("status", "idle")
        label = prog.get("label")

        # Build label text
        header = f"#{obj.track_id}"
        if status == "done" and label:
            id_line = label[:40]
        elif status == "identifying":
            id_line = "Identifying..."
        elif status == "queued":
            id_line = "Queued..."
        elif status == "error":
            id_line = "Error"
        else:
            id_line = None

        # Draw label background
        lines = [header]
        if id_line:
            lines.append(id_line)

        pad = 4
        line_height = 18
        block_h = line_height * len(lines) + pad * 2
        block_w = max(
            cv2.getTextSize(line, font, font_scale, 1)[0][0] for line in lines
        ) + pad * 2

        lx = max(0, x1)
        ly = max(block_h, y1) - block_h

        # Semi-transparent background
        sub = out[ly: ly + block_h, lx: lx + block_w]
        if sub.size > 0:
            overlay = sub.copy()
            cv2.rectangle(overlay, (0, 0), (sub.shape[1], sub.shape[0]), color, -1)
            cv2.addWeighted(overlay, 0.45, sub, 0.55, 0, sub)
            out[ly: ly + block_h, lx: lx + block_w] = sub

        # Draw text
        for i, line in enumerate(lines):
            ty = ly + pad + (i + 1) * line_height - 4
            cv2.putText(out, line, (lx + pad, ty), font, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Status dot
        if status == "done":
            cv2.circle(out, (x1 + 6, y1 + 6), 4, (0, 220, 0), -1)  # green
        elif status in ("identifying", "queued"):
            cv2.circle(out, (x1 + 6, y1 + 6), 4, (0, 165, 255), -1)  # orange

    # HUD bar
    h, w = out.shape[:2]
    bar_h = 24
    overlay = out[0:bar_h, 0:w].copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, out[0:bar_h, 0:w], 0.4, 0, out[0:bar_h, 0:w])

    hud_text = (
        f"FPS: {fps:.1f}  |  Tracks: {len(tracked_objects)}  |  "
        f"Queue: {queue_depth}  |  Press Q to quit"
    )
    cv2.putText(
        out, hud_text, (8, 16),
        font, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── Imports (deferred to allow --help without dependencies) ──────────────
    from edge_detector import EdgeDetector
    from tracker import build_tracker
    from remote_llm_service import RemoteLLMService

    # ── Get remote URL ───────────────────────────────────────────────────────
    remote_url = args.remote_url or os.environ.get("REMOTE_LLM_URL", "")

    if not remote_url:
        print("[Main] ERROR: Remote LLM URL is required.")
        print("[Main] Set --remote-url or REMOTE_LLM_URL env var.")
        print("[Main] Run remote_server/colab_setup.ipynb or kaggle_setup.ipynb")
        return 1

    # ── Build components ─────────────────────────────────────────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  VisionTracker starting")
    print(f"  Remote LLM: {remote_url}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    detector = EdgeDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        canny_thresh1=args.canny_low,
        canny_thresh2=args.canny_high,
        skip_frames=args.skip_frames,
    )

    tracker = build_tracker(
        max_disappeared=args.max_disappeared,
    )

    id_service = RemoteLLMService(
        remote_url=remote_url,
        api_key=args.remote_key or os.environ.get("REMOTE_LLM_KEY"),
        cache_ttl=args.id_ttl,
        batch_wait_ms=args.batch_wait,
    )

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
    print(f"[Main] Detector: EdgeDetector | skip={args.skip_frames}")
    print(f"[Main] Tracker: {tracker.name}")
    if not args.no_display:
        print("[Main] Press Q to quit.\n")

    # ── Video writer (optional) ──────────────────────────────────────────────
    video_writer: Optional[cv2.VideoWriter] = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, 30.0, (actual_w, actual_h)
        )
        print(f"[Main] Saving output to: {args.save_video}")

    active_track_ids: set[int] = set()
    frame_counter = 0

    # ── Main loop ────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] End of stream.")
                break

            frame_counter += 1

            # Prepare display frame
            if args.grayscale:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = frame

            # ── Detect ────────────────────────────────────────────────────────
            det_result = detector.detect(display_frame)

            # ── Track ─────────────────────────────────────────────────────────
            tracked = tracker.update(det_result)
            current_ids = {obj.track_id for obj in tracked}

            # Cleanup disappeared tracks
            for tid in active_track_ids - current_ids:
                pass  # No stillness state to clean up in simplified arch
            active_track_ids = current_ids

            # ── Draw annotated frame ──────────────────────────────────────────
            annotated = draw_annotated_frame(
                frame=display_frame,
                tracked_objects=tracked,
                progress_dict=id_service.progress,
                fps=fps_counter.fps,
                queue_depth=id_service.queue_depth(),
            )

            # ── Submit to remote LLM (throttled by queue) ─────────────────────
            # Only submit every 10 frames to avoid flooding
            if frame_counter % 10 == 0 and len(tracked) > 0:
                # Pick the first untracked object or oldest track
                for obj in tracked:
                    if id_service.get_cached(obj.track_id) is None:
                        track_ids = [t.track_id for t in tracked]
                        submitted = id_service.submit(
                            obj.track_id,
                            annotated,
                            track_ids,
                        )
                        if submitted:
                            print(f"[Main] → queued frame with track #{obj.track_id} "
                                  f"| queue depth: {id_service.queue_depth()}")
                        break

            # ── Render / Save ─────────────────────────────────────────────────
            fps = fps_counter.tick()

            if not args.no_display:
                cv2.imshow("VisionTracker", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    print("[Main] Quit requested.")
                    break

            if video_writer is not None:
                video_writer.write(annotated)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        print("[Main] Cleaning up...")
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        id_service.shutdown()
        print("[Main] Done. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
