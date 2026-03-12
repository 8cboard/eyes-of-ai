"""
main.py — VisionTracker entry point.

Orchestrates the full pipeline:
  Camera → Detector → Tracker → StillnessDetector → IDService → UIOverlay

Run:
  python main.py --mode local --backend blip
  python main.py --mode hybrid                  # HuggingFace free API
  python main.py --mode local --backend ollama  # requires: ollama serve

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
        description="VisionTracker — real-time detection, tracking, and LLM identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    p.add_argument("--input", default="0",
                   help="Camera index (0) or path to video file")
    p.add_argument("--width", type=int, default=1280, help="Capture width in pixels")
    p.add_argument("--height", type=int, default=720, help="Capture height in pixels")
    p.add_argument("--grayscale", action="store_true",
                   help="Force grayscale (faster on CPU; detection still uses BGR internally)")

    # Detector
    p.add_argument("--detector", default="yolov8n.pt",
                   choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"],
                   help="YOLOv8 model size")
    p.add_argument("--skip-frames", type=int, default=2,
                   help="Run detector every N frames (1=every frame)")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Detector input size (320=faster, 640=default, 1280=accurate)")
    p.add_argument("--device", default=None,
                   help="Compute device: cpu, cuda, mps (default: auto-detect)")

    # Tracker
    p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "centroid"],
                   help="Tracker type (centroid is faster on very slow CPUs)")

    # Stillness
    p.add_argument("--still-frames", type=int, default=10,
                   help="M: consecutive below-threshold frames to trigger ID")
    p.add_argument("--still-window", type=int, default=15,
                   help="N: centroid/IoU history window length")
    p.add_argument("--velocity-thresh", type=float, default=5.0,
                   help="Max centroid pixel velocity (px/frame) to count as still")
    p.add_argument("--iou-thresh", type=float, default=0.85,
                   help="Min consecutive IoU to count as geometrically stable")
    p.add_argument("--optical-flow", action="store_true",
                   help="Use optical-flow stillness detector (for moving cameras)")

    # ID Service — OpenRouter / Nemotron Nano 12B VL
    p.add_argument("--openrouter-key", default=None,
                   help="OpenRouter API key (free). "
                        "Obtain at https://openrouter.ai — no credit card needed for :free models. "
                        "Can also be set via OPENROUTER_API_KEY env var.")
    p.add_argument("--model", default=None,
                   help="OpenRouter model slug (default: nvidia/llama-3.1-nemotron-nano-12b-vl-instruct:free)")
    p.add_argument("--id-ttl", type=float, default=45.0,
                   help="Seconds before re-identification of a cached track")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Max crops per OpenRouter API call (1–4). "
                        "Higher = fewer requests but slightly more latency.")
    p.add_argument("--batch-wait", type=int, default=1500,
                   help="ms to wait for a fuller batch before dispatching (default 1500)")

    # ID Service — Self-hosted Gemma 3 4B Remote Server
    p.add_argument("--use-remote-gemma", action="store_true",
                   help="Use self-hosted Gemma 3 4B on Colab/Kaggle instead of OpenRouter")
    p.add_argument("--remote-gemma-url", default=None,
                   help="URL of the remote Gemma server (e.g., https://xxx.ngrok.io). "
                        "Can also be set via REMOTE_GEMMA_URL env var.")
    p.add_argument("--remote-gemma-key", default=None,
                   help="Optional API key for remote server authentication. "
                        "Can also be set via REMOTE_GEMMA_KEY env var.")

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
# Crop helper
# ─────────────────────────────────────────────────────────────────────────────

def crop_object(frame: np.ndarray, xyxy: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crop and return the object region from *frame* with optional padding."""
    h, w = frame.shape[:2]
    x1 = max(0, int(xyxy[0]) - pad)
    y1 = max(0, int(xyxy[1]) - pad)
    x2 = min(w, int(xyxy[2]) + pad)
    y2 = min(h, int(xyxy[3]) + pad)
    return frame[y1:y2, x1:x2].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:  # noqa: C901 (acceptable complexity for an orchestrator)
    args = parse_args()

    # ── Imports (deferred to allow --help without torch) ─────────────────────
    from detector import Detector
    from stability import IoUVelocityStillnessDetector, OpticalFlowStillnessDetector
    from tracker import build_tracker
    from ui_overlay import UIOverlay

    # ── Determine ID service backend ─────────────────────────────────────────
    use_remote_gemma = args.use_remote_gemma
    remote_gemma_url = args.remote_gemma_url or os.environ.get("REMOTE_GEMMA_URL", "")

    if use_remote_gemma or remote_gemma_url:
        from gemma_remote_service import GemmaRemoteService
        id_backend_name = "Gemma 3 4B (self-hosted)"
        id_service_class = GemmaRemoteService
        id_service_kwargs = dict(
            remote_url    = remote_gemma_url,
            api_key       = args.remote_gemma_key or os.environ.get("REMOTE_GEMMA_KEY"),
            cache_ttl     = args.id_ttl,
            batch_size    = args.batch_size or 8,  # Higher default for remote
            batch_wait_ms = args.batch_wait or 1000,
        )
    else:
        from id_service import IdentificationService, NEMOTRON_MODEL
        id_backend_name = "OpenRouter / Nemotron Nano 12B VL"
        id_service_class = IdentificationService
        id_service_kwargs = dict(
            api_key       = args.openrouter_key,
            model         = args.model or NEMOTRON_MODEL,
            cache_ttl     = args.id_ttl,
            batch_size    = args.batch_size,
            batch_wait_ms = args.batch_wait,
        )

    # ── Build components ─────────────────────────────────────────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  VisionTracker starting")
    print(f"  ID backend: {id_backend_name}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    detector = Detector(
        model_name=args.detector,
        skip_frames=args.skip_frames,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    tracker = build_tracker(args.tracker)

    if args.optical_flow:
        stillness = OpticalFlowStillnessDetector(
            history_len=args.still_window,
            still_frames=args.still_frames,
        )
        print("[Main] Using optical-flow stillness detector")
    else:
        stillness = IoUVelocityStillnessDetector(
            history_len=args.still_window,
            still_frames=args.still_frames,
            velocity_thresh=args.velocity_thresh,
            iou_thresh=args.iou_thresh,
        )

    id_service = id_service_class(**id_service_kwargs)

    overlay = UIOverlay(
        show_velocity=args.show_velocity,
    )
    fps_counter = FPSCounter()

    # ── Open camera / video ──────────────────────────────────────────────────
    input_src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(input_src)
    if not cap.isOpened():
        print(f"[Main] ERROR: Cannot open input: {args.input}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # reduce latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Main] Capture: {actual_w}×{actual_h}")
    print(f"[Main] Detector: {args.detector} | skip={args.skip_frames}")
    print(f"[Main] Tracker:  {tracker.name}")
    print(f"[Main] Stillness: M={args.still_frames} frames, N={args.still_window} window")
    if not args.no_display:
        print("[Main] Press Q to quit.\n")

    active_track_ids: set[int] = set()

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

            # ── Detect ────────────────────────────────────────────────────
            det_result = detector.detect(display_frame)

            # ── Track ─────────────────────────────────────────────────────
            tracked = tracker.update(det_result)
            current_ids = {obj.track_id for obj in tracked}

            for tid in active_track_ids - current_ids:
                stillness.remove_track(tid)
            active_track_ids = current_ids

            if args.optical_flow:
                gf = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                stillness.process_frame(gf)  # type: ignore[attr-defined]

            # ── Stillness gating + multi-object ID submission ──────────────
            velocities: dict[int, float] = {}
            for obj in tracked:
                stillness.update(obj.track_id, obj.xyxy)

                if hasattr(stillness, "get_velocity"):
                    velocities[obj.track_id] = stillness.get_velocity(obj.track_id)  # type: ignore

                if stillness.is_still(obj.track_id):
                    if id_service.get_cached(obj.track_id) is None:
                        crop = crop_object(frame, obj.xyxy)
                        if crop.size > 0:
                            submitted = id_service.submit(
                                obj.track_id, crop, obj.class_name, obj.class_id
                            )
                            if submitted:
                                q_depth = id_service.queue_depth()
                                print(f"[Main] → queued #{obj.track_id} ({obj.class_name}) "
                                      f"| queue depth: {q_depth}")
                            stillness.reset(obj.track_id)
                    else:
                        id_service.inject_cached(obj.track_id)

            # ── Render ────────────────────────────────────────────────────
            fps = fps_counter.tick()
            if use_remote_gemma or remote_gemma_url:
                backend_str = f"gemma-remote | q:{id_service.queue_depth()}"
                mode_str = "remote-gemma"
            else:
                backend_str = f"nemotron | q:{id_service.queue_depth()}"
                mode_str = "openrouter"
            annotated = overlay.draw(
                frame=display_frame,
                tracked_objects=tracked,
                progress_dict=id_service.progress,
                fps=fps,
                mode=mode_str,
                backend=backend_str,
                velocities=velocities if args.show_velocity else None,
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
