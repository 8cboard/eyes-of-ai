"""
ui_overlay.py — OpenCV-based UI overlay for VisionTracker.

Draws bounding boxes, track IDs, class labels, identification status/progress,
and HUD statistics on each frame.  Designed to be called from the main thread
only (cv2.imshow is not thread-safe on some platforms).

Key visual elements:
  ┌─────────────────────────────────────────────────────┐
  │  FPS: 18.3 | Tracks: 4 | Mode: local/blip          │
  ├───────┬─────────────────────────────────────────────┤
  │  #12  │  person                                     │  ← class label
  │       │  "a person in a blue jacket"                │  ← ID label
  │       │  [████░░░░░░] 45% Identifying…              │  ← progress bar
  └───────┴─────────────────────────────────────────────┘

Colors cycle by track_id for easy visual disambiguation.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from tracker import TrackedObject


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (cycling by track_id)
# ─────────────────────────────────────────────────────────────────────────────

# 20 visually distinct BGR colours
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
# Overlay renderer
# ─────────────────────────────────────────────────────────────────────────────

class UIOverlay:
    """Draws all visual elements onto an OpenCV frame (in-place).

    Parameters
    ----------
    show_velocity : bool
        Draw a small velocity indicator (px/frame) near each box.
    show_confidence : bool
        Draw detection confidence score on boxes.
    font_scale : float
        Base font scale for text.  Adjust for different resolutions.
    box_thickness : int
        Bounding box line thickness.
    """

    def __init__(
        self,
        show_velocity: bool = False,
        show_confidence: bool = True,
        font_scale: float = 0.5,
        box_thickness: int = 2,
    ) -> None:
        self.show_velocity = show_velocity
        self.show_confidence = show_confidence
        self.font_scale = font_scale
        self.box_thickness = box_thickness

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._fps_history: list[float] = []

    # ── Main draw call ─────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        progress_dict: dict,          # from id_service.IdentificationService.progress
        fps: float,
        mode: str = "local",
        backend: str = "blip",
        velocities: Optional[dict[int, float]] = None,
    ) -> np.ndarray:
        """Render all overlays onto *frame* and return the annotated copy.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV (not modified in-place).
        tracked_objects : list[TrackedObject]
        progress_dict : dict
            Shared identification progress dictionary.
        fps : float
        mode : str
            'local' or 'hybrid' — shown in HUD.
        backend : str
            Backend name — shown in HUD.
        velocities : dict[int, float] | None
            Optional {track_id: velocity} for velocity indicator.
        """
        out = frame.copy()

        for obj in tracked_objects:
            self._draw_object(out, obj, progress_dict, velocities)

        self._draw_hud(out, fps, len(tracked_objects), mode, backend)
        return out

    # ── Per-object drawing ─────────────────────────────────────────────────────

    def _draw_object(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        progress_dict: dict,
        velocities: Optional[dict[int, float]],
    ) -> None:
        color = _color(obj.track_id)
        x1, y1, x2, y2 = obj.xyxy.astype(int)

        # ── Bounding box ───────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        # ── Build label lines ──────────────────────────────────────────────
        prog = progress_dict.get(obj.track_id, {})
        status = prog.get("status", "idle")
        label = prog.get("label")
        progress_val = prog.get("progress", 0.0)

        # Line 1: Class only (no track ID)
        conf_str = f" {obj.confidence:.0%}" if self.show_confidence else ""
        header = f"{obj.class_name}{conf_str}"

        # Line 2: ID result or status
        if status == "done" and label:
            id_line: Optional[str] = _truncate(label, 40)
        elif status in ("identifying", "queued"):
            pct = int(progress_val * 100)
            partial = label or ""
            id_line = f"Identifying… {pct}%  {_truncate(partial, 20)}"
        elif status == "error":
            id_line = f"[err] {obj.class_name}"
        else:
            id_line = None

        # Velocity
        vel_line: Optional[str] = None
        if self.show_velocity and velocities and obj.track_id in velocities:
            vel = velocities[obj.track_id]
            vel_line = f"vel: {vel:.1f}px"

        # ── Draw background + text above box ──────────────────────────────
        lines = [header]
        if id_line:
            lines.append(id_line)
        if vel_line:
            lines.append(vel_line)

        self._draw_label_block(frame, x1, y1, lines, color)

        # ── Progress bar (while identifying) ──────────────────────────────
        if status in ("identifying", "queued") and (x2 - x1) > 20:
            self._draw_progress_bar(frame, x1, y2, x2, progress_val, color)

        # ── Still indicator (green dot) ────────────────────────────────────
        if status == "idle":
            pass  # No indicator when idle
        elif status == "done":
            cv2.circle(frame, (x1 + 6, y1 + 6), 4, (0, 220, 0), -1)  # green
        elif status in ("identifying", "queued"):
            cv2.circle(frame, (x1 + 6, y1 + 6), 4, (0, 165, 255), -1)  # orange

    def _draw_label_block(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        lines: list[str],
        color: tuple[int, int, int],
    ) -> None:
        """Draw a semi-transparent background rectangle and text lines above the box."""
        fs = self.font_scale
        pad = 4
        line_height = int(18 * fs + 4)
        block_h = line_height * len(lines) + pad * 2
        block_w = max(
            cv2.getTextSize(line, self._font, fs, 1)[0][0] for line in lines
        ) + pad * 2

        # Clamp so label stays within frame
        lx = max(0, x1)
        ly = max(block_h, y1) - block_h

        # Semi-transparent background
        sub = frame[ly: ly + block_h, lx: lx + block_w]
        if sub.size > 0:
            overlay = sub.copy()
            cv2.rectangle(overlay, (0, 0), (sub.shape[1], sub.shape[0]), color, -1)
            cv2.addWeighted(overlay, 0.45, sub, 0.55, 0, sub)
            frame[ly: ly + block_h, lx: lx + block_w] = sub

        # Draw each text line
        for i, line in enumerate(lines):
            ty = ly + pad + (i + 1) * line_height - 4
            cv2.putText(frame, line, (lx + pad, ty), self._font, fs,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_progress_bar(
        self,
        frame: np.ndarray,
        x1: int,
        y2: int,
        x2: int,
        progress: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a thin progress bar along the bottom edge of the bounding box."""
        bar_h = 5
        bar_y1 = y2
        bar_y2 = y2 + bar_h
        bar_w = x2 - x1
        filled_w = int(bar_w * min(1.0, max(0.0, progress)))

        # Background
        cv2.rectangle(frame, (x1, bar_y1), (x2, bar_y2), (50, 50, 50), -1)
        # Filled portion
        if filled_w > 0:
            cv2.rectangle(frame, (x1, bar_y1), (x1 + filled_w, bar_y2), color, -1)

    # ── HUD (frame stats) ──────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        n_tracks: int,
        mode: str,
        backend: str,
    ) -> None:
        """Draw semi-transparent top-bar with FPS, track count, mode."""
        h, w = frame.shape[:2]
        bar_h = 24
        overlay = frame[0:bar_h, 0:w].copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame[0:bar_h, 0:w], 0.4, 0, frame[0:bar_h, 0:w])

        hud_text = (
            f"FPS: {fps:.1f}  |  Tracks: {n_tracks}  |  "
            f"Mode: {mode}/{backend}  |  Press Q to quit"
        )
        cv2.putText(
            frame, hud_text, (8, 16),
            self._font, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    """Truncate a string and add ellipsis if it exceeds max_len."""
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"
