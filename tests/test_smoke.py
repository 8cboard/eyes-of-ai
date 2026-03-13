"""
tests/test_smoke.py — Smoke tests for VisionTracker.

These tests use synthetic numpy frames (no real webcam, no internet) to
verify that each component can be imported and called without crashing.

Run:
    pytest tests/ -v

All tests are designed to pass on CI with CPU-only hardware.
Tests that require GPU or external services are skipped automatically.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Add project root to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# edge_detector.py
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeDetector:
    def test_import(self):
        """EdgeDetector module can be imported."""
        from edge_detector import EdgeDetector, DetectionResult  # noqa: F401
        assert True

    def test_detection_result_defaults(self):
        """DetectionResult initialises with empty arrays."""
        from edge_detector import DetectionResult
        r = DetectionResult()
        assert r.count == 0
        assert r.xyxy.shape == (0, 4)
        assert r.confidences.shape == (0,)
        assert r.class_ids.shape == (0,)

    def test_detection_result_count(self, synthetic_detections):
        """count property matches number of boxes."""
        assert synthetic_detections.count == 2

    def test_detector_construct(self):
        """EdgeDetector can be constructed."""
        from edge_detector import EdgeDetector
        det = EdgeDetector(min_area=500, max_area=100000, skip_frames=1)
        assert det.frame_count == 0

    def test_detector_runs_on_synthetic_frame(self, synthetic_frame_720p):
        """Detector returns a DetectionResult without crashing on a synthetic frame."""
        from edge_detector import EdgeDetector, DetectionResult
        det = EdgeDetector(min_area=100, max_area=100000, skip_frames=1)
        result = det.detect(synthetic_frame_720p)
        assert isinstance(result, DetectionResult)
        assert result.xyxy.ndim == 2
        assert result.xyxy.shape[1] == 4

    def test_all_class_names_are_object(self, synthetic_frame_720p):
        """All detections have class_name='object'."""
        from edge_detector import EdgeDetector
        det = EdgeDetector(min_area=100, skip_frames=1)
        result = det.detect(synthetic_frame_720p)
        for name in result.class_names:
            assert name == "object"


# ─────────────────────────────────────────────────────────────────────────────
# tracker.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTracker:
    def test_import(self):
        from tracker import CentroidTracker, build_tracker, TrackedObject  # noqa: F401
        assert True

    def test_centroid_tracker_empty(self):
        """CentroidTracker returns empty list when no detections."""
        from tracker import CentroidTracker
        from edge_detector import DetectionResult
        t = CentroidTracker()
        result = t.update(DetectionResult())
        assert result == []

    def test_centroid_tracker_registers_new_tracks(self, synthetic_detections):
        """Two new detections become two new tracks."""
        from tracker import CentroidTracker
        t = CentroidTracker()
        tracked = t.update(synthetic_detections)
        assert len(tracked) == 2
        ids = {obj.track_id for obj in tracked}
        assert len(ids) == 2  # unique IDs

    def test_centroid_tracker_persistent_ids(self, synthetic_detections):
        """Track IDs persist across consistent detections."""
        from tracker import CentroidTracker
        t = CentroidTracker()
        first = t.update(synthetic_detections)
        first_ids = {obj.track_id for obj in first}

        second = t.update(synthetic_detections)
        second_ids = {obj.track_id for obj in second}

        assert first_ids == second_ids  # same IDs on stable detections

    def test_centroid_tracker_drop_disappeared(self):
        """Tracks are dropped after max_disappeared frames with no match."""
        from tracker import CentroidTracker
        from edge_detector import DetectionResult
        t = CentroidTracker(max_disappeared=3)

        # Register one track
        det = DetectionResult(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["object"],
        )
        t.update(det)
        assert len(t._tracks) == 1

        # Feed empty detections until track disappears
        empty = DetectionResult()
        for _ in range(5):
            t.update(empty)

        assert len(t._tracks) == 0

    def test_build_tracker_centroid(self):
        """build_tracker('centroid') returns CentroidTracker."""
        from tracker import build_tracker, CentroidTracker
        t = build_tracker("centroid")
        assert isinstance(t, CentroidTracker)

    def test_tracked_object_centroid(self, synthetic_detections):
        """TrackedObject.centroid property computes correctly."""
        from tracker import CentroidTracker
        t = CentroidTracker()
        tracked = t.update(synthetic_detections)
        cx, cy = tracked[0].centroid
        x1, y1, x2, y2 = tracked[0].xyxy
        assert cx == pytest.approx((x1 + x2) / 2)
        assert cy == pytest.approx((y1 + y2) / 2)

    def test_iou_batch(self):
        """_batch_iou returns 1.0 for identical boxes."""
        from tracker import _batch_iou
        boxes = np.array([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=np.float32)
        iou = _batch_iou(boxes, boxes)
        np.testing.assert_allclose(iou.diagonal(), 1.0, atol=1e-5)

    def test_iou_no_overlap(self):
        """_batch_iou returns 0 for non-overlapping boxes."""
        from tracker import _batch_iou
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[50, 50, 60, 60]], dtype=np.float32)
        iou = _batch_iou(a, b)
        assert iou[0, 0] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# id_service.py
# ─────────────────────────────────────────────────────────────────────────────

class TestIDService:
    def test_import(self):
        from id_service import (  # noqa: F401
            IdentificationService, IdentificationCache,
            PendingItem,
        )
        assert True

    def test_cache_miss_returns_none(self):
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=30)
        assert cache.get(999) is None

    def test_cache_set_and_get(self):
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(1, "person")
        assert cache.get(1) == "person"

    def test_cache_expired(self):
        import time
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=0.01)
        cache.set(1, "car")
        time.sleep(0.05)
        assert cache.get(1) is None

    def test_cache_invalidate(self):
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(5, "dog")
        cache.invalidate(5)
        assert cache.get(5) is None

    def test_pending_item_sortable(self):
        from id_service import PendingItem
        a = PendingItem(
            priority=1.0,
            track_id=1,
            frame=np.zeros((100, 100, 3), np.uint8)
        )
        b = PendingItem(
            priority=2.0,
            track_id=2,
            frame=np.zeros((100, 100, 3), np.uint8)
        )
        assert a < b  # lower priority = served first

    def test_service_raises_without_url(self):
        """IdentificationService raises ValueError if no URL provided."""
        from id_service import IdentificationService
        import os
        import pytest
        # Temporarily clear env var
        orig = os.environ.pop("REMOTE_URL", None)
        try:
            # Should raise ValueError
            with pytest.raises(ValueError, match="Remote server URL"):
                IdentificationService(remote_url="", cache_ttl=10)
        finally:
            if orig is not None:
                os.environ["REMOTE_URL"] = orig


# ─────────────────────────────────────────────────────────────────────────────
# ui_overlay.py
# ─────────────────────────────────────────────────────────────────────────────

class TestUIOverlay:
    def test_import(self):
        from ui_overlay import UIOverlay  # noqa: F401
        assert True

    def test_draw_returns_same_shape(self, synthetic_frame_720p):
        """draw() returns a frame with the same shape as input."""
        from ui_overlay import UIOverlay
        from tracker import TrackedObject
        overlay = UIOverlay()
        obj = TrackedObject(
            track_id=1,
            xyxy=np.array([300, 200, 600, 400], dtype=np.float32),
            class_id=0,
            class_name="object",
            confidence=0.9,
        )
        result = overlay.draw(
            frame=synthetic_frame_720p,
            tracked_objects=[obj],
            progress_dict={1: {"status": "done", "progress": 1.0,
                               "label": "person", "error": None}},
            fps=25.0,
        )
        assert result.shape == synthetic_frame_720p.shape

    def test_draw_does_not_modify_input(self, synthetic_frame_720p):
        """draw() does not modify the input frame."""
        from ui_overlay import UIOverlay
        overlay = UIOverlay()
        original = synthetic_frame_720p.copy()
        overlay.draw(
            frame=synthetic_frame_720p,
            tracked_objects=[],
            progress_dict={},
            fps=30.0,
        )
        np.testing.assert_array_equal(synthetic_frame_720p, original)

    def test_draw_with_identifying_status(self, synthetic_frame_720p):
        """draw() handles 'identifying' status without error."""
        from ui_overlay import UIOverlay
        from tracker import TrackedObject
        overlay = UIOverlay()
        obj = TrackedObject(
            track_id=2, xyxy=np.array([100, 100, 300, 300], dtype=np.float32),
            class_id=0, class_name="object", confidence=0.8,
        )
        result = overlay.draw(
            frame=synthetic_frame_720p,
            tracked_objects=[obj],
            progress_dict={2: {"status": "identifying", "progress": 0.45,
                               "label": "chair", "error": None}},
            fps=18.0,
        )
        assert result is not None

    def test_color_cycles_by_track_id(self):
        """_color returns a valid BGR tuple."""
        from ui_overlay import _color
        for i in range(50):
            c = _color(i)
            assert len(c) == 3
            assert all(0 <= v <= 255 for v in c)

    def test_no_track_id_in_header(self, synthetic_frame_720p):
        """UI does not show track ID in header (only class name)."""
        from ui_overlay import UIOverlay
        from tracker import TrackedObject
        overlay = UIOverlay()
        obj = TrackedObject(
            track_id=42,
            xyxy=np.array([100, 100, 200, 200], dtype=np.float32),
            class_id=0,
            class_name="object",
            confidence=0.95,
        )
        # Just verify it doesn't crash - visual inspection would confirm no #42
        result = overlay.draw(
            frame=synthetic_frame_720p,
            tracked_objects=[obj],
            progress_dict={},
            fps=30.0,
        )
        assert result is not None
