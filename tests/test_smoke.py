"""
tests/test_smoke.py — Smoke tests for VisionTracker (Simplified Architecture)

These tests use synthetic numpy frames (no real webcam, no internet) to
verify that each component can be imported and called without crashing.

Run:
    pytest tests/ -v

All tests are designed to pass on CI with CPU-only hardware.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Add project root to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_frame_720p() -> np.ndarray:
    """Return a synthetic 720p BGR frame with a colored rectangle."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw a simple colored rectangle (simulated "object")
    frame[200:400, 300:600] = (0, 128, 255)  # orange rectangle
    frame[100:200, 800:1000] = (255, 0, 0)    # blue rectangle
    return frame


@pytest.fixture
def synthetic_detections():
    """Return a fake DetectionResult with two synthetic boxes."""
    from edge_detector import DetectionResult
    return DetectionResult(
        xyxy=np.array([[300, 200, 600, 400], [800, 100, 1000, 200]], dtype=np.float32),
        confidences=np.array([0.92, 0.75], dtype=np.float32),
        class_ids=np.array([0, 0], dtype=np.int32),  # All 0 for edge detection
        class_names=["object", "object"],
        frame_index=1,
        inference_ms=12.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# edge_detector.py
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeDetector:
    def test_import(self):
        """Edge detector module can be imported."""
        from edge_detector import EdgeDetector, DetectionResult  # noqa: F401
        assert True

    def test_detection_result_defaults(self):
        """DetectionResult initialises with empty arrays."""
        from edge_detector import DetectionResult
        r = DetectionResult()
        assert r.count == 0
        assert r.xyxy.shape == (0, 4)
        assert r.confidences.shape == (0,)

    def test_detection_result_count(self, synthetic_detections):
        """count property matches number of boxes."""
        assert synthetic_detections.count == 2

    def test_detector_constructs(self):
        """EdgeDetector can be constructed."""
        from edge_detector import EdgeDetector
        det = EdgeDetector(min_area=1000)
        assert det.frame_count == 0

    def test_detector_runs_on_synthetic_frame(self, synthetic_frame_720p):
        """Detector returns a DetectionResult without crashing."""
        from edge_detector import EdgeDetector, DetectionResult
        det = EdgeDetector(min_area=100, max_area=1000000)
        result = det.detect(synthetic_frame_720p)
        assert isinstance(result, DetectionResult)
        assert result.xyxy.ndim == 2
        assert result.xyxy.shape[1] == 4

    def test_detector_skips_frames(self, synthetic_frame_720p):
        """skip_frames returns cached result."""
        from edge_detector import EdgeDetector
        det = EdgeDetector(skip_frames=2)
        r1 = det.detect(synthetic_frame_720p)
        r2 = det.detect(synthetic_frame_720p)
        assert r1.frame_index == r2.frame_index  # Same cached result


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

    def test_build_tracker_returns_centroid(self):
        """build_tracker returns CentroidTracker (ByteTrack disabled)."""
        from tracker import build_tracker, CentroidTracker
        t = build_tracker()  # No argument needed
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
# remote_llm_service.py
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoteLLMService:
    def test_import(self):
        from remote_llm_service import (  # noqa: F401
            RemoteLLMService, RemoteLLMClient, IdentificationCache,
        )
        assert True

    def test_cache_miss_returns_none(self):
        from remote_llm_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=30)
        assert cache.get(999) is None

    def test_cache_set_and_get(self):
        from remote_llm_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(1, "a blue chair")
        assert cache.get(1) == "a blue chair"

    def test_cache_expired(self):
        import time
        from remote_llm_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=0.01)
        cache.set(1, "a plant")
        time.sleep(0.05)
        assert cache.get(1) is None

    def test_cache_invalidate(self):
        from remote_llm_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(5, "a red laptop")
        cache.invalidate(5)
        assert cache.get(5) is None


# ─────────────────────────────────────────────────────────────────────────────
# ui_overlay.py (kept for compatibility, though main.py now has its own drawing)
# ─────────────────────────────────────────────────────────────────────────────

class TestUIOverlay:
    def test_import(self):
        from ui_overlay import UIOverlay  # noqa: F401
        assert True

    def test_color_cycles_by_track_id(self):
        """_color returns a valid BGR tuple."""
        from ui_overlay import _color
        for i in range(50):
            c = _color(i)
            assert len(c) == 3
            assert all(0 <= v <= 255 for v in c)


# ─────────────────────────────────────────────────────────────────────────────
# Integration test (optional, no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_synthetic(self, synthetic_frame_720p):
        """Test that detector → tracker pipeline works."""
        from edge_detector import EdgeDetector
        from tracker import build_tracker

        # Create components
        detector = EdgeDetector(min_area=100)
        tracker = build_tracker()

        # Run pipeline
        detections = detector.detect(synthetic_frame_720p)
        tracked = tracker.update(detections)

        # Verify
        assert detections.xyxy.ndim == 2
        assert isinstance(tracked, list)
