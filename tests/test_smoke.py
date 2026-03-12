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
def synthetic_crop() -> np.ndarray:
    """Return a small synthetic BGR crop (simulated object region)."""
    crop = np.zeros((100, 80, 3), dtype=np.uint8)
    crop[20:80, 10:70] = (100, 200, 50)
    return crop


@pytest.fixture
def synthetic_detections():
    """Return a fake DetectionResult with two synthetic boxes."""
    from detector import DetectionResult
    return DetectionResult(
        xyxy=np.array([[300, 200, 600, 400], [800, 100, 1000, 200]], dtype=np.float32),
        confidences=np.array([0.92, 0.75], dtype=np.float32),
        class_ids=np.array([0, 73], dtype=np.int32),  # person, book
        class_names=["person", "book"],
        frame_index=1,
        inference_ms=12.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# detector.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDetector:
    def test_import(self):
        """Detector module can be imported."""
        from detector import Detector, DetectionResult  # noqa: F401
        assert True

    def test_detection_result_defaults(self):
        """DetectionResult initialises with empty arrays."""
        from detector import DetectionResult
        r = DetectionResult()
        assert r.count == 0
        assert r.xyxy.shape == (0, 4)
        assert r.confidences.shape == (0,)
        assert r.class_ids.shape == (0,)

    def test_detection_result_count(self, synthetic_detections):
        """count property matches number of boxes."""
        assert synthetic_detections.count == 2

    def test_auto_device_returns_string(self):
        """_auto_select_device returns a valid device string."""
        from detector import _auto_select_device
        device = _auto_select_device()
        assert device in ("cpu", "cuda", "mps")

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ultralytics"),
        reason="ultralytics not installed",
    )
    def test_detector_construct(self):
        """Detector can be constructed with yolov8n (downloads on first run)."""
        from detector import Detector
        # skip_frames=1, small imgsz to run quickly
        det = Detector(model_name="yolov8n.pt", skip_frames=1, imgsz=320, device="cpu")
        assert det.frame_count == 0

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ultralytics"),
        reason="ultralytics not installed",
    )
    def test_detector_runs_on_synthetic_frame(self, synthetic_frame_720p):
        """Detector returns a DetectionResult without crashing on a synthetic frame."""
        from detector import Detector, DetectionResult
        det = Detector(model_name="yolov8n.pt", skip_frames=1, imgsz=320, device="cpu")
        result = det.detect(synthetic_frame_720p)
        assert isinstance(result, DetectionResult)
        # Synthetic frame probably has 0 detections — just check types
        assert result.xyxy.ndim == 2
        assert result.xyxy.shape[1] == 4

    def test_grayscale_input_auto_converts(self, synthetic_frame_720p):
        """_ensure_bgr converts grayscale to 3-channel without error."""
        from detector import _ensure_bgr
        import cv2
        gray = cv2.cvtColor(synthetic_frame_720p, cv2.COLOR_BGR2GRAY)
        bgr = _ensure_bgr(gray)
        assert bgr.ndim == 3 and bgr.shape[2] == 3


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
        from detector import DetectionResult
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
        from detector import DetectionResult
        t = CentroidTracker(max_disappeared=3)

        # Register one track
        det = DetectionResult(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["person"],
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
# stability.py
# ─────────────────────────────────────────────────────────────────────────────

class TestStability:
    def test_import(self):
        from stability import IoUVelocityStillnessDetector, OpticalFlowStillnessDetector  # noqa: F401
        assert True

    def test_not_still_on_first_update(self):
        """A track is not still immediately after first update."""
        from stability import IoUVelocityStillnessDetector
        s = IoUVelocityStillnessDetector(history_len=5, still_frames=3)
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        s.update(1, bbox)
        assert not s.is_still(1)

    def test_becomes_still_after_m_stable_frames(self):
        """Track becomes still after M frames of low motion."""
        from stability import IoUVelocityStillnessDetector
        s = IoUVelocityStillnessDetector(
            history_len=5, still_frames=3, velocity_thresh=10.0, iou_thresh=0.8
        )
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        # Feed 10 identical frames (zero motion, IoU=1.0)
        for _ in range(10):
            s.update(1, bbox)
        assert s.is_still(1)

    def test_not_still_with_high_velocity(self):
        """Track is not still when centroid moves a lot."""
        from stability import IoUVelocityStillnessDetector
        s = IoUVelocityStillnessDetector(
            history_len=5, still_frames=3, velocity_thresh=5.0
        )
        for i in range(15):
            # Object moving 30 px per frame
            bbox = np.array([i * 30, 100, i * 30 + 100, 200], dtype=np.float32)
            s.update(1, bbox)
        assert not s.is_still(1)

    def test_reset_clears_stable(self):
        """reset() clears the stable flag."""
        from stability import IoUVelocityStillnessDetector
        s = IoUVelocityStillnessDetector(history_len=5, still_frames=3)
        bbox = np.array([0, 0, 100, 100], dtype=np.float32)
        for _ in range(10):
            s.update(1, bbox)
        assert s.is_still(1)
        s.reset(1)
        assert not s.is_still(1)

    def test_remove_track(self):
        """remove_track() deletes track state."""
        from stability import IoUVelocityStillnessDetector
        s = IoUVelocityStillnessDetector()
        bbox = np.array([0, 0, 100, 100], dtype=np.float32)
        s.update(42, bbox)
        assert 42 in s._states
        s.remove_track(42)
        assert 42 not in s._states

    def test_box_iou_self(self):
        """_box_iou returns 1.0 for a box with itself."""
        from stability import _box_iou
        box = np.array([10, 10, 50, 50], dtype=np.float32)
        assert _box_iou(box, box) == pytest.approx(1.0)

    def test_box_iou_no_overlap(self):
        from stability import _box_iou
        a = np.array([0, 0, 10, 10], dtype=np.float32)
        b = np.array([20, 20, 30, 30], dtype=np.float32)
        assert _box_iou(a, b) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# id_service.py
# ─────────────────────────────────────────────────────────────────────────────

class TestIDService:
    def test_import(self):
        from id_service import (  # noqa: F401
            IdentificationService, IdentificationCache, TokenBucket,
            RateLimitError, PendingItem, NEMOTRON_MODEL,
        )
        assert True

    def test_cache_miss_returns_none(self):
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=30)
        assert cache.get(999) is None

    def test_cache_set_and_get(self):
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(1, "a blue chair")
        assert cache.get(1) == "a blue chair"

    def test_cache_expired(self):
        import time
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=0.01)
        cache.set(1, "a plant")
        time.sleep(0.05)
        assert cache.get(1) is None

    def test_cache_invalidate(self):
        from id_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(5, "a red laptop")
        cache.invalidate(5)
        assert cache.get(5) is None

    def test_token_bucket_initial_full(self):
        from id_service import TokenBucket
        tb = TokenBucket(rate=1.0, capacity=3.0)
        assert tb.peek() == pytest.approx(3.0, abs=0.1)

    def test_token_bucket_acquire_reduces_tokens(self):
        from id_service import TokenBucket
        tb = TokenBucket(rate=0.01, capacity=3.0)  # very slow refill
        tb.acquire(1.0)
        assert tb.peek() < 3.0

    def test_token_bucket_refills_over_time(self):
        import time
        from id_service import TokenBucket
        tb = TokenBucket(rate=10.0, capacity=5.0)  # fast refill
        tb.acquire(5.0)  # drain it
        assert tb.peek() < 1.0
        time.sleep(0.15)
        assert tb.peek() > 0.5  # should have refilled ~1.5 tokens

    def test_pending_item_sortable(self):
        from id_service import PendingItem
        a = PendingItem(priority=1.0, track_id=1, crop=np.zeros((10,10,3),np.uint8),
                        class_name="cup", class_id=1)
        b = PendingItem(priority=2.0, track_id=2, crop=np.zeros((10,10,3),np.uint8),
                        class_name="dog", class_id=2)
        assert a < b  # lower priority = served first

    def test_rate_limit_error(self):
        from id_service import RateLimitError
        err = RateLimitError(retry_after=45.0)
        assert err.retry_after == 45.0
        assert "45" in str(err)

    def test_service_raises_without_api_key(self):
        """IdentificationService raises ValueError if no API key provided."""
        from id_service import IdentificationService
        import os
        # Temporarily clear env var
        orig = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key"):
                IdentificationService(api_key="")
        finally:
            if orig is not None:
                os.environ["OPENROUTER_API_KEY"] = orig

    def test_service_initialises_with_key(self):
        from id_service import IdentificationService
        svc = IdentificationService(api_key="sk-or-v1-test-fake-key-for-unit-test",
                                     cache_ttl=10)
        svc.shutdown()

    def test_submit_cached_returns_false(self):
        from id_service import IdentificationService
        svc = IdentificationService(api_key="sk-or-v1-test-fake",
                                     cache_ttl=60)
        try:
            # Manually populate cache
            svc._cache.set(42, "already identified")
            result = svc.submit(42, np.zeros((10,10,3),np.uint8), "cup", 1)
            assert result is False
        finally:
            svc.shutdown()

    def test_duplicate_submit_returns_false(self):
        from id_service import IdentificationService
        svc = IdentificationService(api_key="sk-or-v1-test-fake",
                                     cache_ttl=60)
        try:
            crop = np.zeros((50, 50, 3), np.uint8)
            r1 = svc.submit(7, crop, "cup", 1)
            r2 = svc.submit(7, crop, "cup", 1)
            if r1:
                assert r2 is False  # second submit while in-flight
        finally:
            svc.shutdown()

    def test_parse_batch_single(self):
        from id_service import _parse_batch_response, PendingItem
        item = PendingItem(priority=0, track_id=5, crop=np.zeros((1,1,3),np.uint8),
                           class_name="mug", class_id=1)
        result = _parse_batch_response("A red ceramic mug with a handle", [item])
        assert result[5] == "A red ceramic mug with a handle"

    def test_parse_batch_multi(self):
        from id_service import _parse_batch_response, PendingItem
        items = [
            PendingItem(priority=0, track_id=1, crop=np.zeros((1,1,3),np.uint8),
                        class_name="cup", class_id=1),
            PendingItem(priority=1, track_id=2, crop=np.zeros((1,1,3),np.uint8),
                        class_name="laptop", class_id=2),
        ]
        text = "1. A blue ceramic mug\n2. A silver laptop with stickers"
        result = _parse_batch_response(text, items)
        assert result[1] == "A blue ceramic mug"
        assert result[2] == "A silver laptop with stickers"

    def test_parse_batch_fallback_on_bad_format(self):
        from id_service import _parse_batch_response, PendingItem
        items = [
            PendingItem(priority=0, track_id=9, crop=np.zeros((1,1,3),np.uint8),
                        class_name="chair", class_id=56),
        ]
        result = _parse_batch_response("", items)
        # Falls back to class_name
        assert result[9] == "chair"

    def test_encode_crop_returns_valid_b64(self, synthetic_crop):
        from id_service import _encode_crop
        b64 = _encode_crop(synthetic_crop)
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_build_message_content_single(self, synthetic_crop):
        from id_service import _build_message_content, PendingItem
        item = PendingItem(priority=0, track_id=1, crop=synthetic_crop,
                           class_name="bottle", class_id=39)
        content = _build_message_content([item])
        types = [c["type"] for c in content]
        assert "text" in types
        assert "image_url" in types

    def test_build_message_content_batch(self, synthetic_crop):
        from id_service import _build_message_content, PendingItem
        items = [
            PendingItem(priority=i, track_id=i, crop=synthetic_crop,
                        class_name="obj", class_id=0)
            for i in range(3)
        ]
        content = _build_message_content(items)
        image_blocks = [c for c in content if c["type"] == "image_url"]
        assert len(image_blocks) == 3


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
            class_name="person",
            confidence=0.9,
        )
        result = overlay.draw(
            frame=synthetic_frame_720p,
            tracked_objects=[obj],
            progress_dict={1: {"status": "done", "progress": 1.0,
                               "label": "a person in a blue jacket", "error": None}},
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
            class_id=56, class_name="chair", confidence=0.8,
        )
        result = overlay.draw(
            frame=synthetic_frame_720p,
            tracked_objects=[obj],
            progress_dict={2: {"status": "identifying", "progress": 0.45,
                               "label": "a wooden…", "error": None}},
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
