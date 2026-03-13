"""
tests/test_smoke.py — Comprehensive tests for VisionTracker.

All tests run without a webcam, GPU, or internet connection.
Uses synthetic numpy frames and mock objects throughout.

Run:
    pytest tests/ -v
"""
from __future__ import annotations
import sys, os, time
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═════════════════════════════════════════════════════════════════════════════
# edge_detector.py
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectionResult:
    def test_defaults_are_empty(self):
        from edge_detector import DetectionResult
        r = DetectionResult()
        assert r.count == 0
        assert r.xyxy.shape    == (0, 4)
        assert r.confidences.shape == (0,)
        assert r.class_ids.shape   == (0,)
        assert r.class_names       == []

    def test_count_matches_boxes(self, synthetic_detections):
        assert synthetic_detections.count == 2

    def test_inference_ms_default_zero(self):
        from edge_detector import DetectionResult
        assert DetectionResult().inference_ms == 0.0


class TestMergeBoxesByProximity:
    """Unit tests for the proximity-merge helper (the core fragmentation fix)."""

    def test_empty_input(self):
        from edge_detector import _merge_boxes_by_proximity
        assert _merge_boxes_by_proximity([], 20) == []

    def test_single_box_unchanged(self):
        from edge_detector import _merge_boxes_by_proximity
        boxes = [[10, 10, 100, 100]]
        result = _merge_boxes_by_proximity(boxes, 20)
        assert len(result) == 1
        assert result[0] == [10, 10, 100, 100]

    def test_overlapping_boxes_merge(self):
        """Two boxes that overlap should become one."""
        from edge_detector import _merge_boxes_by_proximity
        boxes = [[0, 0, 100, 100], [80, 80, 200, 200]]
        result = _merge_boxes_by_proximity(boxes, expand_px=5)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0]
        assert x1 == 0 and y1 == 0 and x2 == 200 and y2 == 200

    def test_nearby_boxes_merge_within_expand(self):
        """Two boxes 10 px apart should merge when expand_px=15."""
        from edge_detector import _merge_boxes_by_proximity
        boxes = [[0, 0, 100, 100], [110, 0, 200, 100]]  # 10 px gap
        result = _merge_boxes_by_proximity(boxes, expand_px=15)
        assert len(result) == 1

    def test_far_boxes_stay_separate(self):
        """Boxes 200 px apart should NOT merge with expand_px=10."""
        from edge_detector import _merge_boxes_by_proximity
        boxes = [[0, 0, 100, 100], [400, 400, 500, 500]]
        result = _merge_boxes_by_proximity(boxes, expand_px=10)
        assert len(result) == 2

    def test_merge_produces_tight_bbox(self):
        """Merged box must be the tight bounding box of all inputs."""
        from edge_detector import _merge_boxes_by_proximity
        # Three horizontally adjacent boxes
        boxes = [[0, 0, 50, 50], [60, 0, 110, 50], [120, 0, 170, 50]]
        result = _merge_boxes_by_proximity(boxes, expand_px=15)
        assert len(result) == 1
        assert result[0][0] == 0    # x1 = leftmost
        assert result[0][2] == 170  # x2 = rightmost

    def test_zero_expand_no_merge_of_non_overlapping(self):
        """With expand_px=0 only actually-overlapping boxes merge."""
        from edge_detector import _merge_boxes_by_proximity
        boxes = [[0, 0, 50, 50], [60, 0, 110, 50]]  # touching but not overlapping
        result = _merge_boxes_by_proximity(boxes, expand_px=0)
        assert len(result) == 2


class TestAutoCanny:
    def test_returns_two_ints(self):
        from edge_detector import _auto_canny_thresholds
        gray = np.full((100, 100), 120, dtype=np.uint8)
        low, high = _auto_canny_thresholds(gray)
        assert isinstance(low,  int)
        assert isinstance(high, int)

    def test_low_less_than_high(self):
        from edge_detector import _auto_canny_thresholds
        for median_val in [20, 80, 128, 200, 240]:
            gray = np.full((10, 10), median_val, dtype=np.uint8)
            low, high = _auto_canny_thresholds(gray)
            assert low < high, f"low={low} >= high={high} for median={median_val}"

    def test_values_in_valid_range(self):
        from edge_detector import _auto_canny_thresholds
        gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        low, high = _auto_canny_thresholds(gray)
        assert 0 <= low  <= 255
        assert 0 <= high <= 255


class TestEdgeDetector:
    def test_construct_default(self):
        from edge_detector import EdgeDetector
        det = EdgeDetector()
        assert det.frame_count == 0

    def test_detect_returns_detection_result(self, synthetic_frame_720p):
        from edge_detector import EdgeDetector, DetectionResult
        det = EdgeDetector(min_area=100, skip_frames=1)
        r = det.detect(synthetic_frame_720p)
        assert isinstance(r, DetectionResult)
        assert r.xyxy.shape[1] == 4

    def test_all_class_names_are_object(self, synthetic_frame_720p):
        from edge_detector import EdgeDetector
        det = EdgeDetector(min_area=100)
        r = det.detect(synthetic_frame_720p)
        assert all(n == "object" for n in r.class_names)

    def test_frame_counter_increments(self, synthetic_frame_720p):
        from edge_detector import EdgeDetector
        det = EdgeDetector(skip_frames=1)
        det.detect(synthetic_frame_720p)
        det.detect(synthetic_frame_720p)
        assert det.frame_count == 2

    def test_skip_frames_returns_cached(self, synthetic_frame_720p):
        from edge_detector import EdgeDetector
        det = EdgeDetector(skip_frames=3, min_area=100)
        r1 = det.detect(synthetic_frame_720p)  # frame 1 — real detection
        r2 = det.detect(synthetic_frame_720p)  # frame 2 — cached
        r3 = det.detect(synthetic_frame_720p)  # frame 3 — real detection
        # frame 2 and 3 can differ; but cached result must be the same object
        assert r2 is r1  # same object reference on cache hit

    def test_merge_reduces_fragment_count(self, cluttered_frame_720p):
        """Merging should produce fewer or equal boxes than no-merge."""
        from edge_detector import EdgeDetector
        det_merge    = EdgeDetector(min_area=200, merge_distance=30)
        det_no_merge = EdgeDetector(min_area=200, merge_distance=0)
        r_merge    = det_merge.detect(cluttered_frame_720p)
        r_no_merge = det_no_merge.detect(cluttered_frame_720p)
        assert r_merge.count <= r_no_merge.count

    def test_confidences_in_range(self, synthetic_frame_720p):
        from edge_detector import EdgeDetector
        det = EdgeDetector(min_area=100)
        r = det.detect(synthetic_frame_720p)
        if r.count > 0:
            assert np.all(r.confidences >= 0.0)
            assert np.all(r.confidences <= 1.0)

    def test_boxes_within_frame(self, synthetic_frame_720p):
        """All returned bounding boxes must lie within frame dimensions."""
        from edge_detector import EdgeDetector
        h, w = synthetic_frame_720p.shape[:2]
        det = EdgeDetector(min_area=100)
        r = det.detect(synthetic_frame_720p)
        for x1, y1, x2, y2 in r.xyxy:
            assert x1 >= 0 and y1 >= 0
            assert x2 <= w and y2 <= h
            assert x2 > x1 and y2 > y1

    def test_grayscale_input(self):
        """Detector accepts single-channel grayscale input without crashing."""
        from edge_detector import EdgeDetector
        gray = np.zeros((480, 640), dtype=np.uint8)
        gray[100:200, 100:300] = 200
        det = EdgeDetector(min_area=100)
        r = det.detect(gray)
        assert r is not None

    def test_auto_canny_flag(self, synthetic_frame_720p):
        from edge_detector import EdgeDetector
        det_auto   = EdgeDetector(auto_canny=True,  min_area=100)
        det_manual = EdgeDetector(auto_canny=False, min_area=100)
        # Both should return valid results (may differ in count)
        r_auto   = det_auto.detect(synthetic_frame_720p)
        r_manual = det_manual.detect(synthetic_frame_720p)
        assert r_auto   is not None
        assert r_manual is not None


# ═════════════════════════════════════════════════════════════════════════════
# tracker.py
# ═════════════════════════════════════════════════════════════════════════════

class TestTrackerGeometry:
    def test_batch_iou_identical_boxes(self):
        from tracker import _batch_iou
        boxes = np.array([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=np.float32)
        iou = _batch_iou(boxes, boxes)
        np.testing.assert_allclose(iou.diagonal(), 1.0, atol=1e-5)

    def test_batch_iou_no_overlap(self):
        from tracker import _batch_iou
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[50, 50, 60, 60]], dtype=np.float32)
        assert _batch_iou(a, b)[0, 0] == pytest.approx(0.0)

    def test_batch_iou_partial_overlap(self):
        from tracker import _batch_iou
        a = np.array([[0, 0, 100, 100]], dtype=np.float32)
        b = np.array([[50, 0, 150, 100]], dtype=np.float32)
        iou = _batch_iou(a, b)[0, 0]
        assert 0.0 < iou < 1.0

    def test_centroid_distance(self):
        from tracker import _batch_centroid_distance
        a = np.array([[0, 0, 0, 0]], dtype=np.float32)  # centroid (0,0)
        b = np.array([[6, 8, 6, 8]], dtype=np.float32)  # centroid (6,8)
        dist = _batch_centroid_distance(a, b)[0, 0]
        assert dist == pytest.approx(10.0, abs=1e-3)


class TestCentroidTracker:
    def test_empty_detections_returns_empty(self):
        from tracker import CentroidTracker
        from edge_detector import DetectionResult
        t = CentroidTracker()
        assert t.update(DetectionResult()) == []

    def test_registers_new_tracks(self, synthetic_detections):
        from tracker import CentroidTracker
        t = CentroidTracker()
        tracked = t.update(synthetic_detections)
        assert len(tracked) == 2
        assert len({o.track_id for o in tracked}) == 2

    def test_ids_persist_across_frames(self, synthetic_detections):
        from tracker import CentroidTracker
        t = CentroidTracker()
        ids1 = {o.track_id for o in t.update(synthetic_detections)}
        ids2 = {o.track_id for o in t.update(synthetic_detections)}
        assert ids1 == ids2

    def test_track_dropped_after_disappearing(self):
        from tracker import CentroidTracker
        from edge_detector import DetectionResult
        t = CentroidTracker(max_disappeared=2)
        det = DetectionResult(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["object"],
        )
        t.update(det)
        assert len(t._tracks) == 1
        for _ in range(5):
            t.update(DetectionResult())
        assert len(t._tracks) == 0

    def test_build_tracker_centroid(self):
        from tracker import build_tracker, CentroidTracker
        assert isinstance(build_tracker("centroid"), CentroidTracker)

    def test_tracked_object_centroid(self, synthetic_detections):
        from tracker import CentroidTracker
        t = CentroidTracker()
        tracked = t.update(synthetic_detections)
        for obj in tracked:
            cx, cy = obj.centroid
            x1, y1, x2, y2 = obj.xyxy
            assert cx == pytest.approx((x1 + x2) / 2)
            assert cy == pytest.approx((y1 + y2) / 2)

    def test_track_age_increments(self, synthetic_detections):
        from tracker import CentroidTracker
        t = CentroidTracker()
        t.update(synthetic_detections)
        tracked = t.update(synthetic_detections)
        assert all(obj.age == 2 for obj in tracked)

    def test_unknown_tracker_raises(self):
        from tracker import build_tracker
        with pytest.raises(ValueError, match="Unknown tracker type"):
            build_tracker("nonexistent")


# ═════════════════════════════════════════════════════════════════════════════
# id_service.py
# ═════════════════════════════════════════════════════════════════════════════

class TestIdentificationCache:
    def test_miss_returns_none(self):
        from id_service import IdentificationCache
        assert IdentificationCache(30).get(999) is None

    def test_set_and_get(self):
        from id_service import IdentificationCache
        c = IdentificationCache(60)
        c.set(1, "person")
        assert c.get(1) == "person"

    def test_expired_returns_none(self):
        from id_service import IdentificationCache
        c = IdentificationCache(ttl_seconds=0.02)
        c.set(1, "car")
        time.sleep(0.05)
        assert c.get(1) is None

    def test_invalidate(self):
        from id_service import IdentificationCache
        c = IdentificationCache(60)
        c.set(5, "dog")
        c.invalidate(5)
        assert c.get(5) is None

    def test_evict_expired(self):
        from id_service import IdentificationCache
        c = IdentificationCache(ttl_seconds=0.02)
        c.set(1, "a")
        c.set(2, "b")
        time.sleep(0.05)
        removed = c.evict_expired()
        assert removed == 2


class TestPendingItemOrdering:
    def test_lower_priority_first(self):
        from id_service import PendingItem
        a = PendingItem(priority=1.0, track_id=1, frame=np.zeros((1, 1, 3), np.uint8))
        b = PendingItem(priority=2.0, track_id=2, frame=np.zeros((1, 1, 3), np.uint8))
        assert a < b


class TestNullIDService:
    """Tests for the graceful no-URL fallback introduced in this refactor."""

    def test_get_cached_always_none(self):
        from main import NullIdentificationService
        svc = NullIdentificationService()
        assert svc.get_cached(1) is None
        assert svc.get_cached(999) is None

    def test_submit_returns_false(self):
        from main import NullIdentificationService
        svc = NullIdentificationService()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert svc.submit(track_id=1, frame_with_box=frame) is False

    def test_queue_depth_is_zero(self):
        from main import NullIdentificationService
        assert NullIdentificationService().queue_depth() == 0

    def test_shutdown_is_safe(self):
        from main import NullIdentificationService
        NullIdentificationService().shutdown()   # must not raise

    def test_progress_is_empty_dict(self):
        from main import NullIdentificationService
        svc = NullIdentificationService()
        assert svc.progress == {}


class TestIDServiceRaisesWithoutURL:
    def test_raises_value_error_no_url(self):
        from id_service import IdentificationService
        orig = os.environ.pop("REMOTE_URL", None)
        try:
            with pytest.raises((ValueError, Exception)):
                IdentificationService(remote_url="", cache_ttl=10)
        finally:
            if orig is not None:
                os.environ["REMOTE_URL"] = orig


# ═════════════════════════════════════════════════════════════════════════════
# ui_overlay.py
# ═════════════════════════════════════════════════════════════════════════════

class TestUIOverlay:
    def test_draw_preserves_shape(self, synthetic_frame_720p):
        from ui_overlay import UIOverlay
        from tracker import TrackedObject
        overlay = UIOverlay()
        obj = TrackedObject(1, np.array([300, 200, 600, 400], np.float32), 0, "object", 0.9)
        result = overlay.draw(
            synthetic_frame_720p, [obj],
            {1: {"status": "done", "progress": 1.0, "label": "person", "error": None}},
            fps=25.0,
        )
        assert result.shape == synthetic_frame_720p.shape

    def test_draw_does_not_mutate_input(self, synthetic_frame_720p):
        from ui_overlay import UIOverlay
        original = synthetic_frame_720p.copy()
        UIOverlay().draw(synthetic_frame_720p, [], {}, fps=30.0)
        np.testing.assert_array_equal(synthetic_frame_720p, original)

    def test_draw_identifying_status(self, synthetic_frame_720p):
        from ui_overlay import UIOverlay
        from tracker import TrackedObject
        obj = TrackedObject(2, np.array([100, 100, 300, 300], np.float32), 0, "object", 0.8)
        result = UIOverlay().draw(
            synthetic_frame_720p, [obj],
            {2: {"status": "identifying", "progress": 0.45, "label": None, "error": None}},
            fps=18.0,
        )
        assert result is not None

    def test_draw_error_status(self, synthetic_frame_720p):
        from ui_overlay import UIOverlay
        from tracker import TrackedObject
        obj = TrackedObject(3, np.array([50, 50, 200, 200], np.float32), 0, "object", 0.5)
        result = UIOverlay().draw(
            synthetic_frame_720p, [obj],
            {3: {"status": "error", "progress": 1.0, "label": None, "error": "timeout"}},
            fps=10.0,
        )
        assert result is not None

    def test_draw_no_tracks(self, synthetic_frame_720p):
        """draw() with empty track list must not crash."""
        result = __import__("ui_overlay").UIOverlay().draw(
            synthetic_frame_720p, [], {}, fps=0.0
        )
        assert result is not None

    def test_color_cycles_and_valid(self):
        from ui_overlay import _color
        seen = set()
        for i in range(100):
            c = _color(i)
            assert len(c) == 3
            assert all(0 <= v <= 255 for v in c)
            seen.add(c)
        assert len(seen) > 1   # palette has multiple colours


# ═════════════════════════════════════════════════════════════════════════════
# server.py — parse_noun (offline unit test, no model needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestParseNoun:
    """_parse_noun cleans raw LLM output into a tidy noun phrase."""

    def _p(self, text: str) -> str:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "remote_server"))
        from server import _parse_noun
        return _parse_noun(text)

    def test_strips_whitespace(self):
        assert self._p("  person  ") == "person"

    def test_strips_quotes(self):
        assert self._p('"car"') == "car"

    def test_takes_first_line(self):
        assert self._p("dog\ncat\nbird") == "dog"

    def test_removes_trailing_punctuation(self):
        assert self._p("chair.") == "chair"

    def test_strips_preamble(self):
        assert self._p("it is a cat") == "cat"
        assert self._p("this is an apple") == "apple"

    def test_empty_string_returns_object(self):
        assert self._p("") == "object"

    def test_too_many_words_returns_object(self):
        assert self._p("a very long complicated description here") == "object"

    def test_two_word_noun(self):
        assert self._p("coffee mug") == "coffee mug"

    def test_lowercase(self):
        assert self._p("LAPTOP") == "laptop"


# ═════════════════════════════════════════════════════════════════════════════
# main.py — draw_frame_with_single_box
# ═════════════════════════════════════════════════════════════════════════════

class TestDrawSingleBox:
    def test_output_same_shape(self, synthetic_frame_720p):
        from main import draw_frame_with_single_box
        box = np.array([100, 100, 400, 300], dtype=np.float32)
        out = draw_frame_with_single_box(synthetic_frame_720p, box)
        assert out.shape == synthetic_frame_720p.shape

    def test_does_not_mutate_input(self, synthetic_frame_720p):
        from main import draw_frame_with_single_box
        original = synthetic_frame_720p.copy()
        draw_frame_with_single_box(synthetic_frame_720p,
                                   np.array([10, 10, 200, 200], np.float32))
        np.testing.assert_array_equal(synthetic_frame_720p, original)

    def test_green_box_default_colour(self):
        """Default box colour must be green to match the server prompt."""
        from main import draw_frame_with_single_box
        import inspect
        sig = inspect.signature(draw_frame_with_single_box)
        default_color = sig.parameters["box_color"].default
        assert default_color == (0, 255, 0), (
            "Box colour must be green (0,255,0) — server prompt says 'green bounding box'"
        )
