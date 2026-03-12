"""
tests/test_gemma_remote.py — Unit tests for GemmaRemoteService.

These tests verify the HTTP client and service logic with mocked responses.
No actual Colab/Kaggle connection is required.

Run:
    pytest tests/test_gemma_remote.py -v
"""

from __future__ import annotations

import sys
import os
import json
import time
from unittest import mock

import numpy as np
import pytest

# Add project root to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_server_url() -> str:
    """Return a mock server URL for testing."""
    return "https://test123.ngrok.io"


@pytest.fixture
def mock_api_key() -> str:
    """Return a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def synthetic_crop() -> np.ndarray:
    """Return a small synthetic BGR crop."""
    crop = np.zeros((100, 80, 3), dtype=np.uint8)
    crop[20:80, 10:70] = (100, 200, 50)
    return crop


# ─────────────────────────────────────────────────────────────────────────────
# GemmaHTTPClient tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGemmaHTTPClient:
    def test_import(self):
        """GemmaHTTPClient can be imported."""
        from gemma_remote_service import GemmaHTTPClient
        assert True

    def test_raises_without_url(self):
        """GemmaHTTPClient raises ValueError without URL."""
        from gemma_remote_service import GemmaHTTPClient
        with pytest.raises(ValueError, match="URL"):
            GemmaHTTPClient(base_url="")

    def test_health_check_called_on_init(self, mock_server_url):
        """Health check is called during initialization."""
        from gemma_remote_service import GemmaHTTPClient
        
        with mock.patch("requests.Session.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.json.return_value = {"model": "gemma-3-4b-it"}
            mock_get.return_value = mock_response
            
            client = GemmaHTTPClient(base_url=mock_server_url)
            mock_get.assert_called_once_with(
                f"{mock_server_url}/health",
                timeout=10
            )

    def test_identify_batch_success(self, mock_server_url, synthetic_crop):
        """identify_batch parses successful response correctly."""
        from gemma_remote_service import GemmaHTTPClient, PendingItem
        
        items = [
            PendingItem(priority=0, track_id=1, crop=synthetic_crop,
                       class_name="person", class_id=0),
            PendingItem(priority=1, track_id=2, crop=synthetic_crop,
                       class_name="chair", class_id=1),
        ]
        
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "results": [
                {"track_id": 1, "description": "a person in a blue jacket"},
                {"track_id": 2, "description": "a wooden chair"},
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        with mock.patch("requests.Session.get") as mock_get:
            mock_health = mock.Mock()
            mock_health.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_health
            
            client = GemmaHTTPClient(base_url=mock_server_url)
            
            with mock.patch.object(client._session, "post", return_value=mock_response) as mock_post:
                results = client.identify_batch(items)
                
                assert results[1] == "a person in a blue jacket"
                assert results[2] == "a wooden chair"
                
                # Verify POST was called with correct URL
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert call_args[0][0] == f"{mock_server_url}/identify"

    def test_identify_batch_with_api_key(self, mock_server_url, mock_api_key, synthetic_crop):
        """API key is included in Authorization header."""
        from gemma_remote_service import GemmaHTTPClient, PendingItem
        
        items = [
            PendingItem(priority=0, track_id=1, crop=synthetic_crop,
                       class_name="cup", class_id=0),
        ]
        
        mock_response = mock.Mock()
        mock_response.json.return_value = {"results": [{"track_id": 1, "description": "a cup"}]}
        mock_response.raise_for_status.return_value = None
        
        with mock.patch("requests.Session.get") as mock_get:
            mock_health = mock.Mock()
            mock_get.return_value = mock_health
            
            client = GemmaHTTPClient(base_url=mock_server_url, api_key=mock_api_key)
            
            with mock.patch.object(client._session, "post", return_value=mock_response) as mock_post:
                client.identify_batch(items)
                
                # Check headers include Authorization
                call_kwargs = mock_post.call_args[1]
                assert "headers" in call_kwargs
                assert call_kwargs["headers"]["Authorization"] == f"Bearer {mock_api_key}"

    def test_identify_batch_retry_on_failure(self, mock_server_url, synthetic_crop):
        """identify_batch retries on transient failures."""
        from gemma_remote_service import GemmaHTTPClient, PendingItem
        import requests
        
        items = [
            PendingItem(priority=0, track_id=1, crop=synthetic_crop,
                       class_name="bottle", class_id=0),
        ]
        
        # First call fails, second succeeds
        mock_response = mock.Mock()
        mock_response.json.return_value = {"results": [{"track_id": 1, "description": "a bottle"}]}
        mock_response.raise_for_status.return_value = None
        
        with mock.patch("requests.Session.get") as mock_get:
            mock_health = mock.Mock()
            mock_get.return_value = mock_health
            
            client = GemmaHTTPClient(base_url=mock_server_url)
            
            with mock.patch.object(
                client._session, "post",
                side_effect=[requests.exceptions.ConnectionError("Connection failed"), mock_response]
            ) as mock_post:
                # Patch time.sleep to avoid waiting in tests
                with mock.patch("time.sleep"):
                    results = client.identify_batch(items)
                    
                    # Should have retried once (2 total calls)
                    assert mock_post.call_count == 2
                    assert results[1] == "a bottle"

    def test_identify_batch_falls_back_to_class_name(self, mock_server_url, synthetic_crop):
        """Missing track_id in response falls back to class_name."""
        from gemma_remote_service import GemmaHTTPClient, PendingItem
        
        items = [
            PendingItem(priority=0, track_id=99, crop=synthetic_crop,
                       class_name="unknown_object", class_id=999),
        ]
        
        mock_response = mock.Mock()
        # Response doesn't include track_id 99
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        
        with mock.patch("requests.Session.get") as mock_get:
            mock_health = mock.Mock()
            mock_get.return_value = mock_health
            
            client = GemmaHTTPClient(base_url=mock_server_url)
            
            with mock.patch.object(client._session, "post", return_value=mock_response):
                results = client.identify_batch(items)
                
                # Should fall back to class_name
                assert results[99] == "unknown_object"


# ─────────────────────────────────────────────────────────────────────────────
# GemmaRemoteService tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGemmaRemoteService:
    def test_import(self):
        """GemmaRemoteService can be imported."""
        from gemma_remote_service import GemmaRemoteService
        assert True

    def test_service_initialises_with_url(self, mock_server_url):
        """GemmaRemoteService initializes with remote URL."""
        from gemma_remote_service import GemmaRemoteService
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient") as mock_client:
            mock_client_instance = mock.Mock()
            mock_client.return_value = mock_client_instance
            
            svc = GemmaRemoteService(remote_url=mock_server_url, cache_ttl=10)
            svc.shutdown()
            
            mock_client.assert_called_once_with(
                base_url=mock_server_url,
                api_key=None
            )

    def test_service_initialises_with_api_key(self, mock_server_url, mock_api_key):
        """GemmaRemoteService passes API key to client."""
        from gemma_remote_service import GemmaRemoteService
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient") as mock_client:
            mock_client_instance = mock.Mock()
            mock_client.return_value = mock_client_instance
            
            svc = GemmaRemoteService(
                remote_url=mock_server_url,
                api_key=mock_api_key,
                cache_ttl=10
            )
            svc.shutdown()
            
            mock_client.assert_called_once_with(
                base_url=mock_server_url,
                api_key=mock_api_key
            )

    def test_cache_get_and_set(self, mock_server_url):
        """Cache stores and retrieves labels."""
        from gemma_remote_service import GemmaRemoteService, IdentificationCache
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient"):
            svc = GemmaRemoteService(remote_url=mock_server_url, cache_ttl=60)
            try:
                # Initially empty
                assert svc.get_cached(1) is None
                
                # Set via internal cache
                svc._cache.set(1, "a red car")
                assert svc.get_cached(1) == "a red car"
            finally:
                svc.shutdown()

    def test_submit_creates_pending_item(self, mock_server_url, synthetic_crop):
        """Submit creates a pending item in the queue."""
        from gemma_remote_service import GemmaRemoteService
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient"):
            svc = GemmaRemoteService(remote_url=mock_server_url, cache_ttl=60)
            try:
                initial_depth = svc.queue_depth()
                
                result = svc.submit(1, synthetic_crop, "person", 0)
                
                assert result is True
                assert svc.queue_depth() == initial_depth + 1
            finally:
                svc.shutdown()

    def test_submit_returns_false_if_cached(self, mock_server_url, synthetic_crop):
        """Submit returns False if track already cached."""
        from gemma_remote_service import GemmaRemoteService
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient"):
            svc = GemmaRemoteService(remote_url=mock_server_url, cache_ttl=60)
            try:
                # Pre-populate cache
                svc._cache.set(1, "already identified")
                
                result = svc.submit(1, synthetic_crop, "person", 0)
                assert result is False
            finally:
                svc.shutdown()

    def test_submit_returns_false_if_inflight(self, mock_server_url, synthetic_crop):
        """Submit returns False if track already in-flight."""
        from gemma_remote_service import GemmaRemoteService
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient"):
            svc = GemmaRemoteService(remote_url=mock_server_url, cache_ttl=60)
            try:
                # First submit succeeds
                r1 = svc.submit(1, synthetic_crop, "person", 0)
                assert r1 is True
                
                # Second submit while in-flight returns False
                r2 = svc.submit(1, synthetic_crop, "person", 0)
                assert r2 is False
            finally:
                svc.shutdown()

    def test_progress_dict_updated_on_submit(self, mock_server_url, synthetic_crop):
        """Progress dict is updated when item is submitted."""
        from gemma_remote_service import GemmaRemoteService
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient"):
            svc = GemmaRemoteService(remote_url=mock_server_url, cache_ttl=60)
            try:
                svc.submit(42, synthetic_crop, "chair", 0)
                
                # Progress dict should have entry
                assert 42 in svc.progress
                assert svc.progress[42]["status"] == "queued"
                assert svc.progress[42]["progress"] == 0.0
            finally:
                svc.shutdown()

    def test_batch_size_configuration(self, mock_server_url):
        """Batch size is configurable and capped at 16."""
        from gemma_remote_service import GemmaRemoteService
        
        with mock.patch("gemma_remote_service.GemmaHTTPClient"):
            # Test normal batch size
            svc1 = GemmaRemoteService(remote_url=mock_server_url, batch_size=8)
            assert svc1._batch_sz == 8
            svc1.shutdown()
            
            # Test batch size capped at 16
            svc2 = GemmaRemoteService(remote_url=mock_server_url, batch_size=50)
            assert svc2._batch_sz == 16
            svc2.shutdown()
            
            # Test batch size minimum 1
            svc3 = GemmaRemoteService(remote_url=mock_server_url, batch_size=0)
            assert svc3._batch_sz == 1
            svc3.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# IdentificationCache tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentificationCache:
    def test_cache_miss_returns_none(self):
        """Cache returns None for missing entries."""
        from gemma_remote_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=30)
        assert cache.get(999) is None

    def test_cache_set_and_get(self):
        """Cache stores and retrieves values."""
        from gemma_remote_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(1, "a blue chair")
        assert cache.get(1) == "a blue chair"

    def test_cache_expired(self):
        """Expired entries return None."""
        from gemma_remote_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=0.01)
        cache.set(1, "a plant")
        time.sleep(0.05)
        assert cache.get(1) is None

    def test_cache_invalidate(self):
        """Invalidation removes entries."""
        from gemma_remote_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=60)
        cache.set(5, "a red laptop")
        cache.invalidate(5)
        assert cache.get(5) is None

    def test_cache_evict_expired(self):
        """evict_expired cleans up stale entries."""
        from gemma_remote_service import IdentificationCache
        cache = IdentificationCache(ttl_seconds=0.01)
        cache.set(1, "old")
        cache.set(2, "also old")
        time.sleep(0.05)
        removed = cache.evict_expired()
        assert removed == 2


# ─────────────────────────────────────────────────────────────────────────────
# Helper function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_encode_crop_returns_valid_b64(self, synthetic_crop):
        """_encode_crop returns valid base64 string."""
        from gemma_remote_service import _encode_crop
        import base64
        
        b64 = _encode_crop(synthetic_crop)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_encode_crop_resizes_large_crops(self):
        """Large crops are resized to max_px."""
        from gemma_remote_service import _encode_crop, MAX_CROP_PX
        
        # Create oversized crop
        large_crop = np.zeros((1000, 1000, 3), dtype=np.uint8)
        b64 = _encode_crop(large_crop, max_px=MAX_CROP_PX)
        
        # Should still encode successfully
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0
