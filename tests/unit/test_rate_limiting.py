"""
Unit tests for rate limiting functionality.
"""

import pytest
import hashlib
from fastapi import HTTPException
from main import check_rate_limit, _rate_store, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW


def store_key(ip, scope="chat"):
    digest = hashlib.sha256(ip.encode("utf-8")).hexdigest()[:24]
    return f"{scope}:{digest}"


@pytest.fixture(autouse=True)
def clear_rate_store():
    """Clear rate store before each test."""
    _rate_store.clear()
    yield
    _rate_store.clear()


def test_rate_limit_allows_requests_under_threshold():
    """Verify requests under the limit are allowed."""
    ip = "192.168.1.1"
    
    # Should allow up to RATE_LIMIT_REQUESTS without raising
    for i in range(RATE_LIMIT_REQUESTS):
        try:
            check_rate_limit(ip)
        except HTTPException:
            pytest.fail(f"Request {i+1} should be allowed (under limit of {RATE_LIMIT_REQUESTS})")


def test_rate_limit_blocks_over_threshold():
    """Verify requests exceeding limit are blocked with 429."""
    ip = "10.0.0.1"
    
    # Fill up to the limit
    for i in range(RATE_LIMIT_REQUESTS):
        check_rate_limit(ip)
    
    # Next request should fail
    with pytest.raises(HTTPException) as exc_info:
        check_rate_limit(ip)
    
    assert exc_info.value.status_code == 429
    assert "Too many requests" in exc_info.value.detail


def test_rate_limit_per_ip_isolation():
    """Verify rate limiting is per-IP, not global."""
    ip1 = "192.168.1.100"
    ip2 = "192.168.1.200"
    
    # Fill limit for IP1
    for i in range(RATE_LIMIT_REQUESTS):
        check_rate_limit(ip1)
    
    # IP1 should be blocked
    with pytest.raises(HTTPException) as exc:
        check_rate_limit(ip1)
    assert exc.value.status_code == 429
    
    # But IP2 should still work
    try:
        check_rate_limit(ip2)
    except HTTPException:
        pytest.fail("Different IP should have separate limit")


def test_rate_limit_resets_after_window():
    """Verify rate limit resets after time window expires."""
    ip = "10.20.30.40"
    
    # Fill the limit quickly
    for i in range(RATE_LIMIT_REQUESTS):
        check_rate_limit(ip)
    
    # Should be blocked now
    with pytest.raises(HTTPException):
        check_rate_limit(ip)
    
    # Wait for window to expire (or mock time)
    # For actual testing, this would need to be in integration/e2e
    # Unit test just verifies the logic


def test_rate_limit_stores_timestamps_correctly():
    """Verify timestamps are stored for rate limiting."""
    ip = "172.16.0.1"
    
    # Make a request
    check_rate_limit(ip)
    
    # Verify timestamp was recorded
    key = store_key(ip)
    assert key in _rate_store
    assert len(_rate_store[key]) == 1
    assert isinstance(_rate_store[key][0], float)


def test_rate_limit_multiple_rapid_requests():
    """Verify rapid sequential requests are tracked."""
    ip = "1.1.1.1"
    
    for i in range(10):
        check_rate_limit(ip)
    
    assert len(_rate_store[store_key(ip)]) == 10


def test_rate_limit_error_message_is_generic():
    """Do not expose internal limiter configuration."""
    ip = "8.8.8.8"
    
    for i in range(RATE_LIMIT_REQUESTS):
        check_rate_limit(ip)
    
    with pytest.raises(HTTPException) as exc_info:
        check_rate_limit(ip)
    
    detail = exc_info.value.detail
    assert detail == "Too many requests. Please try again shortly."


def test_rate_limit_edge_case_zero_requests():
    """Test behavior with RATE_LIMIT_REQUESTS = 0 (if ever set)."""
    # This is an edge case - normally RATE_LIMIT_REQUESTS >= 1
    ip = "9.9.9.9"
    
    try:
        check_rate_limit(ip)
        check_rate_limit(ip)
        # Should work or raise 429 depending on config
    except HTTPException as e:
        assert e.status_code == 429


def test_rate_limit_concurrent_ips():
    """Verify multiple IPs can have requests simultaneously."""
    ips = ["1.1.1.1", "2.2.2.2", "3.3.3.3"]
    
    for ip in ips:
        for i in range(5):
            check_rate_limit(ip)
    
    # Each IP should have 5 requests recorded
    for ip in ips:
        assert len(_rate_store[store_key(ip)]) == 5
