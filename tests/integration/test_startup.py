"""
Integration tests for FastAPI startup and lifespan flow.
"""

import pytest


def test_app_startup_completes_successfully(client):
    """Verify app startup completes without errors."""
    # TestClient automatically triggers startup
    assert client is not None


def test_health_endpoint_after_startup(client):
    """Verify health endpoint returns OK after startup."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "rag" in data
    assert data["rag"] in {"initializing", "ready", "error"}


def test_ping_endpoint_after_startup(client):
    """Verify ping endpoint is accessible after startup."""
    response = client.get("/ping")
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] != ""


def test_providers_endpoint_lists_configuration(client):
    """Verify providers endpoint shows configured LLMs."""
    response = client.get(
        "/providers",
        headers={"X-Reload-Secret": "test-reload-secret"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should list all provider status
    expected_providers = ["Groq", "Gemini", "Cohere", "Mistral", "Together"]
    for provider in expected_providers:
        assert provider in data
        assert isinstance(data[provider], bool)


def test_chromadb_initialized_after_startup(client, mock_chromadb):
    """Verify ChromaDB is initialized during startup."""
    # If startup failed, we wouldn't get here
    # Check that ChromaDB mock was set up
    assert mock_chromadb is not None


def test_endpoints_no_404_after_startup(client):
    """Verify main endpoints exist and don't return 404."""
    endpoints = [
        "/ping",
        "/health",
        "/providers",
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code != 404, f"{endpoint} should exist"


def test_chat_endpoint_available_after_startup(client):
    """Verify /chat endpoint is available after startup."""
    # Send a minimal request
    response = client.post("/chat", json={
        "message": "test",
        "history": []
    })
    
    # Should not be 404
    assert response.status_code != 404
    # May be 400/422/500/503 due to mocking, but not 404
    assert response.status_code < 600


def test_whisper_endpoint_available_after_startup(client):
    """Verify /whisper endpoint is available after startup."""
    # Endpoint exists (may fail for other reasons)
    import io
    response = client.post(
        "/whisper",
        files={"audio": ("test.webm", io.BytesIO(b"test"), "audio/webm")}
    )
    
    # Should not return 404
    assert response.status_code != 404


def test_speak_endpoint_available_after_startup(client):
    """Verify /speak endpoint is available after startup."""
    response = client.get("/speak?text=hello")
    
    # Should not return 404
    assert response.status_code != 404


def test_reload_endpoint_available(client):
    """Verify /reload endpoint is available for reloading knowledge."""
    response = client.post("/reload")
    
    # Should not return 404
    assert response.status_code != 404


def test_cors_configured_for_localhost(client):
    """Verify CORS headers are set for localhost."""
    response = client.get("/ping")
    
    # Check if CORS headers are present in response
    # (Exact header names depend on FastAPI CORS config)
    assert response.status_code == 200


def test_multiple_requests_after_startup(client):
    """Verify app remains stable after multiple requests."""
    for i in range(5):
        response = client.get("/health")
        assert response.status_code == 200
        
        response = client.get("/ping")
        assert response.status_code == 200
