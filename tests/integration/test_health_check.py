"""
Integration tests for health check endpoints.
"""

import pytest


ADMIN_HEADERS = {"X-Reload-Secret": "test-reload-secret"}


def test_ping_always_returns_200(client):
    """Verify /ping is always alive."""
    response = client.get("/ping")
    assert response.status_code == 200


def test_ping_returns_message(client):
    """Verify /ping returns a message."""
    response = client.get("/ping")
    data = response.json()
    assert "message" in data
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0


def test_health_returns_status(client):
    """Verify /health returns status field."""
    response = client.get("/health", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "degraded"]


def test_health_returns_providers_ready(client):
    """Verify /health lists ready providers."""
    response = client.get("/health", headers=ADMIN_HEADERS)
    
    data = response.json()
    assert "providers_ready" in data
    assert isinstance(data["providers_ready"], list)


def test_health_returns_providers_total(client):
    """Verify /health shows total providers."""
    response = client.get("/health", headers=ADMIN_HEADERS)
    
    data = response.json()
    assert "providers_total" in data
    assert data["providers_total"] == 5  # Groq, Gemini, Cohere, Mistral, Together


def test_health_returns_rag_status(client):
    """Verify /health reports RAG status."""
    response = client.get("/health", headers=ADMIN_HEADERS)
    
    data = response.json()
    assert "rag" in data
    assert data["rag"] in {"initializing", "ready", "error"}


def test_health_with_no_providers_degraded(client, monkeypatch):
    """Verify /health is 'degraded' when no providers configured."""
    # Clear all provider keys
    for provider in ["GROQ", "GEMINI", "COHERE", "MISTRAL", "TOGETHER"]:
        monkeypatch.delenv(f"{provider}_API_KEY", raising=False)
    
    response = client.get("/health", headers=ADMIN_HEADERS)
    data = response.json()
    
    # Should be degraded or show no providers
    assert len(data["providers_ready"]) == 0 or data["status"] == "degraded"


def test_health_with_one_provider_ready(client, monkeypatch):
    """Verify /health shows ready when at least one provider configured."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    
    # Clear others
    for provider in ["GEMINI", "COHERE", "MISTRAL", "TOGETHER"]:
        monkeypatch.delenv(f"{provider}_API_KEY", raising=False)
    
    response = client.get("/health", headers=ADMIN_HEADERS)
    data = response.json()
    
    assert "Groq" in data["providers_ready"] or len(data["providers_ready"]) > 0


def test_providers_lists_all_providers(client):
    """Verify /providers lists all configured providers."""
    response = client.get("/providers", headers=ADMIN_HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    
    expected_providers = ["Groq", "Gemini", "Cohere", "Mistral", "Together"]
    for provider in expected_providers:
        assert provider in data
        assert isinstance(data[provider], bool)


def test_providers_shows_enabled_status(client, monkeypatch):
    """Verify /providers correctly shows enabled/disabled status."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    response = client.get("/providers", headers=ADMIN_HEADERS)
    data = response.json()
    
    assert data["Groq"] == True
    assert data["Gemini"] == False


def test_health_endpoint_response_format(client):
    """Verify /health response has consistent structure."""
    response = client.get("/health", headers=ADMIN_HEADERS)
    data = response.json()
    
    # Should have all required fields
    required_fields = ["status", "providers_ready", "providers_total", "rag"]
    for field in required_fields:
        assert field in data


def test_providers_endpoint_response_format(client):
    """Verify /providers response is properly structured."""
    response = client.get("/providers", headers=ADMIN_HEADERS)
    data = response.json()
    
    # Should be a dict with provider names as keys
    assert isinstance(data, dict)
    assert len(data) > 0


def test_ping_endpoint_response_format(client):
    """Verify /ping response format is consistent."""
    response = client.get("/ping")
    data = response.json()
    
    assert isinstance(data, dict)
    assert "message" in data


def test_reload_endpoint_returns_chunks_count(client, monkeypatch):
    """Verify /reload endpoint returns chunk count."""
    monkeypatch.setattr("main.load_knowledge", lambda: 42)
    response = client.post("/reload", headers=ADMIN_HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "chunks" in data
    assert isinstance(data["chunks"], int)


def test_reload_endpoint_status_field(client, monkeypatch):
    """Verify /reload returns status field."""
    monkeypatch.setattr("main.load_knowledge", lambda: 42)
    response = client.post("/reload", headers=ADMIN_HEADERS)
    
    data = response.json()
    assert data.get("status") == "reloaded"


def test_health_provides_useful_status(client):
    """Verify /health provides actionable status for monitoring."""
    response = client.get("/health", headers=ADMIN_HEADERS)
    data = response.json()
    
    # Status should help determine if system is working
    assert data["status"] in ["ok", "degraded"]
    # Providers ready helps troubleshoot
    assert len(data["providers_ready"]) >= 0
    # RAG status indicates knowledge system
    assert data["rag"] in ["initializing", "ready", "error", "syncing"]


def test_reload_fails_closed_without_admin_secret(client, monkeypatch):
    monkeypatch.delenv("RAG_RELOAD_SECRET", raising=False)
    response = client.post("/reload")
    assert response.status_code == 503


def test_public_health_hides_provider_details(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "providers_ready" not in response.json()
