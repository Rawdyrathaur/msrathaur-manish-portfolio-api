"""
Pytest configuration and fixtures for testing the portfolio backend.
This file provides:
- Mock embeddings and LLM providers
- FastAPI test client
- Temporary test knowledge base
- Environment setup
"""

import pytest
import os
import sys
import tempfile
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Load test environment
test_env_path = backend_path / "tests" / ".env.test"
if test_env_path.exists():
    load_dotenv(str(test_env_path), override=True)


# ══════════════════════════════════════════════════════════
# TEMPORARY TEST KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════

@pytest.fixture
def test_knowledge_dir():
    """Creates temporary folder with 3 test markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test markdown files
        test_files = {
            "about.md": """# About Manish

## Skills
- Python, FastAPI, Django
- JavaScript, React, Vue
- AWS, Docker, Kubernetes

## Experience
5 years as full-stack developer building web applications.""",
            
            "projects.md": """# Projects

## Portfolio App
Built an AI chatbot using FastAPI and RAG (Retrieval-Augmented Generation).
Features: Multi-provider LLM fallback, voice I/O, rate limiting.

## Open Source
Contributed to several open-source Python projects.""",
            
            "blogs.md": """# Blog Posts

## Post 1: Understanding RAG Systems
Deep dive into Retrieval-Augmented Generation and why it matters.

## Post 2: FastAPI Best Practices
Lessons learned building production APIs with FastAPI."""
        }
        
        for filename, content in test_files.items():
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(content)
        
        yield tmpdir


@pytest.fixture
def empty_knowledge_dir():
    """Creates empty temporary directory for edge case testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ══════════════════════════════════════════════════════════
# MOCKED DEPENDENCIES
# ══════════════════════════════════════════════════════════

@pytest.fixture
def mock_embedder(monkeypatch):
    """Mock SentenceTransformer to avoid loading real model."""
    mock = MagicMock()
    # Return fixed 384-dim embeddings for consistent testing
    mock.encode = MagicMock(return_value=[[0.1] * 384] * 10)
    monkeypatch.setattr("rag.SentenceTransformer", lambda x: mock)
    return mock


@pytest.fixture
def mock_chromadb(monkeypatch):
    """Mock ChromaDB for isolated testing."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    
    # Simulate a simple in-memory document store
    documents_store = {"ids": [], "documents": [], "metadatas": []}
    
    def mock_add(**kwargs):
        documents_store["ids"].extend(kwargs.get("ids", []))
        documents_store["documents"].extend(kwargs.get("documents", []))
        documents_store["metadatas"].extend(kwargs.get("metadatas", []))
    
    def mock_query(**kwargs):
        return {"documents": [["Relevant chunk 1"], ["Relevant chunk 2"]]}
    
    mock_collection.add = mock_add
    mock_collection.query = mock_query
    mock_collection.count = MagicMock(return_value=len(documents_store["ids"]))
    
    monkeypatch.setattr("rag._client", mock_client)
    monkeypatch.setattr("rag._collection", mock_collection)
    return mock_collection


@pytest.fixture
def mock_groq(monkeypatch):
    """Mock Groq API."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response from Groq"))]
    mock.chat.completions.create.return_value = mock_response
    monkeypatch.setattr("main.Groq", lambda **kwargs: mock)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-123")
    return mock


@pytest.fixture
def mock_gemini(monkeypatch):
    """Mock Google Gemini API."""
    import google.generativeai as genai
    mock = MagicMock()
    mock.text = "Test response from Gemini"
    
    mock_model = MagicMock()
    mock_chat = MagicMock()
    mock_chat.send_message.return_value = mock
    mock_model.start_chat.return_value = mock_chat
    
    monkeypatch.setattr("main.genai.GenerativeModel", lambda **kwargs: mock_model)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-123")
    return mock


@pytest.fixture
def mock_edge_tts(monkeypatch):
    """Mock Microsoft Edge TTS."""
    mock_communicate = AsyncMock()
    mock_communicate.stream = AsyncMock(return_value=[
        {"type": "audio", "data": b"fake_mp3_audio_data"},
    ])
    monkeypatch.setattr("main.edge_tts.Communicate", lambda text, voice, rate: mock_communicate)
    return mock_communicate


# ══════════════════════════════════════════════════════════
# FASTAPI TEST CLIENT
# ══════════════════════════════════════════════════════════

@pytest.fixture
def client(mock_embedder, mock_chromadb, monkeypatch):
    """
    Create FastAPI TestClient with mocked dependencies.
    Automatically mocks ChromaDB and embedder to avoid real I/O.
    """
    from fastapi.testclient import TestClient
    from main import app
    
    # Set test environment variables
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "100")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("RAG_RELOAD_SECRET", "test-reload-secret")
    
    return TestClient(app)


# ══════════════════════════════════════════════════════════
# TEST ENVIRONMENT
# ══════════════════════════════════════════════════════════

@pytest.fixture
def test_env(monkeypatch):
    """Set all test environment variables."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "100")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("RAG_RELOAD_SECRET", "test-reload-secret")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
