# 🧪 QA Testing Guide for Portfolio Backend

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx

# Run all tests
pytest

# Run with coverage report
pytest --cov --cov-report=html

# Run specific test suite
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/e2e/          # End-to-end tests only

# Run single test file
pytest tests/unit/test_rag.py -v

# Run single test
pytest tests/unit/test_rag.py::test_chunk_markdown_splits_at_double_hash_headings -v

# Run and stop on first failure
pytest -x

# Run with print statements visible
pytest -s
```

## Test Structure

```
backend/tests/
├── __init__.py
├── conftest.py                  # Pytest fixtures & mocks
├── pytest.ini                   # Pytest configuration
├── .env.test                    # Test environment variables
│
├── unit/                        # Fast, isolated unit tests
│   ├── test_rag.py             # Markdown chunking, embeddings
│   ├── test_system_prompt.py   # Bot identity & rules
│   └── test_rate_limiting.py   # Rate limiter logic
│
├── integration/                 # Test with real FastAPI client
│   ├── test_startup.py          # App startup & lifespan
│   ├── test_chat_endpoint.py    # /chat endpoint
│   ├── test_audio.py            # /whisper & /speak
│   └── test_health_check.py     # Health/ping/providers endpoints
│
└── e2e/                         # Full workflow tests
    └── test_full_workflow.py    # Startup → Chat → Response
```

## Test Coverage

| Component | Unit | Integration | E2E |
|-----------|------|-------------|-----|
| RAG Chunking | ✅ | ✅ | ✅ |
| Embeddings | ✅ | - | ✅ |
| System Prompt | ✅ | ✅ | ✅ |
| Rate Limiting | ✅ | ✅ | ✅ |
| Chat Endpoint | - | ✅ | ✅ |
| Audio (STT/TTS) | - | ✅ | ✅ |
| Health Endpoints | - | ✅ | ✅ |
| LLM Fallback | - | ✅ | ✅ |
| Error Handling | ✅ | ✅ | ✅ |

## Running Tests by Category

### Unit Tests (Fast - ~1-2 seconds)
```bash
pytest tests/unit/ -v
# These test isolated components without external dependencies
```

### Integration Tests (Medium - ~5-10 seconds)
```bash
pytest tests/integration/ -v
# These test API endpoints with mocked external services
```

### E2E Tests (Slower - ~10-20 seconds)
```bash
pytest tests/e2e/ -v
# These test full workflows end-to-end
```

### All Tests with Coverage
```bash
pytest --cov --cov-report=html
# Open htmlcov/index.html in browser to see coverage report
```

## Mocking Strategy

### What's Mocked
- ✅ **SentenceTransformer** - Uses fixed embeddings (no model loading)
- ✅ **ChromaDB** - In-memory mock for fast testing
- ✅ **Groq API** - Mocked responses
- ✅ **Gemini API** - Mocked responses
- ✅ **Edge TTS** - Mocked audio stream

### Why Mocking?
- **Speed**: No API calls, no model downloads
- **Cost**: Free testing (no API credits used)
- **Reliability**: No external service dependencies
- **Isolation**: Tests don't affect real data

## Understanding Fixtures (conftest.py)

### `client` - FastAPI Test Client
```python
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
```

### `test_knowledge_dir` - Temporary Knowledge Files
```python
def test_loading_knowledge(client, test_knowledge_dir):
    # test_knowledge_dir contains 3 sample .md files
```

### `mock_chromadb` - Mocked Vector Database
```python
def test_rag_retrieval(client, mock_chromadb):
    # ChromaDB calls don't hit real database
```

### `test_env` - Test Environment Variables
```python
def test_with_env(client, test_env):
    # All API keys are set to test values
```

## Common Test Patterns

### Testing an Endpoint
```python
def test_chat_endpoint_success(client):
    response = client.post("/chat", json={
        "message": "What are your skills?",
        "history": []
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "provider" in data
```

### Testing with Mocking
```python
def test_with_provider_failure(client, monkeypatch):
    # Mock Groq to fail
    monkeypatch.setattr("main.try_groq", lambda msgs: None)
    # Mock Gemini to succeed
    monkeypatch.setattr("main.try_gemini", lambda msgs: "Response")
    
    response = client.post("/chat", json={"message": "test", "history": []})
    assert response.status_code == 200
```

### Testing Exceptions
```python
def test_rate_limit_blocks_request(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    
    client.post("/chat", json={"message": "test", "history": []})
    
    with pytest.raises(HTTPException) as exc_info:
        client.post("/chat", json={"message": "test", "history": []})
    
    assert exc_info.value.status_code == 429
```

## Running Tests in CI/CD

### GitHub Actions
Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          cd backend
          pytest --cov --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Coverage Report

After running tests with coverage:

```bash
# Generate HTML report
pytest --cov --cov-report=html

# Open in browser
open htmlcov/index.html
```

The HTML report shows:
- Line-by-line coverage
- Uncovered code paths
- Coverage percentage per file
- Branch coverage (if/else paths)

## Debugging Failed Tests

### Print Debug Info
```bash
pytest -s tests/unit/test_rag.py  # Show print() statements
pytest -vv  # Very verbose output
```

### Run Single Test
```bash
pytest tests/unit/test_rag.py::test_chunk_markdown_splits_at_double_hash_headings -vv
```

### Drop into Debugger
```python
def test_something(client):
    import pdb; pdb.set_trace()  # Stops execution here
    response = client.get("/health")
```

### Check Mock Calls
```python
def test_mock_calls(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("main.Groq", mock)
    
    # ... run code that calls Groq ...
    
    print(mock.call_count)       # How many times called
    print(mock.call_args_list)   # Arguments for each call
    assert mock.called           # Verify was called
```

## Writing New Tests

### Test File Template
```python
"""
Tests for [component name].
Covers [what is being tested].
"""

import pytest


def test_specific_behavior(client):
    """Test description - what should happen."""
    # Setup
    payload = {"data": "value"}
    
    # Execute
    response = client.post("/endpoint", json=payload)
    
    # Assert
    assert response.status_code == 200
    assert response.json()["result"] is not None
```

### Naming Conventions
- File: `test_*.py` (e.g., `test_rag.py`)
- Function: `test_specific_behavior_description`
- Fixture: `descriptive_name` (e.g., `test_knowledge_dir`)

## Best Practices

✅ **Do:**
- One assertion per test (or related group)
- Descriptive test names that explain what's tested
- Use fixtures instead of hardcoding setup
- Mock external dependencies
- Test both success and failure cases

❌ **Don't:**
- Make real API calls (always mock)
- Write tests that depend on each other
- Use sleep/time.sleep() in tests
- Create files without temp directory
- Write huge tests - break into smaller ones

## Troubleshooting

### Tests timeout
```bash
pytest --timeout=10  # Add timeout plugin
```

### Module import errors
```bash
# Ensure backend is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Applications/github/portfolio/backend"
```

### Mock not working
```python
# Mock BEFORE importing the module
monkeypatch.setenv("VAR", "value")
from module import something  # Import after mocking
```

### Tests pass locally but fail in CI
- Check environment variables in CI
- Ensure Python version matches
- Check file permissions

## Next Steps

1. ✅ Run all tests: `pytest`
2. ✅ Check coverage: `pytest --cov`
3. ✅ Set up CI/CD with GitHub Actions
4. ✅ Add pre-commit hook to run tests before committing
5. ✅ Monitor coverage over time (aim for >80%)

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/advanced/testing-websockets/)
- [Python Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
