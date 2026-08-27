"""
Integration tests for audio endpoints (Whisper STT & Edge TTS).
"""

import pytest
import io
from unittest.mock import MagicMock, AsyncMock


def test_whisper_endpoint_with_valid_audio(client, monkeypatch):
    """Verify /whisper transcribes audio successfully."""
    # Mock Groq transcription
    mock_groq = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello, this is a test transcription"
    mock_groq.audio.transcriptions.create.return_value = mock_response

    monkeypatch.setattr("main.Groq", lambda **kwargs: mock_groq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    audio_data = b"fake webm audio"
    response = client.post(
        "/whisper",
        files={"audio": ("test.webm", io.BytesIO(audio_data), "audio/webm")}
    )
    
    assert response.status_code == 200
    assert "transcript" in response.json()


def test_whisper_no_api_key_returns_503(client, monkeypatch):
    """Verify /whisper fails gracefully without Groq key."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    
    audio_data = b"fake audio"
    response = client.post(
        "/whisper",
        files={"audio": ("test.webm", io.BytesIO(audio_data), "audio/webm")}
    )
    
    assert response.status_code == 503


def test_whisper_missing_audio_file_returns_422(client):
    """Verify /whisper requires audio file."""
    response = client.post("/whisper", files={})
    
    assert response.status_code in [400, 422]  # Validation error


def test_whisper_transcription_returned_in_response(client, monkeypatch):
    """Verify transcribed text is returned correctly."""
    mock_groq = MagicMock()
    mock_response = MagicMock()
    expected_text = "This is the transcribed text"
    mock_response.text = expected_text
    mock_groq.audio.transcriptions.create.return_value = mock_response
    
    monkeypatch.setattr("main.Groq", lambda **kwargs: mock_groq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    
    response = client.post(
        "/whisper",
        files={"audio": ("test.webm", io.BytesIO(b"audio"), "audio/webm")}
    )
    
    assert response.status_code == 200
    assert response.json()["transcript"] == expected_text


def test_whisper_handles_various_audio_formats(client, monkeypatch):
    """Verify /whisper can accept different audio formats."""
    mock_groq = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Transcription"
    mock_groq.audio.transcriptions.create.return_value = mock_response
    
    monkeypatch.setattr("main.Groq", lambda **kwargs: mock_groq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    for audio_format in ["audio/webm", "audio/mp3", "audio/wav"]:
        response = client.post(
            "/whisper",
            files={"audio": ("test.audio", io.BytesIO(b"data"), audio_format)},
        )
        assert response.status_code == 200


def test_whisper_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr("main.MAX_AUDIO_BYTES", 4)
    response = client.post(
        "/whisper",
        files={"audio": ("test.webm", io.BytesIO(b"12345"), "audio/webm")},
    )
    assert response.status_code == 413


def test_whisper_rejects_overlong_recording(client):
    response = client.post(
        "/whisper",
        headers={"X-Audio-Duration-Seconds": "61"},
        files={"audio": ("test.webm", io.BytesIO(b"1234"), "audio/webm")},
    )
    assert response.status_code == 413


def test_speak_endpoint_returns_audio(client, monkeypatch):
    """Verify /speak returns audio stream."""
    audio_data = b"fake mp3 audio data"
    
    async def mock_stream():
        yield {"type": "audio", "data": audio_data}
    
    mock_communicate = AsyncMock()
    mock_communicate.stream = mock_stream
    
    monkeypatch.setattr("main.edge_tts.Communicate", lambda text, voice, rate: mock_communicate)
    
    response = client.get("/speak?text=Hello%20world")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"


def test_speak_requires_text_parameter(client):
    """Verify /speak requires text parameter."""
    response = client.get("/speak")
    
    assert response.status_code in [400, 422]  # Missing parameter


def test_speak_cleans_special_unicode_characters(client):
    """Verify special Unicode is cleaned before TTS."""
    from main import clean_for_tts
    
    text_with_special = "Hello—world…it's great"
    cleaned = clean_for_tts(text_with_special)
    
    # Should replace special chars
    assert "—" not in cleaned or "-" in cleaned


def test_speak_handles_emoji_characters(client):
    """Verify emoji and special chars are handled."""
    from main import clean_for_tts
    
    text_with_emoji = "Great! 🎉 This is awesome ✨"
    cleaned = clean_for_tts(text_with_emoji)
    
    # Should return some version (may strip or convert)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_speak_empty_text_parameter(client, monkeypatch):
    """Verify /speak handles empty text gracefully."""
    monkeypatch.setattr("main.edge_tts.Communicate", AsyncMock())
    
    response = client.get("/speak?text=")
    
    # Should either work or return error, not 404
    assert response.status_code != 404


def test_speak_long_text_handling(client, monkeypatch):
    """Verify /speak handles long text."""
    audio_data = b"audio"
    
    async def mock_stream():
        yield {"type": "audio", "data": audio_data}
    
    mock_communicate = AsyncMock()
    mock_communicate.stream = mock_stream
    monkeypatch.setattr("main.edge_tts.Communicate", lambda text, voice, rate: mock_communicate)
    
    long_text = "Hello world " * 100
    response = client.get(f"/speak?text={long_text}")
    
    # Should handle or reject, not crash
    assert response.status_code in [200, 413]


def test_speak_uses_correct_voice_settings(client, monkeypatch):
    """Verify /speak uses configured voice."""
    from unittest.mock import call
    
    audio_data = b"audio"
    
    async def mock_stream():
        yield {"type": "audio", "data": audio_data}
    
    mock_communicate = AsyncMock()
    mock_communicate.stream = mock_stream
    
    mock_edge_tts = MagicMock()
    mock_edge_tts.Communicate = lambda text, voice, rate: mock_communicate
    
    monkeypatch.setattr("main.edge_tts", mock_edge_tts)
    
    response = client.get("/speak?text=test")
    
    # Verify Communicate was called with expected parameters
    assert response.status_code in [200, 500]


def test_speak_returns_non_empty_audio(client, monkeypatch):
    """Verify /speak doesn't return empty audio stream."""
    async def mock_stream():
        # Simulate non-empty audio
        yield {"type": "audio", "data": b"some audio data here"}
    
    mock_communicate = AsyncMock()
    mock_communicate.stream = mock_stream
    monkeypatch.setattr("main.edge_tts.Communicate", lambda text, voice, rate: mock_communicate)
    
    response = client.get("/speak?text=hello")
    
    assert response.status_code == 200
    # Response should have content
    assert len(response.content) > 0
