import base64

from connectors import portfolio


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_only_indexes_portfolio_content_files(monkeypatch):
    tree = {
        "tree": [
            {
                "path": "src/content/projects.jsx",
                "type": "blob",
                "size": 100,
                "sha": "abc",
                "url": "https://api.github.test/blob/projects",
            },
            {
                "path": "src/components/Secret.jsx",
                "type": "blob",
                "size": 100,
                "sha": "private",
                "url": "https://api.github.test/blob/secret",
            },
        ]
    }
    source = "export const projects = [{ title: 'Tab Story', description: 'Local AI tab manager' }]"
    blob = {"content": base64.b64encode(source.encode()).decode()}

    def fake_get(url, **kwargs):
        return FakeResponse(blob if "blob/projects" in url else tree)

    monkeypatch.setattr(portfolio.requests, "get", fake_get)
    monkeypatch.setattr(portfolio, "_headers", lambda: {})

    chunks = portfolio.get_portfolio_chunks()

    assert chunks
    assert {chunk["source"] for chunk in chunks} == {"src/content/projects.jsx"}
    assert "Tab Story" in chunks[0]["text"]
    assert all(chunk["visibility"] == "public" for chunk in chunks)


def test_large_content_is_split_into_bounded_chunks():
    chunks = portfolio._split_text("A" * 4100, max_chars=1000)
    assert len(chunks) == 5
    assert max(map(len, chunks)) <= 1000
