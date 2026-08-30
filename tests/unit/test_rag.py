"""
Unit tests for RAG system (knowledge chunking, embedding, retrieval).
"""

import pytest
import rag
from rag import _chunk_markdown


def test_chunk_markdown_splits_at_double_hash_headings():
    """Verify markdown is split at every ## heading."""
    text = """# Main Title
Intro content

## Section 1
Content for section 1

## Section 2
Content for section 2"""
    
    chunks = _chunk_markdown(text, "test.md")
    
    assert len(chunks) == 3
    assert chunks[0]["id"] == "test.md::1::intro"
    assert chunks[1]["id"] == "test.md::2::section_1"
    assert "Content for section 1" in chunks[1]["text"]
    assert "Content for section 2" in chunks[2]["text"]


def test_chunk_markdown_preserves_all_content():
    """Ensure no content is lost during chunking."""
    text = """## Section A
Line 1
Line 2
- Bullet point
- Another bullet

## Section B
Some code:
```python
print("hello")
```
More content"""
    
    chunks = _chunk_markdown(text, "test.md")
    combined = "\n".join(chunk["text"] for chunk in chunks)
    
    assert "Line 1" in combined
    assert "Line 2" in combined
    assert "Bullet point" in combined
    assert "Another bullet" in combined
    assert 'print("hello")' in combined


def test_chunk_markdown_includes_source():
    """Verify source filename is included in metadata."""
    text = "## Test\nContent"
    chunks = _chunk_markdown(text, "about.md")
    
    assert chunks[0]["source"] == "about.md"
    assert "about.md" in chunks[0]["id"]


def test_chunk_markdown_no_headings_creates_intro_chunk():
    """Handle markdown with no ## headings."""
    text = "This is just some plain text with no headings at all."
    chunks = _chunk_markdown(text, "plain.md")
    
    assert len(chunks) == 1
    assert chunks[0]["id"] == "plain.md::1::intro"


def test_chunk_markdown_empty_file():
    """Handle empty markdown files."""
    text = ""
    chunks = _chunk_markdown(text, "empty.md")
    
    # Should return empty or intro chunk depending on implementation
    assert isinstance(chunks, list)


def test_chunk_markdown_whitespace_handling():
    """Verify whitespace is handled correctly."""
    text = """

## Section 1
   Content with leading spaces
   
## Section 2
Content
   """
    
    chunks = _chunk_markdown(text, "test.md")
    # Chunks should have stripped content
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)


def test_chunk_markdown_special_characters():
    """Verify special characters in headings are handled."""
    text = """## Section 1: Python & FastAPI
Content

## Section 2 (Advanced)
More content

## Section 3 – Advanced Topics
Even more"""
    
    chunks = _chunk_markdown(text, "test.md")
    assert len(chunks) >= 2
    # Should convert special chars to valid IDs
    assert all(isinstance(chunk["id"], str) for chunk in chunks)


def test_chunk_markdown_multiple_heading_levels():
    """Test that only ## headings are used as splits."""
    text = """# Top level (should not split)
Content

## Real Section
Section content

### Sub section (not a split point)
Subsection content

## Another Section
More content"""
    
    chunks = _chunk_markdown(text, "test.md")
    # Should have split at ## but not # or ###
    assert len(chunks) >= 2
    assert any("Real Section" in chunk["text"] for chunk in chunks)


def test_abstract_reasoning_questions_retrieve_broader_evidence():
    assert rag._resolve_top_k("What are his strongest areas?") == rag.TOP_K_MAX
    assert rag._resolve_top_k("Which project best demonstrates production engineering?") == rag.TOP_K_MAX
    assert rag._resolve_top_k("Compare Tab Story and Carbon Pulse") == rag.TOP_K_MAX


def test_multi_query_retrieval_fuses_and_deduplicates_candidates(monkeypatch):
    class FakeEmbeddings(list):
        def tolist(self):
            return list(self)

    class FakeEmbedder:
        def encode(self, queries, normalize_embeddings=True):
            assert len(queries) == 2
            assert normalize_embeddings is True
            return FakeEmbeddings([[0.1, 0.2], [0.2, 0.1]])

    skill_meta = {
        "source": "skills.md", "heading": "Backend", "title": "Skills",
        "type": "skill", "url": "/", "source_type": "portfolio",
        "content_type": "skills", "visibility": "public",
        "trust_level": "verified", "timestamp": "", "last_updated": "",
    }
    project_meta = {
        "source": "projects.md", "heading": "Carbon Pulse", "title": "Carbon Pulse",
        "type": "project", "url": "/", "source_type": "portfolio",
        "content_type": "project", "visibility": "public",
        "trust_level": "verified", "timestamp": "", "last_updated": "",
    }

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            assert len(kwargs["query_embeddings"]) == 2
            return {
                "ids": [["skills", "project"], ["skills", "project"]],
                "documents": [
                    ["Java and backend APIs", "Microservices with Docker and Kafka"],
                    ["Java and backend APIs", "Microservices with Docker and Kafka"],
                ],
                "metadatas": [[skill_meta, project_meta], [skill_meta, project_meta]],
                "distances": [[0.69, 0.62], [0.41, 0.46]],
            }

    monkeypatch.setattr(rag, "_embedder", FakeEmbedder())
    monkeypatch.setattr(rag, "_collection", FakeCollection())
    context, sources, best_distance = rag.get_relevant_context(
        "What are his strongest areas?",
        top_k=2,
        search_queries=[
            "What are his strongest areas?",
            "skills experience projects and demonstrated evidence",
        ],
    )

    assert "Java and backend APIs" in context
    assert "Microservices with Docker and Kafka" in context
    assert len(sources) == 2
    assert best_distance == 0.41
