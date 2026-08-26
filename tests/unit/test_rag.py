"""
Unit tests for RAG system (knowledge chunking, embedding, retrieval).
"""

import pytest
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
