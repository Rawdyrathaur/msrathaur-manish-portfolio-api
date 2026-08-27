"""Fetch the public portfolio content directly from GitHub for RAG indexing."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import base64
from pathlib import PurePosixPath

import requests

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
PORTFOLIO_REPOSITORY = os.getenv(
    "PORTFOLIO_REPOSITORY", "Rawdyrathaur/portfolio"
)
PORTFOLIO_BRANCH = os.getenv("PORTFOLIO_BRANCH", "main")
REQUEST_TIMEOUT = float(os.getenv("SOURCE_REQUEST_TIMEOUT", "12"))
MAX_SOURCE_BYTES = int(os.getenv("MAX_SOURCE_BYTES", "250000"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1800"))

_ALLOWED_PREFIXES = ("src/content/",)
_ALLOWED_SUFFIXES = {".js", ".jsx", ".json", ".md", ".mdx", ".txt"}


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "manish-portfolio-rag/2.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        try:
            from connectors.github import get_installation_token
            token = get_installation_token()
        except Exception as exc:
            logger.warning("Portfolio GitHub App authentication failed: %s", exc)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _is_content_file(path: str) -> bool:
    pure_path = PurePosixPath(path)
    return path.startswith(_ALLOWED_PREFIXES) and pure_path.suffix.lower() in _ALLOWED_SUFFIXES


def _clean_source(text: str, suffix: str) -> str:
    """Keep human-authored content while removing common JSX/module noise."""
    if suffix in {".md", ".mdx", ".txt"}:
        return text.strip()

    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"^\s*//.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*import\s+.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\b(?:export\s+default|export\s+const|const)\b", " ", text)
    text = re.sub(r"[{}\[\]();]", " ", text)
    text = re.sub(r"\s*,\s*", "\n", text)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    blocks = re.split(r"\n\s*\n|(?=^##?\s)", text, flags=re.MULTILINE)
    chunks: list[str] = []
    current = ""
    for block in (block.strip() for block in blocks if block.strip()):
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                block[index:index + max_chars]
                for index in range(0, len(block), max_chars)
            )
        elif not current:
            current = block
        elif len(current) + len(block) + 2 <= max_chars:
            current = f"{current}\n\n{block}"
        else:
            chunks.append(current)
            current = block
    if current:
        chunks.append(current)
    return chunks


def get_portfolio_chunks() -> list[dict]:
    """Return fresh, public portfolio content from the configured GitHub repository."""
    headers = _headers()
    tree_url = (
        f"{GITHUB_API_URL}/repos/{PORTFOLIO_REPOSITORY}/git/trees/"
        f"{PORTFOLIO_BRANCH}?recursive=1"
    )
    try:
        response = requests.get(tree_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        tree = response.json().get("tree", [])
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Could not fetch portfolio file tree: %s", exc)
        return []

    chunks: list[dict] = []
    for item in tree:
        path = item.get("path", "")
        if item.get("type") != "blob" or not _is_content_file(path):
            continue
        if int(item.get("size") or 0) > MAX_SOURCE_BYTES:
            logger.info("Skipping oversized portfolio source: %s", path)
            continue

        try:
            source_response = requests.get(
                item["url"], headers=headers, timeout=REQUEST_TIMEOUT
            )
            source_response.raise_for_status()
            payload = source_response.json()
            source_text = base64.b64decode(payload.get("content", "")).decode(
                "utf-8", errors="replace"
            )
            cleaned = _clean_source(
                source_text, PurePosixPath(path).suffix.lower()
            )
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Could not fetch portfolio source %s: %s", path, exc)
            continue

        title = PurePosixPath(path).stem.replace("-", " ").replace("_", " ").title()
        source_url = (
            f"https://github.com/{PORTFOLIO_REPOSITORY}/blob/"
            f"{PORTFOLIO_BRANCH}/{path}"
        )
        source_hash = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
        for index, content in enumerate(_split_text(cleaned), start=1):
            chunks.append({
                "id": f"portfolio::{source_hash}::{index}",
                "text": content,
                "source": path,
                "heading": title,
                "title": title,
                "type": "portfolio_content",
                "url": source_url,
                "source_type": "portfolio_live",
                "content_type": "portfolio_content",
                "visibility": "public",
                "trust_level": "verified",
                "timestamp": item.get("sha", ""),
                "last_updated": item.get("sha", ""),
            })

    logger.info("Fetched %d live portfolio chunks", len(chunks))
    return chunks
