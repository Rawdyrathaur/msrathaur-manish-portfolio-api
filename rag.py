import json
import hashlib
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
KNOWLEDGE_DIR   = BASE_DIR / "knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "manish_portfolio_v2"
CACHE_DIR       = Path(os.getenv("RAG_CACHE_DIR", "/tmp/manish-portfolio-rag"))
PERSIST_DIR     = CACHE_DIR / "chroma"
MANIFEST_PATH   = CACHE_DIR / "manifest.json"

# TOP_K is adaptive — simple questions get 3, broad questions get 7
TOP_K_DEFAULT = 5
TOP_K_MAX     = 12
RETRIEVAL_CANDIDATES = 50
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

# ── Load model once at startup ────────────────────────────
logger.info("Loading embedding model...")
_embedder_start = time.perf_counter()
_embedder = SentenceTransformer(EMBED_MODEL)
logger.info("Embedding model loaded in %.3fs.", time.perf_counter() - _embedder_start)

# ── ChromaDB persistent client ────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_client     = chromadb.PersistentClient(path=str(PERSIST_DIR))
_collection = _client.get_or_create_collection(
    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)
_index_lock = threading.RLock()
_index_state = {
    "status": "initializing",
    "chunks": 0,
    "live_portfolio_chunks": 0,
    "github_chunks": 0,
    "fallback_chunks": 0,
    "last_sync": None,
    "last_error": None,
}


def get_index_status() -> dict:
    return dict(_index_state)


def _current_manifest() -> dict:
    files = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        stat = path.stat()
        files.append({
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return {
        "schema_version": 5,  # Bumping to 5 to force cache invalidation for the new chunk ids
        "collection_name": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "files": files,
    }


def _load_saved_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read knowledge manifest: %s", exc)
        return None


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# ══════════════════════════════════════════════════════════
#  CHUNKING — parses YAML frontmatter and splits
# ══════════════════════════════════════════════════════════

def _chunk_markdown(text: str, source: str) -> list[dict]:
    lines = text.splitlines()
    title = "Unknown"
    m_type = "unknown"
    url = "/"
    timestamp = ""
    last_updated = ""
    
    body_lines = []
    in_frontmatter = False
    
    # Parse frontmatter
    if lines and lines[0].strip() == "---":
        in_frontmatter = True
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                in_frontmatter = False
                body_lines = lines[i+1:]
                break
            if ":" in lines[i]:
                key, val = lines[i].split(":", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key == "title": title = val
                elif key == "type": m_type = val
                elif key == "url": url = val
                elif key == "timestamp": timestamp = val
                elif key == "last_updated": last_updated = val
    else:
        body_lines = lines

    chunks = []
    current = []
    heading = "Intro"
    chunk_counter = 0
    
    def add_chunk(text_block, current_heading):
        nonlocal chunk_counter
        if text_block:
            chunk_counter += 1
            chunks.append({
                "id": f"{source}::{chunk_counter}::{current_heading.lower().replace(' ', '_')}",
                "text": text_block.strip(),
                "source": source,
                "heading": current_heading,
                "title": title,
                "type": m_type,
                "url": url,
                "source_type": "portfolio",
                "content_type": m_type,
                "visibility": "public",
                "trust_level": "verified",
                "timestamp": timestamp,
                "last_updated": last_updated
            })
    
    for line in body_lines:
        if line.startswith("## ") or line.startswith("### "):
            add_chunk("\n".join(current), heading)
            heading = line.lstrip("#").strip()
            current = [line]
        else:
            current.append(line)
            
    add_chunk("\n".join(current), heading)
        
    return [c for c in chunks if c["text"].strip()]


def _split_tokens(text: str) -> list[str]:
    """Use one model-aware chunking policy for every connector."""
    tokenizer = getattr(_embedder, "tokenizer", None)
    configured = int(os.getenv("RAG_CHUNK_TOKENS", "480"))
    model_limit = int(getattr(_embedder, "max_seq_length", configured) or configured)
    chunk_tokens = max(120, min(configured, model_limit - 2))
    overlap = max(1, int(chunk_tokens * float(os.getenv("RAG_CHUNK_OVERLAP", "0.12"))))

    if tokenizer and callable(getattr(tokenizer, "encode", None)):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= chunk_tokens:
            return [text.strip()] if text.strip() else []
        chunks = []
        step = chunk_tokens - overlap
        for start in range(0, len(token_ids), step):
            window = token_ids[start:start + chunk_tokens]
            if not window:
                break
            chunks.append(tokenizer.decode(window, skip_special_tokens=True).strip())
            if start + chunk_tokens >= len(token_ids):
                break
        return [chunk for chunk in chunks if chunk]

    # Test/development fallback when a tokenizer is not available.
    words = text.split()
    if len(words) <= chunk_tokens:
        return [text.strip()] if text.strip() else []
    step = chunk_tokens - overlap
    return [" ".join(words[start:start + chunk_tokens]) for start in range(0, len(words), step)]


def _standardize_chunks(chunks: list[dict]) -> list[dict]:
    standardized = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        parts = _split_tokens(text)
        for index, part in enumerate(parts, start=1):
            normalized = dict(chunk)
            if len(parts) > 1:
                normalized["id"] = f"{chunk['id']}::part{index}"
            normalized["text"] = part
            normalized["source_url"] = chunk.get("source_url", chunk.get("url", "/"))
            normalized["content_hash"] = hashlib.sha256(part.encode("utf-8")).hexdigest()
            standardized.append(normalized)
    return standardized


def _metadata_for(chunk: dict) -> dict:
    return {
        "source": chunk.get("source", "unknown"),
        "heading": chunk.get("heading", ""),
        "title": chunk.get("title", "Unknown"),
        "type": chunk.get("type", "unknown"),
        "url": chunk.get("url", "/"),
        "source_url": chunk.get("source_url", chunk.get("url", "/")),
        "source_type": chunk.get("source_type", "portfolio"),
        "content_type": chunk.get("content_type", "unknown"),
        "visibility": chunk.get("visibility", "public"),
        "trust_level": chunk.get("trust_level", "verified"),
        "timestamp": chunk.get("timestamp", ""),
        "last_updated": chunk.get("last_updated", ""),
        "content_hash": chunk.get("content_hash", ""),
    }


# ══════════════════════════════════════════════════════════
#  LOAD & EMBED
# ══════════════════════════════════════════════════════════

def load_knowledge() -> int:
    global _collection
    load_start = time.perf_counter()
    _index_state.update({"status": "syncing", "last_error": None})
    shadow_name = None
    backup_name = None

    try:
        from connectors.github import get_github_chunks
        from connectors.linkedin import get_linkedin_chunks
        from connectors.portfolio import get_portfolio_chunks

        portfolio_chunks = get_portfolio_chunks()
        github_chunks = get_github_chunks()
        linkedin_chunks = get_linkedin_chunks(KNOWLEDGE_DIR)

        fallback_chunks = []
        if not portfolio_chunks:
            logger.warning("Live portfolio fetch failed; using bundled fallback content")
            for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
                fallback_chunks.extend(
                    _chunk_markdown(path.read_text(encoding="utf-8"), path.name)
                )

        normalized_chunks = _standardize_chunks(
            portfolio_chunks + github_chunks + linkedin_chunks + fallback_chunks
        )
        chunk_map = {
            chunk["id"]: chunk
            for chunk in normalized_chunks
            if chunk.get("id") and chunk.get("text", "").strip()
        }
        all_chunks = list(chunk_map.values())
        if not all_chunks:
            raise RuntimeError("No public portfolio sources could be indexed")

        texts = [chunk["text"].strip() for chunk in all_chunks]
        metadatas = [_metadata_for(chunk) for chunk in all_chunks]
        embeddings = _embedder.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        # Readers continue using the live collection while sources and embeddings
        # are prepared. The short mutation window below is serialized.
        with _index_lock:
            suffix = uuid.uuid4().hex[:10]
            shadow_name = f"{COLLECTION_NAME}_shadow_{suffix}"
            backup_name = f"{COLLECTION_NAME}_backup_{suffix}"
            shadow = _client.create_collection(
                shadow_name, metadata={"hnsw:space": "cosine"}
            )
            shadow.add(
                ids=list(chunk_map),
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            if shadow.count() != len(all_chunks):
                raise RuntimeError("Shadow index validation failed")

            previous = _collection
            previous.modify(name=backup_name)
            try:
                shadow.modify(name=COLLECTION_NAME)
            except Exception:
                previous.modify(name=COLLECTION_NAME)
                raise
            _collection = shadow
            try:
                _client.delete_collection(backup_name)
            except Exception as exc:
                logger.warning("Could not remove previous RAG collection: %s", exc)

        _index_state.update({
            "status": "ready",
            "chunks": len(all_chunks),
            "live_portfolio_chunks": len(portfolio_chunks),
            "github_chunks": len(github_chunks),
            "fallback_chunks": len(fallback_chunks),
            "last_sync": int(time.time()),
        })
        logger.info(
            "[RAG] Atomically indexed %d chunks in %.2fs",
            len(all_chunks), time.perf_counter() - load_start,
        )
        return len(all_chunks)
    except Exception as exc:
        if shadow_name:
            try:
                _client.delete_collection(shadow_name)
            except Exception:
                pass
        _index_state.update({"status": "error", "last_error": str(exc)[:300]})
        logger.exception("[RAG] Index refresh failed")
        try:
            with _index_lock:
                return _collection.count()
        except Exception:
            return 0


# ══════════════════════════════════════════════════════════
#  ADAPTIVE TOP_K
# ══════════════════════════════════════════════════════════

_BROAD_KEYWORDS = {
    "everything", "all", "tell me about", "overview", "summary",
    "who is", "what does", "background", "skills", "experience",
    "projects", "work", "journey", "full", "complete",
    "github", "repos", "repositories", "list",
}

def _resolve_top_k(query: str) -> int:
    q_lower = query.lower()
    if any(kw in q_lower for kw in _BROAD_KEYWORDS):
        return TOP_K_MAX
    return TOP_K_DEFAULT


# ══════════════════════════════════════════════════════════
#  RETRIEVE
# ══════════════════════════════════════════════════════════

def get_relevant_context(query: str, top_k: int | None = None) -> tuple[str, list[dict], float]:
    """
    Returns:
      1. Formatted context string for LLM
      2. List of source dictionaries for UI
      3. Best (lowest) L2 distance score
    """
    query_embedding = _embedder.encode(
        [query], normalize_embeddings=True
    ).tolist()
    with _index_lock:
        collection_count = _collection.count()
        if collection_count == 0:
            return "", [], 999.0
        k = min(top_k or _resolve_top_k(query), collection_count)
        candidate_count = min(max(k * 3, RETRIEVAL_CANDIDATES), collection_count)
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "", [], 999.0

    query_terms = {
        token for token in re.findall(r"[a-z0-9+#.-]{2,}", query.lower())
        if token not in {"what", "which", "about", "tell", "manish", "does", "have", "with"}
    }
    ranked = []
    category_sources = {
        "project": "projects",
        "skill": "skills",
        "experience": "experience",
        "blog": "blog",
        "article": "blog",
        "contact": "contact",
    }
    for doc, meta, distance in zip(docs, metadatas, distances):
        haystack = f"{meta.get('title', '')} {meta.get('heading', '')} {doc}".lower()
        overlap = sum(1 for term in query_terms if term in haystack)
        score = float(distance) - (0.08 * overlap)
        source = meta.get("source", "").lower()
        if meta.get("source_type") == "portfolio_live":
            score -= 0.10
        for term, source_name in category_sources.items():
            if term in query.lower() and source_name in source:
                score -= 0.40
        if "project" in query.lower() and meta.get("content_type") == "repo_overview":
            score -= 0.08
        ranked.append((max(score, 0.0), float(distance), doc, meta))
    ranked.sort(key=lambda item: item[0])

    labeled_chunks = []
    selected_sources = []
    selected_distances = []
    max_distance = float(os.getenv("RAG_MAX_DISTANCE", "0.68"))

    context_chars = 0
    chunks_per_title: dict[str, int] = {}
    for rerank_score, dist, doc, meta in ranked:
        if len(labeled_chunks) >= k:
            break
        if dist > max_distance:
            continue
            
        source  = meta.get("source", "unknown")
        heading = meta.get("heading", "")
        title   = meta.get("title", source)
        if chunks_per_title.get(title, 0) >= 2:
            continue
        m_type  = meta.get("type", "unknown")
        url     = meta.get("url", "/")
        source_type = meta.get("source_type", "portfolio")
        content_type = meta.get("content_type", m_type)
        visibility = meta.get("visibility", "public")
        trust_level = meta.get("trust_level", "verified")
        
        # Hard grounding constraint: Only allow verified or public sources
        if visibility != "public" or trust_level not in (
            "verified", "public", "user_provided", "untrusted_external"
        ):
            continue
            
        timestamp = meta.get("timestamp", "")
        last_updated = meta.get("last_updated", "")
        
        label   = f"[Source: {title} ({source_type}) — {heading}]"
        labeled = f"{label}\n{doc}"
        if context_chars + len(labeled) > MAX_CONTEXT_CHARS:
            continue
        labeled_chunks.append(labeled)
        selected_distances.append(dist)
        context_chars += len(labeled)
        chunks_per_title[title] = chunks_per_title.get(title, 0) + 1
        selected_sources.append({
            "title": title,
            "type": m_type,
            "url": url,
            "source_type": source_type,
            "content_type": content_type,
            "visibility": visibility,
            "trust_level": trust_level,
            "timestamp": timestamp,
            "last_updated": last_updated,
            "distance": dist,
            "rerank_score": rerank_score,
        })

    context = "\n\n---\n\n".join(labeled_chunks)
    best_distance = selected_distances[0] if selected_distances else 999.0
    return context, selected_sources, best_distance


# ══════════════════════════════════════════════════════════
#  DYNAMIC UPSERT (WEBHOOKS)
# ══════════════════════════════════════════════════════════

def upsert_github_repo(repo_data: dict) -> bool:
    """Upserts a repository's deep chunks into ChromaDB."""
    try:
        if repo_data.get("private"):
            logger.info("[RAG] Ignoring private GitHub repository webhook")
            return False
        from connectors.github import (
            GITHUB_USERNAME,
            format_repo_chunks,
            get_installation_token,
            get_repo_details,
        )
        
        repo_name = repo_data.get("name")
        if not repo_name:
            return False
        owner = repo_data.get("owner", {}).get("login", GITHUB_USERNAME)
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = get_installation_token()
        if token:
            headers["Authorization"] = f"token {token}"
        details = get_repo_details(repo_name, owner, headers=headers)
        chunks = _standardize_chunks(format_repo_chunks(repo_data, details))
        if not chunks:
            return False

        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [_metadata_for(chunk) for chunk in chunks]
        embeddings = _embedder.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        with _index_lock:
            existing = _collection.get(where={"title": repo_name}, include=[])
            existing_ids = set(existing.get("ids", []))
            _collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            stale_ids = list(existing_ids - set(ids))
            if stale_ids:
                _collection.delete(ids=stale_ids)
            _index_state.update({
                "chunks": _collection.count(),
                "last_sync": int(time.time()),
                "status": "ready",
            })
            
        logger.info(f"[RAG] Upserted deep GitHub repo: {repo_name}")
        return True
    except Exception as e:
        logger.error(f"[RAG] Failed to upsert GitHub repo: {e}")
        return False

def delete_github_repo(repo_name: str) -> bool:
    """Deletes all chunks for a repository from ChromaDB."""
    try:
        with _index_lock:
            _collection.delete(where={"$and": [
                {"title": repo_name},
                {"source_type": "github"},
            ]})
            _index_state.update({
                "chunks": _collection.count(),
                "last_sync": int(time.time()),
            })
                
        logger.info(f"[RAG] Deleted GitHub repo chunks: {repo_name}")
        return True
    except Exception as e:
        logger.error(f"[RAG] Failed to delete GitHub repo {repo_name}: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    total = load_knowledge()
    print(f"\n✅ Loaded {total} chunks\n")
    
    queries = [
        "what projects has he built",
        "who is his gf",
        "what is 5+5",
        "how did he contribute to kubestellar"
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        ctx, srcs, dist = get_relevant_context(q)
        print(f"Distance: {dist:.3f}")
        for s in srcs:
            print(f"Source: {s['title']} ({s['type']})")
