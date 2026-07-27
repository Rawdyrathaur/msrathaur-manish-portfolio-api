import json
import logging
import os
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
KNOWLEDGE_DIR   = BASE_DIR / "knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "manish_portfolio"
CACHE_DIR       = Path(os.getenv("RAG_CACHE_DIR", "backend/.cache/rag"))
PERSIST_DIR     = CACHE_DIR / "chroma"
MANIFEST_PATH   = CACHE_DIR / "manifest.json"

# TOP_K is adaptive — simple questions get 3, broad questions get 7
TOP_K_DEFAULT = 5
TOP_K_MAX     = 8

# ── Load model once at startup ────────────────────────────
logger.info("Loading embedding model...")
_embedder_start = time.perf_counter()
_embedder = SentenceTransformer(EMBED_MODEL)
logger.info("Embedding model loaded in %.3fs.", time.perf_counter() - _embedder_start)

# ── ChromaDB persistent client ────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_client     = chromadb.PersistentClient(path=str(PERSIST_DIR))
_collection = _client.get_or_create_collection(COLLECTION_NAME)


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


# ══════════════════════════════════════════════════════════
#  LOAD & EMBED
# ══════════════════════════════════════════════════════════

def load_knowledge() -> int:
    global _collection
    load_start = time.perf_counter()
    manifest       = _current_manifest()
    saved_manifest = _load_saved_manifest()

    if saved_manifest == manifest:
        existing_count = _collection.count()
        if existing_count > 0:
            logger.info("[RAG] Cache HIT - using existing Chroma collection")
            return existing_count

    md_files = sorted(str(path) for path in KNOWLEDGE_DIR.glob("*.md"))

    all_chunks = []
    
    # 1. Local Markdown Files
    for filepath in md_files:
        source = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_markdown(text, source)
        all_chunks.extend(chunks)

    # 2. External Connectors
    try:
        from connectors.github import get_github_chunks
        from connectors.linkedin import get_linkedin_chunks
        
        logger.info("[RAG] Fetching GitHub chunks...")
        all_chunks.extend(get_github_chunks())
        
        logger.info("[RAG] Fetching LinkedIn chunks...")
        all_chunks.extend(get_linkedin_chunks(KNOWLEDGE_DIR))
    except Exception as e:
        logger.warning(f"Error loading connectors: {e}")

    if not all_chunks:
        return 0

    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(COLLECTION_NAME)

    texts     = [c["text"]    for c in all_chunks]
    ids       = [c["id"]      for c in all_chunks]
    # Add our new rich metadata
    metadatas = [{
        "source": c.get("source", "unknown"), 
        "heading": c.get("heading", ""),
        "title": c.get("title", "Unknown"),
        "type": c.get("type", "unknown"),
        "url": c.get("url", "/"),
        "source_type": c.get("source_type", "portfolio"),
        "content_type": c.get("content_type", "unknown"),
        "visibility": c.get("visibility", "public"),
        "trust_level": c.get("trust_level", "verified"),
        "timestamp": c.get("timestamp", ""),
        "last_updated": c.get("last_updated", "")
    } for c in all_chunks]

    logger.info("[RAG] Cache MISS - rebuilding embeddings")
    embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()

    _collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas,
    )

    _save_manifest(manifest)
    return len(all_chunks)


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
    if _collection.count() == 0:
        return "", [], 999.0

    k = top_k if top_k is not None else _resolve_top_k(query)
    k = min(k, _collection.count())

    query_embedding = _embedder.encode([query]).tolist()

    results = _collection.query(
        query_embeddings = query_embedding,
        n_results        = k,
        include          = ["documents", "metadatas", "distances"],
    )

    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "", [], 999.0

    labeled_chunks = []
    sources_map = {}
    
    best_distance = distances[0] if distances else 999.0

    for doc, meta, dist in zip(docs, metadatas, distances):
        # Hard cutoff for complete garbage
        if dist > 1.6:
            continue
            
        source  = meta.get("source", "unknown")
        heading = meta.get("heading", "")
        title   = meta.get("title", source)
        m_type  = meta.get("type", "unknown")
        url     = meta.get("url", "/")
        source_type = meta.get("source_type", "portfolio")
        content_type = meta.get("content_type", m_type)
        visibility = meta.get("visibility", "public")
        trust_level = meta.get("trust_level", "verified")
        
        # Hard grounding constraint: Only allow verified or public sources
        if trust_level not in ("verified", "public"):
            continue
            
        timestamp = meta.get("timestamp", "")
        last_updated = meta.get("last_updated", "")
        
        label   = f"[Source: {title} ({source_type}) — {heading}]"
        labeled_chunks.append(f"{label}\n{doc}")
        
        if title not in sources_map:
            sources_map[title] = {
                "title": title,
                "type": m_type,
                "url": url,
                "source_type": source_type,
                "content_type": content_type,
                "visibility": visibility,
                "trust_level": trust_level,
                "timestamp": timestamp,
                "last_updated": last_updated
            }

    context = "\n\n---\n\n".join(labeled_chunks)
    sources = list(sources_map.values())
    
    return context, sources, best_distance


# ══════════════════════════════════════════════════════════
#  DYNAMIC UPSERT (WEBHOOKS)
# ══════════════════════════════════════════════════════════

def upsert_github_repo(repo_data: dict) -> bool:
    """Upserts a repository's deep chunks into ChromaDB."""
    try:
        from connectors.github import format_repo_chunks, get_repo_details, GITHUB_USERNAME
        
        repo_name = repo_data.get("name")
        owner = repo_data.get("owner", {}).get("login", GITHUB_USERNAME)
        
        # Need to fetch details since we're doing a deep sync
        # Note: In a real webhook, we might want to pass the token in headers, but for now we'll fetch basic if public.
        # To keep it simple, we just pass empty headers (works for public repos).
        details = get_repo_details(repo_name, owner, headers={})
        
        chunks = format_repo_chunks(repo_data, details)
        
        # Delete existing chunks for this repo to avoid duplicates
        delete_github_repo(repo_name)
        
        for chunk in chunks:
            text = chunk["text"]
            chunk_id = chunk["id"]
            meta = {
                "source": chunk.get("source", "unknown"),
                "heading": chunk.get("heading", ""),
                "title": chunk.get("title", "Unknown"),
                "type": chunk.get("type", "unknown"),
                "url": chunk.get("url", "/"),
                "source_type": chunk.get("source_type", "portfolio"),
                "content_type": chunk.get("content_type", "unknown"),
                "visibility": chunk.get("visibility", "public"),
                "trust_level": chunk.get("trust_level", "verified"),
                "timestamp": chunk.get("timestamp", ""),
                "last_updated": chunk.get("last_updated", "")
            }
            
            embedding = _embedder.encode([text], show_progress_bar=False).tolist()[0]
                
            _collection.add(
                ids=[chunk_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[meta]
            )
            
        logger.info(f"[RAG] Upserted deep GitHub repo: {repo_name}")
        return True
    except Exception as e:
        logger.error(f"[RAG] Failed to upsert GitHub repo: {e}")
        return False

def delete_github_repo(repo_name: str) -> bool:
    """Deletes all chunks for a repository from ChromaDB."""
    try:
        # Delete by iterating over potential chunk IDs
        ids_to_delete = [
            f"github_repo::{repo_name}::overview",
            f"github_repo::{repo_name}::readme",
            f"github_repo::{repo_name}::tree"
        ]
        
        for chunk_id in ids_to_delete:
            try:
                _collection.delete(ids=[chunk_id])
            except Exception:
                pass
                
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