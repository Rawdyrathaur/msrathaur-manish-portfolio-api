import glob
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
KNOWLEDGE_DIR   = BASE_DIR / "knowledge"  # folder with all .md files
EMBED_MODEL     = "all-MiniLM-L6-v2"      # small, fast, local — no API needed
TOP_K           = 3                       # how many chunks to retrieve per query
COLLECTION_NAME = "manish_portfolio"
CACHE_DIR       = Path(os.getenv("RAG_CACHE_DIR", "backend/.cache/rag"))
PERSIST_DIR     = CACHE_DIR / "chroma"
MANIFEST_PATH   = CACHE_DIR / "manifest.json"

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
        "schema_version": 1,
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
#  CHUNKING  — splits .md file by ## headings
# ══════════════════════════════════════════════════════════

def _chunk_markdown(text: str, source: str) -> list[dict]:
    """
    Splits markdown content into chunks at every ## heading.
    Each chunk = { id, text, source }
    """
    chunks  = []
    current = []
    heading = "intro"

    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                chunks.append({
                    "id":     f"{source}::{heading}",
                    "text":   "\n".join(current).strip(),
                    "source": source,
                })
            heading = line[3:].strip().lower().replace(" ", "_")
            current = [line]
        else:
            current.append(line)

    # last chunk
    if current:
        chunks.append({
            "id":     f"{source}::{heading}",
            "text":   "\n".join(current).strip(),
            "source": source,
        })

    return [c for c in chunks if c["text"]]


# ══════════════════════════════════════════════════════════
#  LOAD & EMBED — reads all .md files and stores in Chroma
# ══════════════════════════════════════════════════════════

def load_knowledge() -> int:
    """
    Reads every .md file in knowledge/ folder,
    chunks and embeds them, stores in ChromaDB.
    Returns total number of chunks loaded.
    """
    global _collection
    load_start = time.perf_counter()
    manifest = _current_manifest()
    saved_manifest = _load_saved_manifest()

    if saved_manifest == manifest:
        existing_count = _collection.count()
        if existing_count > 0:
            logger.info("[RAG] Cache HIT - using existing Chroma collection")
            logger.info("load_knowledge() finished in %.3fs with %d chunks.", time.perf_counter() - load_start, existing_count)
            return existing_count

    md_files = sorted(str(path) for path in KNOWLEDGE_DIR.glob("*.md"))

    if not md_files:
        logger.warning(f"No .md files found in '{KNOWLEDGE_DIR}/' folder.")
        logger.info("load_knowledge() finished in %.3fs with 0 chunks.", time.perf_counter() - load_start)
        return 0

    all_chunks = []
    for filepath in md_files:
        source = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_markdown(text, source)
        all_chunks.extend(chunks)
        logger.info(f"Loaded {len(chunks)} chunks from {source}")

    if not all_chunks:
        logger.warning("No chunks found after parsing.")
        logger.info("load_knowledge() finished in %.3fs with 0 chunks.", time.perf_counter() - load_start)
        return 0

    # Clear old data before reloading
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(COLLECTION_NAME)

    # Embed all chunks
    texts     = [c["text"] for c in all_chunks]
    ids       = [c["id"]   for c in all_chunks]
    sources   = [c["source"] for c in all_chunks]
    metadatas = [{"source": s} for s in sources]

    logger.info("[RAG] Cache MISS - rebuilding embeddings")
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()

    _collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas,
    )

    _save_manifest(manifest)

    logger.info(f"Knowledge base ready — {len(all_chunks)} chunks indexed.")
    logger.info("load_knowledge() finished in %.3fs with %d chunks.", time.perf_counter() - load_start, len(all_chunks))
    return len(all_chunks)


# ══════════════════════════════════════════════════════════
#  RETRIEVE — main function called by main.py
# ══════════════════════════════════════════════════════════

def get_relevant_context(query: str, top_k: int = TOP_K) -> str:
    """
    Embeds the query, searches ChromaDB, returns
    the top_k most relevant chunks as a single string.
    """
    if _collection.count() == 0:
        logger.warning("Knowledge base is empty. Run load_knowledge() first.")
        return ""

    query_embedding = _embedder.encode([query]).tolist()

    results = _collection.query(
        query_embeddings = query_embedding,
        n_results        = min(top_k, _collection.count()),
        include          = ["documents", "metadatas"],
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    context = "\n\n---\n\n".join(docs)
    logger.info(f"Retrieved {len(docs)} chunks for query: '{query[:50]}...'")
    return context


# ══════════════════════════════════════════════════════════
#  QUICK TEST  — run: python rag.py
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    total = load_knowledge()
    print(f"\n✅ Loaded {total} chunks\n")

    test_queries = [
        "Who is Manish?",
        "What technologies does Manish know?",
        "What projects has Manish built?",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        ctx = get_relevant_context(q)
        print(f"Context:\n{ctx[:300]}...")
        print("-" * 60)