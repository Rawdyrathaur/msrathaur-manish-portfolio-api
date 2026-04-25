import os
import glob
import logging
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────
KNOWLEDGE_DIR   = "knowledge"        # folder with all .md files
EMBED_MODEL     = "all-MiniLM-L6-v2" # small, fast, local — no API needed
TOP_K           = 3                  # how many chunks to retrieve per query
COLLECTION_NAME = "manish_portfolio"

# ── Load model once at startup ────────────────────────────
logger.info("Loading embedding model...")
_embedder = SentenceTransformer(EMBED_MODEL)
logger.info("Embedding model loaded.")

# ── ChromaDB in-memory client ─────────────────────────────
_client     = chromadb.Client()
_collection = _client.get_or_create_collection(COLLECTION_NAME)


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
    md_files = sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md")))

    if not md_files:
        logger.warning(f"No .md files found in '{KNOWLEDGE_DIR}/' folder.")
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
        return 0

    # Clear old data before reloading
    global _collection
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

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()

    _collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas,
    )

    logger.info(f"Knowledge base ready — {len(all_chunks)} chunks indexed.")
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