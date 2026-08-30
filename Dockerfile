FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_CACHE_DIR=/tmp/manish-portfolio-rag

WORKDIR /app

COPY --chown=user requirements.lock .
RUN pip install --no-cache-dir --no-deps \
    "https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b"
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user main.py .
COPY --chown=user rag.py .
COPY --chown=user system_prompt.py .
COPY --chown=user intent_router.py .
COPY --chown=user query_understanding.py .
COPY --chown=user connectors/ ./connectors/
COPY --chown=user knowledge/ ./knowledge/

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
