FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user main.py .
COPY --chown=user rag.py .
COPY --chown=user system_prompt.py .
COPY --chown=user knowledge/ ./knowledge/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
