---
title: Portfolio RAG API
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Production-ready portfolio RAG and voice API
---

# Portfolio RAG API

A FastAPI backend for a grounded portfolio assistant. It automatically indexes configured portfolio content and public repository metadata, retrieves relevant evidence with ChromaDB, and generates answers through a fallback chain of LLM providers.

## What is included

- Automatic portfolio and public GitHub metadata synchronization
- Model-aware chunking, embeddings, retrieval, and reranking
- Groq, Gemini, Cohere, Mistral, and Together provider fallback
- Grounded extractive responses when providers are unavailable
- Prompt-injection boundaries and hidden internal source paths
- Rate limits, request budgets, structured logs, and optional Redis storage
- Speech-to-text and text-to-speech endpoints
- Atomic index replacement, scheduled refresh, and signed webhook updates

> [!CAUTION]
> **Never commit `.env`, API keys, GitHub credentials, or private keys. Configure `RAG_RELOAD_SECRET` and `GITHUB_WEBHOOK_SECRET` before exposing administrative or webhook endpoints. Without an administrative secret, protected endpoints intentionally fail closed.**

## Quick setup

Requirements: Git, Python 3.11, and at least one supported LLM API key for generated responses.

```bash
git clone <repository-url>
cd <repository-directory>

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Edit `.env` and set the minimum configuration:

```dotenv
GROQ_API_KEY="your_api_key"
PORTFOLIO_REPOSITORY="owner/repository"
PORTFOLIO_BRANCH="main"
RAG_RELOAD_SECRET="a_long_random_secret"
```

Groq is the first provider, but any supported provider key can be used. GitHub App credentials or `GITHUB_TOKEN` are optional for authenticated repository access.

Start the API:

```bash
uvicorn main:app --reload --port 8000
```

Verify it:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What projects are included in this portfolio?"}'
```

The first startup can take longer while the embedding model is downloaded and the initial index is created.

## Docker

The image listens on port `7860` and includes the embedding model.

```bash
docker build -t portfolio-rag-api .
docker run --env-file .env -p 7860:7860 portfolio-rag-api
curl http://127.0.0.1:7860/health
```

## Main endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Public service and RAG readiness |
| `POST` | `/chat` | Grounded portfolio question answering |
| `POST` | `/whisper` | Audio transcription |
| `GET` | `/speak?text=...` | MP3 text-to-speech |
| `GET` | `/providers` | Protected provider status |
| `POST` | `/reload` | Protected full index refresh |
| `POST` | `/webhook/github` | Signed GitHub synchronization events |

Protected administrative requests require this header:

```text
X-Reload-Secret: <RAG_RELOAD_SECRET>
```

## Development checks

```bash
python -m pip install -r requirements-dev.txt
pytest -q --no-cov
python evals/run_rag_eval.py
```

The quality workflow also runs dependency auditing and secret scanning on every push and pull request.

## Code map

| Path | Responsibility |
|---|---|
| `main.py` | API routes, provider routing, limits, and voice services |
| `rag.py` | Index lifecycle, embeddings, retrieval, and reranking |
| `connectors/` | Portfolio and GitHub data ingestion |
| `system_prompt.py` | Grounding and trust-boundary instructions |
| `intent_router.py` | Greeting, portfolio, follow-up, and off-topic routing |
| `knowledge/` | Bundled fallback knowledge |
| `tests/` and `evals/` | Automated tests and golden RAG evaluations |

For production, use Redis through `REDIS_URL` so rate limits and daily budgets are shared across workers and survive restarts. Keep `/health` public, but never expose administrative secrets in frontend code.
