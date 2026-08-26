# Manish Portfolio API

Production FastAPI backend for the AI assistant on [manishrathaur.tech](https://www.manishrathaur.tech).

## How it works

1. On startup, the service fetches `src/content/**` directly from `Rawdyrathaur/portfolio`.
2. It fetches Manish's public GitHub profile and public repository metadata, READMEs, languages, and file trees.
3. Content is chunked, embedded with `all-MiniLM-L6-v2`, and indexed in a local cosine-similarity ChromaDB collection.
4. `/chat` retrieves and reranks relevant public chunks, then generates a grounded response through the first available LLM provider.
5. If every LLM provider is unavailable, the API returns a grounded extractive answer instead of failing.

The index refreshes hourly by default. A signed GitHub push webhook or protected `/reload` call refreshes it immediately. Bundled Markdown is used only when the live portfolio source cannot be fetched.

## API

- `POST /chat` — grounded portfolio question answering
- `GET /health` — API, provider configuration, and index status
- `GET /providers` — configured provider summary
- `POST /reload` — refresh index; protected by `X-Reload-Secret` when configured
- `POST /webhook/github` — signed GitHub webhook receiver
- `POST /whisper` — speech-to-text
- `GET /speak?text=...` — text-to-speech

## Configuration

Copy `.env.example` to `.env`. At least one LLM provider is recommended. GitHub App credentials allow the service to read the private portfolio source repository, while only whitelisted portfolio content and public repositories are indexed.

Important variables:

- `GROQ_API_KEY`, `GROQ_CHAT_MODEL`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `GITHUB_APP_ID`, `GITHUB_INSTALLATION_ID`, `GITHUB_PRIVATE_KEY`
- `GITHUB_WEBHOOK_SECRET`, `RAG_RELOAD_SECRET`
- `PORTFOLIO_REPOSITORY`, `PORTFOLIO_BRANCH`
- `RAG_REFRESH_SECONDS`

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Test

```bash
pytest -q
```

The Docker image is configured for deployment on Hugging Face Spaces at port `7860`.
