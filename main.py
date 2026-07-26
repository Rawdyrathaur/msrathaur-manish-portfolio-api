import os
import io
import time
import logging
import tempfile
import asyncio
import hmac
import hashlib
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Env Var Validation ────────────────────────────────────
import sys

REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "GITHUB_APP_ID",
    "GITHUB_INSTALLATION_ID",
    "GITHUB_WEBHOOK_SECRET",
    "GITHUB_PRIVATE_KEY"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"\n❌ CRITICAL ERROR: Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
    print("Please create a .env file based on .env.example before running the application.\n", file=sys.stderr)
    sys.exit(1)

# ── Local modules ─────────────────────────────────────────
from rag import load_knowledge, get_relevant_context, upsert_github_repo, delete_github_repo
from system_prompt import build_system_prompt
from intent_router import classify_intent


# ── Logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  STARTUP — load knowledge base once when server starts
# ══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — loading knowledge base...")
    total = load_knowledge()
    if total == 0:
        logger.warning("⚠️  No chunks loaded. Check your knowledge/ folder.")
    else:
        logger.info(f"✅ Knowledge base ready — {total} chunks indexed.")
    yield
    logger.info("🛑 Shutting down.")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Manish Portfolio API",
    description="RAG-powered portfolio chatbot — built by Manish",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.manishrathaur.tech",
        "https://manishrathaur.tech",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════
#  RATE LIMITING — per IP, in-memory
# ══════════════════════════════════════════════════════════

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 30))
RATE_LIMIT_WINDOW   = int(os.getenv("RATE_LIMIT_WINDOW",   60))

_rate_store: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(ip: str) -> None:
    now          = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    timestamps   = [t for t in _rate_store[ip] if t > window_start]
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s.",
        )
    timestamps.append(now)
    _rate_store[ip] = timestamps


# ══════════════════════════════════════════════════════════
#  REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════

class Source(BaseModel):
    title: str
    type: str
    url: str
    source_type: str = "portfolio"
    content_type: str = "unknown"
    visibility: str = "public"
    trust_level: str = "verified"
    timestamp: Optional[str] = None
    last_updated: Optional[str] = None

class RelatedLink(BaseModel):
    title: str
    url: str

class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    answer:      str
    provider:    str
    chunks_used: int
    sources:     List[Source] = []
    related:     List[RelatedLink] = []
    confidence:  str = "high"


# ══════════════════════════════════════════════════════════
#  MESSAGE BUILDER
# ══════════════════════════════════════════════════════════

def build_messages(
    system:  str,
    history: List[ChatMessage],
    message: str,
) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    for h in history[-10:]:
        msgs.append({"role": h.role, "content": h.content})
    msgs.append({"role": "user", "content": message})
    return msgs


# ══════════════════════════════════════════════════════════
#  TTS HELPER — cleans text before sending to Edge TTS
# ══════════════════════════════════════════════════════════

def clean_for_tts(text: str) -> str:
    """Removes special unicode characters that break Edge TTS."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "-",   # bullet
        "\u00b7": "-",   # middle dot
        "\u2012": "-",   # figure dash
        "\u2015": "-",   # horizontal bar
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


# ══════════════════════════════════════════════════════════
#  LLM PROVIDERS — tried in order, first success wins
# ══════════════════════════════════════════════════════════

def try_groq(msgs: list[dict]) -> str | None:
    """Primary — Groq (llama-3.3-70b) — 14,400 req/day free"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        from groq import Groq
        client = Groq(api_key=api_key)
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            max_tokens=800,
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq failed: {e}")
        return None


def try_gemini(msgs: list[dict]) -> str | None:
    """Fallback 1 — Google Gemini 2.0 Flash — 1,500 req/day free"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        system       = msgs[0]["content"]
        history_msgs = []
        for m in msgs[1:-1]:
            history_msgs.append({
                "role":  "user" if m["role"] == "user" else "model",
                "parts": [m["content"]],
            })
        user_message = msgs[-1]["content"]
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system,
        )
        chat = model.start_chat(history=history_msgs)
        res  = chat.send_message(user_message)
        return res.text
    except Exception as e:
        logger.warning(f"Gemini failed: {e}")
        return None


def try_cohere(msgs: list[dict]) -> str | None:
    """Fallback 2 — Cohere Command-R — 1,000 req/day free"""
    try:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            return None
        import cohere
        client = cohere.ClientV2(api_key=api_key)
        res = client.chat(
            model="command-r-plus",
            messages=msgs,
        )
        return res.message.content[0].text
    except Exception as e:
        logger.warning(f"Cohere failed: {e}")
        return None


def try_mistral(msgs: list[dict]) -> str | None:
    """Fallback 3 — Mistral Small — free tier"""
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")
        res = client.chat.completions.create(
            model="mistral-small-latest",
            messages=msgs,
            max_tokens=800,
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.warning(f"Mistral failed: {e}")
        return None


def try_together(msgs: list[dict]) -> str | None:
    """Fallback 4 — Together AI (Meta Llama) — $1 free credit"""
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.together.ai/v1")
        res = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=msgs,
            max_tokens=800,
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.warning(f"Together failed: {e}")
        return None


# ── Bifrost router ────────────────────────────────────────
PROVIDERS = [
    ("Groq",     try_groq),
    ("Gemini",   try_gemini),
    ("Cohere",   try_cohere),
    ("Mistral",  try_mistral),
    ("Together", try_together),
]


def configured_providers() -> list[str]:
    return [
        name for name, _ in PROVIDERS
        if os.getenv(f"{name.upper()}_API_KEY")
    ]


def configured_provider_hint() -> str:
    return ", ".join(configured_providers()) or "none"

def route_llm(msgs: list[dict]) -> tuple[str, str]:
    route_start = time.perf_counter()
    configured = configured_providers()

    if not configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM provider API keys are configured. Set at least one of: "
                "GROQ_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, MISTRAL_API_KEY, TOGETHER_API_KEY."
            ),
        )

    for name, fn in PROVIDERS:
        logger.info(f"Trying provider: {name}")
        reply = fn(msgs)
        if reply:
            logger.info(f"✅ Success with: {name}")
            logger.info("route_llm() finished in %.3fs via %s.", time.perf_counter() - route_start, name)
            return reply, name
    logger.info("route_llm() finished in %.3fs with no provider success.", time.perf_counter() - route_start)
    raise HTTPException(
        status_code=503,
        detail=(
            "Configured LLM providers failed. "
            f"Configured: {configured_provider_hint()}. "
            "Check provider credentials, quotas, and upstream service status."
        ),
    )


# ══════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/ping")
def ping():
    return {"message": "Backend is alive!"}


@app.get("/health")
def health():
    configured = configured_providers()
    return {
        "status": "ok" if configured else "degraded",
        "providers_ready": configured,
        "providers_total": len(PROVIDERS),
        "rag": "ready",
    }


@app.head("/health")
def health_head():
    return Response(status_code=200)


@app.get("/providers")
def providers():
    return {
        name: name in configured_providers()
        for name, _ in PROVIDERS
    }


@app.post("/reload")
def reload():
    total = load_knowledge()
    return {"status": "reloaded", "chunks": total}


@app.post("/webhook/github")
async def github_webhook(request: Request):
    """Handles GitHub App webhook events to keep the RAG index fresh."""
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
        
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature")
        
    body = await request.body()
    
    # Verify HMAC signature
    expected_mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    expected_sig = f"sha256={expected_mac}"
    if not hmac.compare_digest(signature, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid signature")
        
    event = request.headers.get("X-GitHub-Event")
    payload = await request.json()
    
    if event == "repository":
        action = payload.get("action")
        repo_data = payload.get("repository", {})
        repo_name = repo_data.get("name")
        
        if action in ["created", "edited", "publicized", "unarchived"]:
            success = upsert_github_repo(repo_data)
            return {"status": "upserted" if success else "failed", "repo": repo_name}
            
        elif action in ["deleted", "archived", "privatized"]:
            success = delete_github_repo(repo_name)
            return {"status": "deleted" if success else "failed", "repo": repo_name}
            
    elif event == "ping":
        return {"status": "pong"}
        
    return {"status": "ignored", "event": event}



@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")
    check_rate_limit(client_ip)

    logger.info(f"Chat query: '{req.message[:60]}'")
    
    # ── 1. Intent Classification ──
    intent = classify_intent(req.message, req.history)
    
    # ── 2. Static Responses for Fluff ──
    if intent == "GREETING":
        reply = "Hi! I'm Manish Rathaur's AI portfolio assistant. I can answer questions about his projects, GitHub repositories, technical skills, experience, and the technologies he has worked with. What would you like to know?"
        return ChatResponse(
            answer=reply,
            provider="ROUTER",
            chunks_used=0,
            sources=[],
            related=[],
            confidence="high"
        )
        
    if intent == "OFF_TOPIC":
        reply = "I'm specialized in answering questions about Manish Rathaur's professional portfolio, GitHub projects, technical skills, and experience. I can't reliably answer unrelated topics, but I'd be happy to help you explore his work."
        return ChatResponse(
            answer=reply,
            provider="ROUTER",
            chunks_used=0,
            sources=[],
            related=[],
            confidence="high"
        )

    # ── 3. RAG Pipeline for Portfolio Queries & Follow-ups ──
    rag_context, sources, best_distance = get_relevant_context(req.message)
    chunks_used = len(rag_context.split("---")) if rag_context else 0
    
    logger.info(f"RAG returned {chunks_used} chunk(s) with best_distance={best_distance:.3f}")

    if best_distance > 1.2 or not rag_context:
        logger.warning(f"No good matches found for query (best dist: {best_distance})")
        reply = "I couldn't find verified information about that in Manish's connected GitHub repositories or portfolio data. I prefer not to guess. You can ask me about his projects, technologies, repositories, or experience."
        return ChatResponse(
            answer=reply,
            provider="ROUTER",
            chunks_used=0,
            sources=[],
            related=[],
            confidence="low"
        )

    system = build_system_prompt(rag_context)
    msgs   = build_messages(system, req.history or [], req.message)
    reply, provider = route_llm(msgs)

    structured_sources = [Source(**s) for s in sources]
    
    unique_links = []
    seen_urls = set()
    for s in sources:
        url = s.get("url", "/")
        if url != "/" and url not in seen_urls:
            seen_urls.add(url)
            unique_links.append({"title": s.get("title", "Link"), "url": url})
            
    structured_related = [RelatedLink(**r) for r in unique_links]

    return ChatResponse(
        answer=reply, 
        provider=provider, 
        chunks_used=chunks_used,
        sources=structured_sources,
        related=structured_related,
        confidence="high"
    )


# ══════════════════════════════════════════════════════════
#  STT — Groq Whisper
# ══════════════════════════════════════════════════════════

@app.post("/whisper")
async def whisper(audio: UploadFile = File(...)):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="Groq API key not set.")

        from groq import Groq
        client   = Groq(api_key=api_key)
        contents = await audio.read()

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=("recording.webm", f, "audio/webm"),
            )

        os.unlink(tmp_path)
        return {"transcript": transcription.text}

    except Exception as e:
        logger.error(f"Whisper failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════
#  TTS — Microsoft Edge TTS (unlimited, natural, free)
# ══════════════════════════════════════════════════════════

@app.get("/speak")
async def speak(text: str):
    try:
        import edge_tts

        voice      = "en-US-GuyNeural"   # Young male, clear and natural
        clean_text = clean_for_tts(text)

        communicate  = edge_tts.Communicate(clean_text, voice, rate="+15%")
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)

        if audio_buffer.getbuffer().nbytes == 0:
            raise ValueError("Edge TTS returned empty audio.")

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-cache"},
        )

    except Exception as e:
        logger.error(f"Edge TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))