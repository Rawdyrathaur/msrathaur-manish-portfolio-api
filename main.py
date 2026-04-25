import os
import io
import time
import logging
import tempfile
import asyncio
import unicodedata
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Local modules ─────────────────────────────────────────
from rag import load_knowledge, get_relevant_context
from system_prompt import build_system_prompt

load_dotenv()

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

class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    reply:       str
    provider:    str
    chunks_used: int


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
            max_tokens=512,
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
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        res = client.chat.complete(
            model="mistral-small-latest",
            messages=msgs,
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
        from together import Together
        client = Together(api_key=api_key)
        res = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=msgs,
            max_tokens=512,
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

def route_llm(msgs: list[dict]) -> tuple[str, str]:
    for name, fn in PROVIDERS:
        logger.info(f"Trying provider: {name}")
        reply = fn(msgs)
        if reply:
            logger.info(f"✅ Success with: {name}")
            return reply, name
    raise HTTPException(
        status_code=503,
        detail="All LLM providers failed. Check your .env API keys.",
    )


# ══════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/ping")
def ping():
    return {"message": "Backend is alive!"}


@app.get("/health")
def health():
    configured = [
        name for name, _ in PROVIDERS
        if os.getenv(f"{name.upper()}_API_KEY")
    ]
    return {
        "status":          "ok" if configured else "degraded",
        "providers_ready": configured,
        "providers_total": len(PROVIDERS),
        "rag":             "ready",
    }


@app.get("/providers")
def providers():
    return {
        name: bool(os.getenv(f"{name.upper()}_API_KEY"))
        for name, _ in PROVIDERS
    }


@app.post("/reload")
def reload():
    total = load_knowledge()
    return {"status": "reloaded", "chunks": total}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    logger.info(f"RAG query: '{req.message[:60]}'")
    rag_context = get_relevant_context(req.message)
    chunks_used = len(rag_context.split("---")) if rag_context else 0
    logger.info(f"RAG returned {chunks_used} chunk(s)")

    system = build_system_prompt(rag_context)
    msgs   = build_messages(system, req.history or [], req.message)
    reply, provider = route_llm(msgs)

    return ChatResponse(reply=reply, provider=provider, chunks_used=chunks_used)


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