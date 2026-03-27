import sys
import os
import logging
import threading

# Ensure the backend directory is on sys.path when running from any cwd
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.websocket_handler import router as ws_router
from config.settings import WEBSOCKET_HOST, WEBSOCKET_PORT, AUDIO_SETTINGS

# NOTE: rag_faiss.retriever and audio.manager are imported LAZILY inside
# _background_warmup() because their module-level `import faiss` and
# heavy initialisers can take 5-15s on Render's free tier, blocking
# the port from binding in time.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AIVA - AI Virtual Assistant",
    description="AI Virtual Assistant for Sri Eshwar College of Engineering",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)



# ── Warm-up status tracking ──────────────────────────────────────────────────
_warmup_complete = False
_warmup_status = "initializing"


def _background_warmup():
    """
    Run ALL heavy initialisation in a background thread so uvicorn
    binds the port instantly and Render's port-scanner succeeds.
    """
    global _warmup_complete, _warmup_status

    try:
        # ── 1. Load FAISS index ────────────────────────────────────────
        _warmup_status = "loading FAISS index"
        logger.info("[STARTUP-BG] Loading FAISS index...")
        from rag_faiss.retriever import _ensure_loaded as _load_faiss
        _load_faiss()
        logger.info("[STARTUP-BG] ✅ FAISS index ready")

        # ── 2. Pre-warm Groq Llama client ──────────────────────────────
        _warmup_status = "initializing Groq LLM client"
        logger.info("[STARTUP-BG] Initializing Groq LLM client...")
        try:
            from agent.groq_llama_agent import _get_groq_client
            _get_groq_client()
            logger.info("[STARTUP-BG] ✅ Groq client cached")
        except Exception as e:
            logger.warning(f"[STARTUP-BG] Groq client init failed (non-fatal): {e}")

        # ── 3. Audio manager ───────────────────────────────────────────
        _warmup_status = "initializing audio pipeline"
        logger.info("[STARTUP-BG] Initializing audio pipeline...")
        try:
            from audio.manager import get_audio_manager as _get_am
            audio_mgr = _get_am()
            caps = audio_mgr.get_supported_formats()
            logger.info(f"[STARTUP-BG] ✅ STT: {caps['stt']['provider']}")
            logger.info(f"[STARTUP-BG] ✅ TTS: {caps['tts']['provider']}")
        except Exception as e:
            logger.warning(f"[STARTUP-BG] Audio init failed (non-fatal): {e}")

        _warmup_complete = True
        _warmup_status = "ready"
        logger.info(
            "[STARTUP-BG] 🚀 AIVA ready! "
            "Embedding: Gemini cloud API | LLM: Groq llama-3.1-8b | "
            "STT: Groq whisper | TTS: Edge/Sarvam"
        )
    except Exception as e:
        _warmup_status = f"error: {e}"
        logger.error(f"[STARTUP-BG] ❌ Background warm-up crashed: {e}")


@app.on_event("startup")
async def startup():
    """
    Startup fires BEFORE uvicorn announces the port as open.
    We MUST return immediately so the port binds within Render's
    ~60-second scan window.  All heavy work goes to a daemon thread.
    """


    logger.info("[STARTUP] 🔥 Launching background warm-up thread...")
    t = threading.Thread(target=_background_warmup, daemon=True)
    t.start()
    logger.info("[STARTUP] ✅ Port is binding NOW — warm-up continues in background")


@app.get("/")
async def health_check():
    """Health check endpoint — must respond fast even during warm-up."""
    base = {
        "status": "healthy",
        "service": "AIVA AI Virtual Assistant",
        "version": "3.1.0",
        "warmup_complete": _warmup_complete,
        "warmup_status": _warmup_status,
    }

    if not _warmup_complete:
        # Lightweight response so Render health-checks pass immediately
        return base

    # Full response once everything is loaded
    try:
        from audio.manager import get_audio_manager
        audio_manager = get_audio_manager()
        capabilities = audio_manager.get_supported_formats()
    except Exception:
        capabilities = {}

    return {
        **base,
        "embedding": "Gemini embedding-001 (cloud API)",
        "llm": "Groq llama-3.1-8b-instant",
        "stt_en": "Groq whisper-large-v3-turbo",
        "stt_ta": "Sarvam saarika:v2",
        "tts_en": "Edge TTS",
        "tts_ta": "Sarvam bulbul:v2",
        "features": {
            "text_chat": True,
            "speech_to_text": True,
            "text_to_speech": True,
            "audio_conversation": True,
            "response_cache": True,
        },
        "audio_capabilities": capabilities,
        "endpoints": {
            "websocket": "/ws",
            "health": "/",
            "audio_info": "/audio/info",
            "docs": "/docs"
        }
    }


@app.get("/audio/info")
async def get_audio_info():
    """Get detailed audio processing information"""
    from audio.manager import get_audio_manager
    audio_manager = get_audio_manager()
    return {
        "capabilities": audio_manager.get_supported_formats(),
        "settings": {
            "stt_provider":       AUDIO_SETTINGS["stt_provider"],
            "stt_tamil_provider": AUDIO_SETTINGS["stt_tamil_provider"],
            "tts_provider":       AUDIO_SETTINGS["tts_provider"],
            "tts_tamil_provider": AUDIO_SETTINGS["tts_tamil_provider"],
            "max_audio_size":     AUDIO_SETTINGS["max_audio_size"],
            "max_duration":       AUDIO_SETTINGS["max_audio_duration"],
            "supported_languages": ["en", "ta", "hi"],
        }
    }


@app.get("/audio/voices/{language}")
async def get_voices(language: str = "en"):
    """Get available voices for a language"""
    from audio.manager import get_audio_manager
    audio_manager = get_audio_manager()
    return await audio_manager.get_voice_options(language)


if __name__ == "__main__":
    import uvicorn
    # Read PORT live from env so Render's injected $PORT is always honoured.
    # WEBSOCKET_PORT was resolved at import time from .env and may be stale.
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"[MAIN] Starting AIVA on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

