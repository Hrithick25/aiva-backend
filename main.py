import sys
import os
import logging
import asyncio

# Ensure the backend directory is on sys.path when running from any cwd
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.websocket_handler import router as ws_router
from rag_faiss.retriever import _ensure_loaded as load_faiss_index
from config.settings import WEBSOCKET_HOST, WEBSOCKET_PORT, AUDIO_SETTINGS
from audio.manager import get_audio_manager

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


@app.on_event("startup")
async def startup():
    """
    Pre-warm ALL heavy components at startup so the first user request
    is fast. Without pre-warming, the first embed call loads the 438MB
    BGE model (~3-5 seconds), which would blow the <3s latency budget.
    """
    logger.info("[STARTUP] 🔥 Pre-warming AIVA components...")

    # ── 1. Load FAISS index ────────────────────────────────────────────
    logger.info("[STARTUP] Loading FAISS index...")
    load_faiss_index()
    logger.info("[STARTUP] ✅ FAISS index ready")

    # ── 2. Pre-warm BGE embedding model ───────────────────────────────
    # Run in executor so we don't block the event loop during model load.
    logger.info("[STARTUP] Loading BGE embedding model (BAAI/bge-base-en-v1.5)...")
    loop = asyncio.get_running_loop()
    try:
        from rag_faiss.embedder import embed_query as _warmup_embed
        await loop.run_in_executor(None, _warmup_embed, "warmup query")
        logger.info("[STARTUP] ✅ BGE embedding model warm and ready")
    except Exception as e:
        logger.warning(f"[STARTUP] BGE pre-warm failed (non-fatal): {e}")

    # ── 3. Pre-warm Groq Llama client ─────────────────────────────────
    logger.info("[STARTUP] Initializing Groq LLM client...")
    try:
        from agent.groq_llama_agent import _get_groq_client
        _get_groq_client()
        logger.info("[STARTUP] ✅ Groq client cached")
    except Exception as e:
        logger.warning(f"[STARTUP] Groq client init failed (non-fatal): {e}")

    # ── 4. Audio manager ──────────────────────────────────────────────
    logger.info("[STARTUP] Initializing audio pipeline...")
    audio_manager = get_audio_manager()
    capabilities = audio_manager.get_supported_formats()
    logger.info(f"[STARTUP] ✅ STT: {capabilities['stt']['provider']}")
    logger.info(f"[STARTUP] ✅ TTS: {capabilities['tts']['provider']}")

    logger.info(
        "[STARTUP] 🚀 AIVA ready! All components pre-warmed. "
        "First request latency: STT ~400ms | RAG ~85ms | LLM ~300ms | TTS stream ~300ms"
    )


@app.get("/")
async def health_check():
    """Health check endpoint"""
    audio_manager = get_audio_manager()
    capabilities = audio_manager.get_supported_formats()
    return {
        "status": "healthy",
        "service": "AIVA AI Virtual Assistant",
        "version": "3.1.0",
        "embedding": "BAAI/bge-base-en-v1.5 (local, no API)",
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

