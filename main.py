import sys
import os
import logging

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
    title="Chopper AI Agent", 
    description="AI Agent with Speech-to-Text and Text-to-Speech capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    logger.info("[STARTUP] Initializing Chopper AI Agent...")
    
    # Load FAISS index
    logger.info("[STARTUP] Loading FAISS index...")
    load_faiss_index()
    logger.info("[STARTUP] ✅ FAISS index loaded")
    
    # Initialize audio manager
    logger.info("[STARTUP] Initializing audio processing...")
    audio_manager = get_audio_manager()
    
    # Log audio capabilities
    capabilities = audio_manager.get_supported_formats()
    logger.info(f"[STARTUP] ✅ Audio STT: {capabilities['stt']['provider']}")
    logger.info(f"[STARTUP] ✅ Audio TTS: {capabilities['tts']['provider']}")
    logger.info(f"[STARTUP] ✅ Supported languages: {capabilities['stt']['languages']}")
    
    logger.info("[STARTUP] 🚀 Ready to serve!")


@app.get("/")
async def health_check():
    """Health check endpoint"""
    audio_manager = get_audio_manager()
    capabilities = audio_manager.get_supported_formats()

    return {
        "status": "healthy",
        "service": "Chopper AI Agent",
        "version": "2.0.0",
        "features": {
            "text_chat": True,
            "speech_to_text": True,
            "text_to_speech": True,
            "audio_conversation": True,
            "api_key_rotation": True
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
            "stt_provider": AUDIO_SETTINGS["stt_provider"],
            "tts_provider": AUDIO_SETTINGS["tts_provider"],
            "max_audio_size": AUDIO_SETTINGS["max_audio_size"],
            "max_duration": AUDIO_SETTINGS["max_audio_duration"],
            "supported_languages": ["en", "ta", "hi"],
            "api_key_rotation": AUDIO_SETTINGS["enable_key_rotation"]
        }
    }


@app.get("/audio/voices/{language}")
async def get_voices(language: str = "en"):
    """Get available voices for a language"""
    audio_manager = get_audio_manager()
    return await audio_manager.get_voice_options(language)


if __name__ == "__main__":
    import uvicorn
    logger.info(f"[MAIN] Starting server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    uvicorn.run(
        "main:app", 
        host=WEBSOCKET_HOST, 
        port=WEBSOCKET_PORT, 
        reload=True,
        log_level="info"
    )
