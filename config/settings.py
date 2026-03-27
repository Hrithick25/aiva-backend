import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Always load the backend .env from the project root.
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ── Server ────────────────────────────────────────────────────────────
WEBSOCKET_HOST = os.getenv("HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("PORT", "8000"))

# ── Audio Processing Settings ─────────────────────────────────────────
AUDIO_SETTINGS = {
    # STT: Groq Whisper-large-v3 for English; Sarvam saarika:v2 for Tamil
    "stt_provider":        "groq_whisper_v3",
    "stt_tamil_provider":  "sarvam",
    "supported_languages": ["en", "ta", "hi"],
    "stt_language":        os.getenv("STT_LANGUAGE", "auto"),
    "max_audio_size":      int(os.getenv("MAX_AUDIO_SIZE", "10485760")),
    "max_audio_duration":  int(os.getenv("MAX_AUDIO_DURATION", "300")),

    # TTS: Edge TTS for English; Sarvam bulbul:v2 for Tamil
    "tts_provider":          "edge_tts",
    "tts_tamil_provider":    "sarvam",
    "default_voice":         os.getenv("DEFAULT_VOICE", "en-US-JennyNeural"),
    "speech_rate":           int(os.getenv("SPEECH_RATE", "150")),
    "default_tts_language":  os.getenv("DEFAULT_TTS_LANGUAGE", "en"),

    # Audio processing flags
    "enable_audio_logging":  os.getenv("ENABLE_AUDIO_LOGGING", "false").lower() == "true",
    "audio_cache_enabled":   os.getenv("AUDIO_CACHE_ENABLED", "false").lower() == "true",
    "audio_temp_dir":        os.path.join(BASE_DIR, "temp", "audio"),
    # Key rotation — Groq keys
    "enable_key_rotation":   True,
    "api_key_rotation":      True,
}

# Ensure temp directory exists
os.makedirs(AUDIO_SETTINGS["audio_temp_dir"], exist_ok=True)
