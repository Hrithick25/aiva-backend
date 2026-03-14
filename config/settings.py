import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Always load the backend .env from the project root, regardless of where uvicorn is started.
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
DOCUMENT_COLLECTION_NAME = "KnowledgeDocuments"
CHUNK_COLLECTION_NAME = "KnowledgeChunks"
CHUNK_RESULTS = 2
ROUTER_MAX_SOURCES = 2
ROUTER_TEXT_LIMIT = 6000
KNOWLEDGE_FILES = {
	"Dataset": os.path.join(BASE_DIR, "Dataset.txt"),
}
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8000

# Audio Processing Settings
AUDIO_SETTINGS = {
    # STT Settings (Groq)
    "stt_provider": "groq_whisper_v3",
    "stt_language": os.getenv("STT_LANGUAGE", "auto"),  # auto, en, ta
    "max_audio_size": int(os.getenv("MAX_AUDIO_SIZE", "10485760")),  # 10MB default
    "max_audio_duration": int(os.getenv("MAX_AUDIO_DURATION", "300")),  # 5 minutes default
    
    # TTS Settings (Gemini)
    "tts_provider": "gemini_tts",
    "default_voice": os.getenv("DEFAULT_VOICE", "Bard"),
    "speech_rate": int(os.getenv("SPEECH_RATE", "150")),  # words per minute
    "default_tts_language": os.getenv("DEFAULT_TTS_LANGUAGE", "en"),
    
    # API Key Rotation
    "enable_key_rotation": os.getenv("ENABLE_KEY_ROTATION", "true").lower() == "true",
    "key_rotation_logging": os.getenv("KEY_ROTATION_LOGGING", "true").lower() == "true",
    
    # Audio processing
    "enable_audio_logging": os.getenv("ENABLE_AUDIO_LOGGING", "false").lower() == "true",
    "audio_cache_enabled": os.getenv("AUDIO_CACHE_ENABLED", "false").lower() == "true",
    "audio_temp_dir": os.path.join(BASE_DIR, "temp", "audio")
}

# Ensure temp directory exists
os.makedirs(AUDIO_SETTINGS["audio_temp_dir"], exist_ok=True)
