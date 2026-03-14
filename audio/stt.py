"""Speech-to-Text module using Groq."""

import asyncio
import logging
from typing import Any, Dict
import io

from groq import Groq
from .stt_post_processor import get_stt_post_processor

logger = logging.getLogger(__name__)


class STTProcessor:
    _SUPPORTED_LANGUAGES = {"en", "ta"}
    # Tamil language codes that Whisper may return
    _TAMIL_CODES = {"ta", "tamil"}

    def __init__(self):
        """Initialize the STT processor."""
        self._client = None
        self._current_key = None  # Track key for rotation support

    def _get_client(self) -> Groq:
        """Create or reuse a Groq client, rotating when the key changes."""
        import os
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("No Groq API key available")

        # Re-create client only when the key rotates
        if self._client is None or api_key != self._current_key:
            self._client = Groq(api_key=api_key)
            self._current_key = api_key
        return self._client

    async def transcribe_audio(self, audio_data: bytes, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio with single-pass auto-detection (no double-pass)."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._transcribe_bytes, audio_data, language)
            
            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result["language"],
                "confidence": result.get("confidence", 0.0),
                "provider": "groq",
                "is_tamil": result.get("is_tamil", False),
                "detected_language": result.get("detected_language", "unknown"),
            }
        except Exception as error:
            logger.error(f"Groq STT transcription error: {error}")
            return {
                "success": False,
                "error": str(error),
                "text": "",
                "language": "unknown",
                "confidence": 0.0,
                "provider": "groq",
                "is_tamil": False,
            }

    def _normalize_language(self, language: str) -> str:
        """Normalize app language aliases to the codes expected by downstream services."""
        if not language:
            return "en"

        normalized = language.strip().lower().replace("_", "-")
        aliases = {
            "en": "en",
            "en-us": "en",
            "en-in": "en",
            "english": "en",
            "ta": "ta",
            "ta-in": "ta",
            "tamil": "ta",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in self._SUPPORTED_LANGUAGES:
            return normalized
        return "en"

    def _transcribe_bytes(self, audio_data: bytes, language: str = "auto") -> Dict[str, Any]:
        """Single-pass transcription using Whisper's native language auto-detection.
        
        OPTIMIZED: Removed the old double-pass approach that was adding 500-1500ms
        for Tamil queries. Whisper auto-detects the language in one call.
        """
        try:
            client = self._get_client()

            # Try to infer container/codec from headers for correct filename extension.
            # Some providers use the filename to detect encoding.
            detected_format = self._sniff_audio_format(audio_data)
            
            # Create a file-like object from audio bytes
            audio_file = io.BytesIO(audio_data)
            ext = {
                "wav": "wav",
                "mp3": "mp3",
                "ogg": "ogg",
                "flac": "flac",
                "m4a": "m4a",
                "webm": "webm",
                "unknown": "wav",
            }.get(detected_format, "wav")
            audio_file.name = f"audio.{ext}"
            
            # Determine if we should force a specific language
            normalized_lang = self._normalize_language(language)
            
            # Build transcription kwargs
            transcribe_kwargs = {
                "file": audio_file,
                "model": "whisper-large-v3-turbo",
                "response_format": "verbose_json",
                "temperature": 0.0,
            }
            
            # If caller explicitly requested a specific language, use it
            # Otherwise let Whisper auto-detect (single pass, no double-call)
            if language != "auto" and normalized_lang in self._SUPPORTED_LANGUAGES:
                transcribe_kwargs["language"] = normalized_lang
                logger.info(f"Transcribing with forced language: {normalized_lang}")
            else:
                # AUTO-DETECT: Let Whisper figure out the language in ONE call
                logger.info("Transcribing with auto language detection (single pass)...")
            
            transcript_response = client.audio.transcriptions.create(**transcribe_kwargs)
            
            transcript_text = transcript_response.text or ""
            
            # Detect language from Whisper's response metadata
            detected_language = getattr(transcript_response, 'language', None) or "en"
            detected_language = detected_language.strip().lower()
            
            # Check if Tamil was detected
            is_tamil_input = detected_language in self._TAMIL_CODES
            
            # Also check for Tamil Unicode characters as a fallback signal
            if not is_tamil_input:
                has_tamil_chars = any('\u0b80' <= char <= '\u0bff' for char in transcript_text)
                if has_tamil_chars:
                    is_tamil_input = True
                    detected_language = "ta"
            
            # Normalize the final language code
            final_language = "ta" if is_tamil_input else "en"
            
            # Confidence based on detection method
            confidence = 0.92 if is_tamil_input else 0.95
            
            logger.info(f"STT Result (single-pass) - Language: {final_language}, "
                       f"Tamil: {is_tamil_input}, Text: '{transcript_text[:50]}...'")
            
            return {
                "text": transcript_text,
                "confidence": confidence,
                "language": final_language,
                "is_tamil": is_tamil_input,
                "detected_language": detected_language,
            }
            
        except Exception as error:
            logger.error(f"Groq transcription failed: {error}")
            raise

    async def validate_audio_format(self, audio_data: bytes) -> Dict[str, Any]:
        """Validate whether the input looks like supported audio data."""
        try:
            if len(audio_data) < 1000:
                return {
                    "valid": False,
                    "error": "Audio data too small",
                }

            headers = {
                b"RIFF": "wav",
                b"\xff\xfb": "mp3",
                b"\xff\xf3": "mp3",
                b"\xff\xf2": "mp3",
                b"OggS": "ogg",
                b"fLaC": "flac",
                b"ftypM4A": "m4a",
                b"\x1a\x45\xdf\xa3": "webm",
            }

            audio_format = "unknown"
            for header, fmt in headers.items():
                if audio_data.startswith(header):
                    audio_format = fmt
                    break

            return {
                "valid": True,
                "format": audio_format,
                "size": len(audio_data),
            }
        except Exception as error:
            return {
                "valid": False,
                "error": str(error),
            }

    def _sniff_audio_format(self, audio_data: bytes) -> str:
        """Best-effort header sniffing for common audio containers."""
        try:
            if not audio_data or len(audio_data) < 4:
                return "unknown"

            headers = {
                b"RIFF": "wav",
                b"\xff\xfb": "mp3",
                b"\xff\xf3": "mp3",
                b"\xff\xf2": "mp3",
                b"OggS": "ogg",
                b"fLaC": "flac",
                b"ftypM4A": "m4a",
                b"\x1a\x45\xdf\xa3": "webm",
            }

            for header, fmt in headers.items():
                if audio_data.startswith(header):
                    return fmt

            return "unknown"
        except Exception:
            return "unknown"


_stt_processor = None


def get_stt_processor() -> STTProcessor:
    """Get the global STT processor instance."""
    global _stt_processor
    if _stt_processor is None:
        _stt_processor = STTProcessor()
    return _stt_processor