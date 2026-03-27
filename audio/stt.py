"""
Speech-to-Text module.

Routing:
  • Tamil mode (language="ta") → Sarvam saarika:v2 (native Tanglish support)
      Falls back to Groq whisper-large-v3-turbo if SARVAM_API_KEY not configured.
  • English / auto mode         → Groq whisper-large-v3-turbo (~400ms)

Tanglish detection:
  • Whisper romanizes Tamil; if result language is "en" but contains known
    Tamil words, we override is_tamil=True so the agent responds in Tanglish.
"""

import asyncio
import logging
from typing import Any, Dict

import io
from groq import Groq
from audio.sarvam import transcribe_tamil, is_configured as sarvam_configured
from agent.groq_llama_agent import _key_manager as _groq_key_manager

logger = logging.getLogger(__name__)

# ── Tanglish signature words (romanized Tamil Whisper outputs) ────────────────
_TANGLISH_SIGNATURE_WORDS = {
    "romba", "nalla", "pakka", "theriyum", "enna", "epdi", "illa", "iruku", "irukku",
    "sollu", "varuva", "varuvaanga", "aana", "aanaanga", "inga", "anga", "konjam",
    "vera", "yenna", "aprom", "seri", "paaru", "irukkum", "pochu", "vandhaanga",
    "pesuvom", "pannuvaanga", "panraanga", "edukka", "theriyuma", "sollunga",
    "nandri", "vanakkam", "ungaluku", "eppdi", "paarunga", "irunga", "solluvaanga",
    "kelunga", "varuvanga", "kidaikum", "kidaikkum", "solren", "ketu", "paapo",
}


class STTProcessor:
    _SUPPORTED_LANGUAGES = {"en", "ta", "hi"}
    _TAMIL_CODES  = {"ta", "tamil"}
    _HINDI_CODES  = {"hi", "hindi"}

    def __init__(self):
        pass   # Groq clients are managed by the shared _groq_key_manager

    def _get_client(self) -> Groq:
        """Return the currently active Groq client (may rotate on 429)."""
        return _groq_key_manager.current_client()

    # ── Public API ────────────────────────────────────────────────────────────

    async def transcribe_audio(self, audio_data: bytes, language: str = "auto") -> Dict[str, Any]:
        """
        Transcribe audio with smart provider routing.

        Tamil mode → Sarvam saarika:v2 (falls back to Groq if key missing).
        English/auto → Groq whisper-large-v3.
        """
        try:
            normalized_lang = self._normalize_language(language)

            # ── Tamil mode: prefer Sarvam ──────────────────────────────────
            if normalized_lang == "ta":
                if sarvam_configured():
                    logger.info("[STT] Tamil mode → Sarvam saarika:v2")
                    fmt  = self._sniff_audio_format(audio_data)
                    # Run blocking HTTP call in thread pool (ot block async loop)
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, transcribe_tamil, audio_data, fmt
                    )
                    if result["success"]:
                        return {
                            "success":           True,
                            "text":              result["text"],
                            "language":          "ta",
                            "confidence":        result["confidence"],
                            "provider":          "sarvam",
                            "is_tamil":          True,
                            "detected_language": "ta",
                        }
                    logger.warning("[STT] Sarvam failed, falling back to Groq for Tamil")
                else:
                    logger.info("[STT] SARVAM_API_KEY not set → using Groq for Tamil")

            # ── English / auto / Groq fallback ────────────────────────────
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_bytes, audio_data, language
            )
            return {
                "success":           True,
                "text":              result["text"].strip(),
                "language":          result["language"],
                "confidence":        result.get("confidence", 0.0),
                "provider":          "groq",
                "is_tamil":          result.get("is_tamil", False),
                "detected_language": result.get("detected_language", "unknown"),
            }

        except Exception as error:
            logger.error(f"STT transcription error: {error}")
            return {
                "success":   False,
                "error":     str(error),
                "text":      "",
                "language":  "unknown",
                "confidence": 0.0,
                "provider":  "groq",
                "is_tamil":  False,
            }

    # ── Groq Whisper (English + fallback) ────────────────────────────────────

    def _normalize_language(self, language: str) -> str:
        if not language:
            return "en"
        normalized = language.strip().lower().replace("_", "-")
        aliases = {
            "en": "en", "en-us": "en", "en-in": "en", "english": "en",
            "ta": "ta", "ta-in": "ta", "tamil":  "ta",
            "hi": "hi", "hi-in": "hi", "hindi":  "hi",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in self._SUPPORTED_LANGUAGES else "en"

    def _transcribe_bytes(self, audio_data: bytes, language: str = "auto") -> Dict[str, Any]:
        """Groq Whisper-large-v3 transcription with automatic key rotation on 429."""
        detected_format = self._sniff_audio_format(audio_data)
        ext_map = {
            "wav": "wav", "mp3": "mp3", "ogg": "ogg",
            "flac": "flac", "m4a": "m4a", "webm": "webm", "unknown": "wav",
        }
        ext             = ext_map.get(detected_format, "wav")
        normalized_lang = self._normalize_language(language)

        n_keys  = max(1, len(_groq_key_manager._keys) if _groq_key_manager._keys else 1)
        rotated = 0
        transcript_text   = ""
        detected_language = "en"

        while True:
            try:
                audio_file = io.BytesIO(audio_data)
                audio_file.name = f"audio.{ext}"
                client = self._get_client()
                kwargs: Dict[str, Any] = {
                    "file":            audio_file,
                    "model":           "whisper-large-v3-turbo",
                    "response_format": "verbose_json",
                    "temperature":     0.0,
                }
                if language != "auto" and normalized_lang in self._SUPPORTED_LANGUAGES:
                    kwargs["language"] = normalized_lang
                    logger.info(f"[STT] Forced language: {normalized_lang}")
                else:
                    logger.info("[STT] Auto language detection")

                resp              = client.audio.transcriptions.create(**kwargs)
                transcript_text   = resp.text or ""
                detected_language = (getattr(resp, "language", None) or "en").strip().lower()
                break   # success — exit retry loop

            except Exception as error:
                err_str = str(error).lower()
                if ("rate" in err_str or "429" in err_str) and rotated < n_keys - 1:
                    logger.warning(
                        f"[STT] Whisper rate-limited on key #{_groq_key_manager._index + 1}, rotating…"
                    )
                    if _groq_key_manager.rotate():
                        rotated += 1
                        continue
                logger.error(f"Groq transcription failed: {error}")
                raise

        # ── Language override checks ──────────────────────────────────────────
        is_tamil_input = detected_language in self._TAMIL_CODES
        is_hindi_input = detected_language in self._HINDI_CODES

        # Tamil Unicode chars
        if not is_tamil_input:
            if any('\u0b80' <= c <= '\u0bff' for c in transcript_text):
                is_tamil_input    = True
                detected_language = "ta"

        # Tanglish signature words (code-switching mis-labeled as English)
        if not is_tamil_input:
            words = set(transcript_text.lower().split())
            if words & _TANGLISH_SIGNATURE_WORDS:
                is_tamil_input    = True
                detected_language = "ta"
                logger.info("[STT] Tanglish signature detected → Tamil mode")

        # Hindi Unicode chars
        if not is_hindi_input and not is_tamil_input:
            if any('\u0900' <= c <= '\u097f' for c in transcript_text):
                is_hindi_input    = True
                detected_language = "hi"

        final_language = (
            "ta" if is_tamil_input else
            "hi" if is_hindi_input else
            "en"
        )
        confidence = 0.92 if (is_tamil_input or is_hindi_input) else 0.95

        logger.info(
            f"[STT] Groq result → lang={final_language}, tamil={is_tamil_input}, "
            f"text='{transcript_text[:60]}'"
        )
        return {
            "text":              transcript_text,
            "confidence":        confidence,
            "language":          final_language,
            "is_tamil":          is_tamil_input,
            "is_hindi":          is_hindi_input,
            "detected_language": detected_language,
        }

    def _sniff_audio_format(self, audio_data: bytes) -> str:
        """Best-effort header sniffing for common audio containers."""
        if not audio_data or len(audio_data) < 4:
            return "unknown"
        header_map = {
            b"RIFF":             "wav",
            b"\xff\xfb":         "mp3",
            b"\xff\xf3":         "mp3",
            b"\xff\xf2":         "mp3",
            b"OggS":             "ogg",
            b"fLaC":             "flac",
            b"ftypM4A":          "m4a",
            b"\x1a\x45\xdf\xa3": "webm",
        }
        for header, fmt in header_map.items():
            if audio_data.startswith(header):
                return fmt
        return "unknown"

    async def validate_audio_format(self, audio_data: bytes) -> Dict[str, Any]:
        if len(audio_data) < 1000:
            return {"valid": False, "error": "Audio data too small"}
        fmt = self._sniff_audio_format(audio_data)
        return {"valid": True, "format": fmt, "size": len(audio_data)}


# ── Global singleton ──────────────────────────────────────────────────────────
_stt_processor = None

def get_stt_processor() -> STTProcessor:
    global _stt_processor
    if _stt_processor is None:
        _stt_processor = STTProcessor()
    return _stt_processor