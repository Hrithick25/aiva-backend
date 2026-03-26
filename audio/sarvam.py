"""
sarvam.py — Sarvam AI STT + TTS for Tamil / Tanglish mode ONLY.

Used when:
  - STT: user has Tamil mode selected (language="ta") → saarika:v2
  - TTS: response is in Tamil mode → bulbul:v2

English mode:
  - STT: Groq Whisper-large-v3-turbo
  - TTS: Microsoft Edge TTS

API:
  - Base URL: https://api.sarvam.ai
  - Auth header: api-subscription-key: <SARVAM_API_KEY>

Models:
  - STT: saarika:v2 — purpose-built Indian ASR, native Tanglish support
  - TTS: bulbul:v2 — natural Tanglish/Tamil TTS

Setup:
  1. Get free API key at https://www.sarvam.ai/
  2. Set in .env: SARVAM_API_KEY=your_key_here
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_SARVAM_BASE   = "https://api.sarvam.ai"
_STT_ENDPOINT  = f"{_SARVAM_BASE}/speech-to-text"
_TTS_ENDPOINT  = f"{_SARVAM_BASE}/text-to-speech"

# Persistent session reuses TCP connections across calls
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Content-Type": "application/json"})
    return _session


def _get_api_key() -> str:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SARVAM_API_KEY not set in .env")
    return key


# ── STT (Speech → Text) ─────────────────────────────────────────────────────

def transcribe_tamil(audio_data: bytes, audio_format: str = "wav") -> Dict[str, Any]:
    """
    Synchronous — always run in a thread executor (never call from async directly).
    Transcribes Tamil / Tanglish audio with Sarvam saarika:v2.
    """
    try:
        key = _get_api_key()
        mime_map = {
            "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
            "ogg": "audio/ogg", "webm": "audio/webm",
            "mp4": "audio/mp4", "m4a": "audio/mp4",
            "unknown": "audio/wav",
        }
        mime   = mime_map.get(audio_format.lower(), "audio/wav")
        files  = {"file": (f"audio.{audio_format}", io.BytesIO(audio_data), mime)}
        data   = {
            "model":           "saarika:v2",
            "language_code":   "ta-IN",
            "with_timestamps": "false",
        }
        headers = {"api-subscription-key": key}

        resp = requests.post(
            _STT_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=15,
        )
        if not resp.ok:
            logger.error(f"[SARVAM STT] ❌ {resp.status_code}: {resp.text[:300]}")
            return _stt_error(f"HTTP {resp.status_code}")

        transcript = (resp.json().get("transcript", "") or "").strip()
        logger.info(f"[SARVAM STT] ✅ '{transcript[:80]}'")
        return {
            "success":           True,
            "text":              transcript,
            "language":          "ta",
            "is_tamil":          True,
            "confidence":        0.93,
            "provider":          "sarvam",
            "detected_language": "ta",
        }

    except RuntimeError as e:
        logger.warning(f"[SARVAM STT] Key not configured: {e}")
        return _stt_error(str(e), key_missing=True)
    except requests.Timeout:
        logger.error("[SARVAM STT] Request timed out after 15s")
        return _stt_error("Sarvam STT timeout")
    except Exception as exc:
        logger.error(f"[SARVAM STT] Unexpected error: {exc}")
        return _stt_error(str(exc))


def _stt_error(msg: str, key_missing: bool = False) -> Dict[str, Any]:
    return {
        "success":     False,
        "text":        "",
        "language":    "ta",
        "is_tamil":    True,
        "confidence":  0.0,
        "provider":    "sarvam",
        "error":       msg,
        "key_missing": key_missing,
    }


# ── TTS (Text → Speech) ──────────────────────────────────────────────────────

def synthesize_tamil(text: str, speaker: str = "ananya") -> Dict[str, Any]:
    """
    Synchronous — always run in a thread executor (never call from async directly).
    Synthesizes Tanglish text using Sarvam bulbul:v2.

    Speaker options:
      "ananya" — warm, natural Indian female (default)
      "meera"  — slightly softer alternative
    """
    try:
        key     = _get_api_key()
        payload = {
            "inputs":               [text],
            "target_language_code": "ta-IN",
            "speaker":              speaker,
            "model":                "bulbul:v2",
            "speech_sample_rate":   22050,
            "enable_preprocessing": True,
            # Tuned for natural, human-like Tanglish speech:
            # pace 0.9  → slightly slower, sounds more thoughtful/human
            # pitch 0.05 → tiny warmth lift (0.0 is neutral)
            # loudness 1.1 → clearer but not over-loud (1.2 felt robotic)
            "pitch":    0.05,
            "pace":     0.9,
            "loudness": 1.1,
        }
        headers = {
            "api-subscription-key": key,
            "Content-Type":         "application/json",
        }
        resp = _get_session().post(
            _TTS_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20,
        )
        if not resp.ok:
            logger.error(f"[SARVAM TTS] ❌ {resp.status_code}: {resp.text[:300]}")
            return _tts_error(f"HTTP {resp.status_code}")

        audios = resp.json().get("audios", [])
        if not audios:
            return _tts_error("Sarvam TTS returned empty audio list")

        audio_bytes = base64.b64decode(audios[0])
        duration    = max(0.5, (len(text.split()) / 130) * 60)
        logger.info(f"[SARVAM TTS] ✅ {len(audio_bytes)} bytes | '{text[:50]}'")
        return {
            "success":    True,
            "audio_data": audio_bytes,
            "format":     "wav",
            "provider":   "sarvam",
            "duration":   duration,
            "size":       len(audio_bytes),
        }

    except RuntimeError as e:
        logger.warning(f"[SARVAM TTS] Key not configured: {e}")
        return _tts_error(str(e), key_missing=True)
    except requests.Timeout:
        logger.error("[SARVAM TTS] Request timed out after 20s")
        return _tts_error("Sarvam TTS timeout")
    except Exception as exc:
        logger.error(f"[SARVAM TTS] Unexpected error: {exc}")
        return _tts_error(str(exc))


async def synthesize_tamil_stream(text: str, websocket, speaker: str = "ananya") -> None:
    """
    Async wrapper: runs synthesize_tamil() in a thread executor so it never
    blocks the event loop. Streams the resulting WAV in 8KB chunks to the WS.
    WebSocket disconnect is handled gracefully (logs, does not raise).
    """
    try:
        loop   = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, synthesize_tamil, text, speaker),
            timeout=25.0,
        )
    except asyncio.TimeoutError:
        raise RuntimeError("Sarvam TTS timed out (25s)")

    if not result.get("success") or not result.get("audio_data"):
        raise RuntimeError(result.get("error", "Sarvam TTS failed"))

    data       = result["audio_data"]
    chunk_size = 8192
    sent_bytes = 0
    try:
        for i in range(0, len(data), chunk_size):
            await websocket.send_bytes(data[i:i + chunk_size])
            sent_bytes += min(chunk_size, len(data) - i)
        logger.info(f"[SARVAM TTS] ✅ Streamed {sent_bytes} bytes")
    except Exception as ws_err:
        # Client likely disconnected — log and return cleanly, don't raise
        logger.warning(f"[SARVAM TTS] WS send interrupted (client disconnect?): {ws_err}")


def _tts_error(msg: str, key_missing: bool = False) -> Dict[str, Any]:
    return {
        "success":     False,
        "audio_data":  b"",
        "format":      "wav",
        "provider":    "sarvam",
        "error":       msg,
        "key_missing": key_missing,
    }


def is_configured() -> bool:
    """Return True if SARVAM_API_KEY is set."""
    return bool(os.getenv("SARVAM_API_KEY", "").strip())
