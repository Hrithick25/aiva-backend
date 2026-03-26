"""
Text-to-Speech module.

Routing:
  • Tamil mode (language="ta") → Sarvam bulbul:v2 (natural Tanglish TTS)
      Falls back to Edge TTS (ta-IN-PallaviNeural) if SARVAM_API_KEY not set.
  • English (language="en")    → Edge TTS en-US-JennyNeural (warmer, human-like)
  • Hindi (language="hi")      → Edge TTS hi-IN-AnanyaNeural (warm Indian voice)
"""
import asyncio
import logging
import re
from typing import Optional, Dict, Any, List

import edge_tts
from audio.sarvam import synthesize_tamil, synthesize_tamil_stream, is_configured as sarvam_configured

logger = logging.getLogger(__name__)


class TTSProcessor:
    def __init__(self):
        """Initialize TTS with warm, human-like voices for each language."""
        # Edge TTS voice mapping
        # en-US-JennyNeural: warm, conversational assistant tone — most human-sounding
        # hi-IN-AnanyaNeural: natural warm Indian female voice
        # ta-IN-PallaviNeural: Tamil fallback (Sarvam is primary for Tamil)
        self.voice_mapping = {
            "en": "en-US-JennyNeural",
            "ta": "ta-IN-PallaviNeural",
            "hi": "hi-IN-AnanyaNeural",
        }
        # Prosody settings — slightly slower rate sounds more natural/thoughtful
        self.prosody = {
            "en": {"rate": "-5%",  "pitch": "+2Hz"},  # warm, calm pace
            "hi": {"rate": "-5%",  "pitch": "+0Hz"},  # natural Hindi pace
            "ta": {"rate": "-8%",  "pitch": "+0Hz"},  # Tamil fallback (slower)
        }

    async def synthesize_speech(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None,
        emotion: str = "none",
    ) -> Dict[str, Any]:
        """
        Convert text to speech.
        Tamil mode → Sarvam bulbul:v2 (with Edge TTS fallback).
        English/Hindi → Edge TTS.
        """
        resolved_lang = (language or "en").strip().lower()

        # ── Tamil: prefer Sarvam ──────────────────────────────────────
        if resolved_lang == "ta" and sarvam_configured():
            logger.info("[TTS] Tamil mode → Sarvam bulbul:v2")
            result = synthesize_tamil(text)
            if result["success"]:
                return result
            logger.warning("[TTS] Sarvam TTS failed, falling back to Edge TTS")

        # ── Edge TTS (English / Hindi / Tamil fallback) ───────────────
        selected_voice = voice or self.voice_mapping.get(
            resolved_lang, self.voice_mapping["en"]
        )

        last_error = None
        for attempt in range(3):
            try:
                prosody = self.prosody.get(resolved_lang, self.prosody["en"])
                logger.info(
                    f"[TTS] Edge TTS attempt {attempt+1}: voice={selected_voice} "
                    f"rate={prosody['rate']} pitch={prosody['pitch']}"
                )
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=selected_voice,
                    rate=prosody["rate"],
                    pitch=prosody["pitch"],
                )
                audio_bytes = b""

                async def _collect():
                    nonlocal audio_bytes
                    async for chunk in communicate.stream():
                        if chunk.get("type") == "audio" and chunk.get("data"):
                            audio_bytes += chunk["data"]

                await asyncio.wait_for(_collect(), timeout=30)

                if not audio_bytes:
                    raise Exception("Edge TTS returned empty audio")

                return {
                    "success":    True,
                    "audio_data": audio_bytes,
                    "format":     "mp3",
                    "voice":      selected_voice,
                    "language":   resolved_lang,
                    "duration":   self._estimate_duration(text),
                    "size":       len(audio_bytes),
                    "provider":   "edge_tts",
                }
            except asyncio.TimeoutError:
                last_error = "Edge TTS timed out after 30s"
                logger.warning(f"[TTS] attempt {attempt+1} timed out, retrying...")
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[TTS] attempt {attempt+1} failed: {last_error}")
                await asyncio.sleep(0.5 * (attempt + 1))

        logger.error(f"[TTS] Edge TTS failed after 3 attempts: {last_error}")
        return {
            "success":    False,
            "error":      last_error,
            "audio_data": b"",
            "format":     "mp3",
            "provider":   "edge_tts",
        }


    # Edge TTS silently truncates text above ~1 000 chars.
    # This limit keeps each chunk safely within that boundary.
    _EDGE_TTS_MAX_CHARS = 800

    def _split_for_edge_tts(self, text: str) -> List[str]:
        """
        Split text into segments of at most _EDGE_TTS_MAX_CHARS characters,
        always breaking on sentence / clause boundaries so speech sounds natural.
        """
        text = text.strip()
        if not text:
            return []

        # If the whole text fits, no splitting needed
        if len(text) <= self._EDGE_TTS_MAX_CHARS:
            return [text]

        # Sentence-level split (period / exclamation / question)
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)

        segments: List[str] = []
        current = ""

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Single sentence already exceeds limit — split on commas/semicolons
            if len(sentence) > self._EDGE_TTS_MAX_CHARS:
                # flush whatever we had
                if current:
                    segments.append(current.strip())
                    current = ""
                sub_parts = re.split(r'(?<=[,;])\s+', sentence)
                sub_buf = ""
                for part in sub_parts:
                    if len(sub_buf) + len(part) + 1 <= self._EDGE_TTS_MAX_CHARS:
                        sub_buf = (sub_buf + " " + part).strip() if sub_buf else part
                    else:
                        if sub_buf:
                            segments.append(sub_buf.strip())
                        # Still too long? hard-cut by words
                        if len(part) > self._EDGE_TTS_MAX_CHARS:
                            words = part.split()
                            word_buf = ""
                            for w in words:
                                if len(word_buf) + len(w) + 1 <= self._EDGE_TTS_MAX_CHARS:
                                    word_buf = (word_buf + " " + w).strip() if word_buf else w
                                else:
                                    if word_buf:
                                        segments.append(word_buf)
                                    word_buf = w
                            if word_buf:
                                sub_buf = word_buf
                            else:
                                sub_buf = ""
                        else:
                            sub_buf = part
                if sub_buf:
                    current = sub_buf
                continue

            # Normal sentence: does it fit in the current segment?
            prospective = (current + " " + sentence).strip() if current else sentence
            if len(prospective) <= self._EDGE_TTS_MAX_CHARS:
                current = prospective
            else:
                if current:
                    segments.append(current.strip())
                current = sentence

        if current.strip():
            segments.append(current.strip())

        return [s for s in segments if s]

    async def stream_edge_tts(self, text: str, language: str, websocket):
        """Stream TTS audio to a WebSocket.

        Tamil mode → Sarvam bulbul:v2 chunked WAV (with Edge TTS fallback).
        English/Hindi → Edge TTS MP3 stream.
        """
        resolved_lang = (language or "en").strip().lower()

        # ── Tamil: prefer Sarvam ──────────────────────────────────────
        if resolved_lang == "ta" and sarvam_configured():
            logger.info("[TTS stream] Tamil → Sarvam bulbul:v2")
            try:
                await synthesize_tamil_stream(text, websocket)
                return
            except Exception as e:
                logger.warning(f"[TTS stream] Sarvam failed: {e}, falling back to Edge TTS")

        # ── Edge TTS (English / Hindi / Tamil fallback) ───────────────
        selected_voice = self.voice_mapping.get(resolved_lang, self.voice_mapping["en"])
        prosody        = self.prosody.get(resolved_lang, self.prosody["en"])
        segments       = self._split_for_edge_tts(text)
        if not segments:
            logger.warning("[TTS stream] Empty text — skipping")
            return

        logger.info(
            "[TTS stream] %d segment(s) via Edge TTS | voice=%s rate=%s",
            len(segments), selected_voice, prosody["rate"],
        )
        total_bytes_all = 0

        for seg_idx, segment in enumerate(segments):
            last_error = None
            seg_ok = False

            for attempt in range(3):
                try:
                    communicate = edge_tts.Communicate(
                        text=segment,
                        voice=selected_voice,
                        rate=prosody["rate"],
                        pitch=prosody["pitch"],
                    )
                    seg_bytes = 0

                    async def _stream_segment():
                        nonlocal seg_bytes, total_bytes_all
                        async for chunk in communicate.stream():
                            if chunk.get("type") == "audio" and chunk.get("data"):
                                await websocket.send_bytes(chunk["data"])
                                seg_bytes       += len(chunk["data"])
                                total_bytes_all += len(chunk["data"])

                    await asyncio.wait_for(_stream_segment(), timeout=30)

                    if seg_bytes == 0:
                        raise Exception("Edge TTS returned 0 bytes for segment")

                    logger.info(
                        "[TTS] Segment %d/%d done — %d bytes | '%s…'",
                        seg_idx + 1, len(segments), seg_bytes, segment[:40],
                    )
                    seg_ok = True
                    break   # segment succeeded

                except asyncio.TimeoutError:
                    last_error = f"Segment {seg_idx+1} timed out (attempt {attempt+1})"
                    logger.warning("[TTS] %s — retrying…", last_error)
                    await asyncio.sleep(0.5 * (attempt + 1))
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning("[TTS] Segment %d attempt %d failed: %s — retrying…",
                                   seg_idx + 1, attempt + 1, last_error)
                    await asyncio.sleep(0.5 * (attempt + 1))

            if not seg_ok:
                # Log and skip the failed segment rather than aborting the whole stream
                logger.error(
                    "[TTS] Segment %d/%d permanently failed after 3 attempts: %s — skipping.",
                    seg_idx + 1, len(segments), last_error,
                )

        if total_bytes_all == 0:
            raise Exception("Edge TTS stream produced 0 bytes for all segments")

        logger.info("[TTS] ✅ Full stream complete — %d bytes total", total_bytes_all)

    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration based on text length."""
        words = len(text.split())
        return max(0.5, (words / 150) * 60)

    def get_available_voices(self, language: str = "en") -> Dict[str, Any]:
        """Get list of available voices for a language."""
        voices_info = {
            "en": [
                {"name": "en-US-JennyNeural",  "gender": "Female",
                 "description": "US English, warm conversational — most human-like"},
            ],
            "ta": [
                {"name": "Sarvam bulbul:v2 (ananya)", "gender": "Female",
                 "description": "Tamil/Tanglish, natural Indian accent"},
                {"name": "ta-IN-PallaviNeural", "gender": "Female",
                 "description": "Tamil Edge TTS fallback"},
            ],
            "hi": [
                {"name": "hi-IN-AnanyaNeural", "gender": "Female",
                 "description": "Hindi, warm natural Indian voice"},
            ],
        }
        return {
            "provider": "edge_tts",
            "language": language,
            "voices":   voices_info.get(language, voices_info["en"]),
        }

    def validate_text_input(self, text: str) -> Dict[str, Any]:
        """Validate text input for TTS."""
        try:
            if not text or not text.strip():
                return {"valid": False, "error": "Empty text provided"}

            if len(text) > 5000:
                return {
                    "valid": False,
                    "error": f"Text too long: {len(text)} characters (max: 5000)",
                }

            return {
                "valid": True,
                "length": len(text),
                "word_count": len(text.split()),
                "estimated_duration": self._estimate_duration(text),
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into complete sentences for streaming TTS."""
        if not text or not text.strip():
            return []

        # Handle common abbreviations to avoid false splits
        text = text.replace("Mr.", "Mr").replace("Dr.", "Dr").replace("Ms.", "Ms")
        text = text.replace("A.M.", "AM").replace("P.M.", "PM")
        text = text.replace("a.m.", "am").replace("p.m.", "pm")

        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:
                if sentence[-1] not in '.!?':
                    sentence += '.'
                clean_sentences.append(sentence)

        return clean_sentences

    async def synthesize_sentences_streaming(
        self,
        text: str,
        language: str = "en",
        voice: str = None,
        emotion: str = "none",
    ) -> List[Dict[str, Any]]:
        """Synthesize text as streaming sentence chunks."""
        try:
            sentences = self.split_into_sentences(text)

            if not sentences:
                return [
                    {
                        "success": False,
                        "error": "No valid sentences to synthesize",
                        "chunk_id": 0,
                        "text_chunk": "",
                        "audio_data": b"",
                    }
                ]

            logger.info(
                f"TTS streaming {len(sentences)} sentences: {[s[:30]+'...' for s in sentences]}"
            )

            results = []

            for i, sentence in enumerate(sentences):
                try:
                    result = await self.synthesize_speech(sentence, language, voice, emotion)

                    if result["success"]:
                        results.append(
                            {
                                "success": True,
                                "chunk_id": i,
                                "total_chunks": len(sentences),
                                "text_chunk": sentence,
                                "audio_data": result["audio_data"],
                                "format": result["format"],
                                "size": result["size"],
                                "duration": result.get("duration", 0.0),
                                "is_final": (i == len(sentences) - 1),
                            }
                        )
                        logger.info(
                            f"TTS chunk {i+1}/{len(sentences)} complete: '{sentence[:40]}...'"
                        )
                    else:
                        logger.error(
                            f"TTS failed for sentence {i+1}: {result.get('error')}"
                        )
                        results.append(
                            {
                                "success": False,
                                "chunk_id": i,
                                "total_chunks": len(sentences),
                                "text_chunk": sentence,
                                "error": result.get("error", "TTS failed"),
                                "audio_data": b"",
                                "is_final": (i == len(sentences) - 1),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error processing sentence {i+1}: {e}")
                    results.append(
                        {
                            "success": False,
                            "chunk_id": i,
                            "total_chunks": len(sentences),
                            "text_chunk": sentence,
                            "error": str(e),
                            "audio_data": b"",
                            "is_final": (i == len(sentences) - 1),
                        }
                    )

            logger.info(f"TTS streaming complete: {len(results)} chunks processed")
            return results

        except Exception as error:
            logger.error(f"TTS streaming error: {error}")
            return [
                {
                    "success": False,
                    "error": str(error),
                    "chunk_id": 0,
                    "text_chunk": text[:50] + "..." if len(text) > 50 else text,
                    "audio_data": b"",
                }
            ]


# Global TTS processor instance
_tts_processor = None


def get_tts_processor() -> TTSProcessor:
    """Get global TTS processor instance"""
    global _tts_processor
    if _tts_processor is None:
        _tts_processor = TTSProcessor()
    return _tts_processor