"""
Text-to-Speech module using Microsoft Edge TTS exclusively.
Supports English, Tamil, and Hindi with streaming audio.
"""
import asyncio
import logging
import re
from typing import Optional, Dict, Any, List

import edge_tts

logger = logging.getLogger(__name__)


class TTSProcessor:
    def __init__(self):
        """Initialize TTS processor with Edge TTS voices for en, ta, hi."""
        # Edge TTS voice mapping by language
        self.voice_mapping = {
            "en": "en-US-AriaNeural",
            "ta": "ta-IN-PallaviNeural",
            "hi": "hi-IN-SwaraNeural",
        }

    async def synthesize_speech(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None,
        emotion: str = "none",
    ) -> Dict[str, Any]:
        """
        Convert text to speech using Edge TTS.

        Args:
            text: Text to synthesize
            language: Language code ("en", "ta", "hi")
            voice: Specific voice name override (optional)
            emotion: (unused, kept for interface compat)

        Returns:
            Dict with audio data and metadata
        """
        try:
            resolved_lang = (language or "en").strip().lower()
            selected_voice = voice or self.voice_mapping.get(
                resolved_lang, self.voice_mapping["en"]
            )

            logger.info(
                f"Synthesizing with Edge TTS: voice={selected_voice}, lang={resolved_lang}"
            )

            communicate = edge_tts.Communicate(text=text, voice=selected_voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    audio_bytes += chunk["data"]

            if not audio_bytes:
                raise Exception("Edge TTS returned empty audio")

            duration = self._estimate_duration(text)

            return {
                "success": True,
                "audio_data": audio_bytes,
                "format": "mp3",
                "voice": selected_voice,
                "language": resolved_lang,
                "duration": duration,
                "size": len(audio_bytes),
                "provider": "edge_tts",
            }
        except Exception as e:
            logger.error(f"Edge TTS synthesis error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": b"",
                "format": "mp3",
                "provider": "edge_tts",
            }

    async def stream_edge_tts(self, text: str, language: str, websocket):
        """Stream Edge TTS audio chunks directly to a WebSocket as binary frames.

        Args:
            text: Text to synthesize
            language: Language code ("en", "ta", "hi")
            websocket: FastAPI WebSocket instance
        """
        resolved_lang = (language or "en").strip().lower()
        selected_voice = self.voice_mapping.get(
            resolved_lang, self.voice_mapping["en"]
        )

        logger.info(
            f"Streaming Edge TTS: voice={selected_voice}, lang={resolved_lang}"
        )

        communicate = edge_tts.Communicate(text=text, voice=selected_voice)
        chunk_count = 0
        total_bytes = 0

        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                await websocket.send_bytes(chunk["data"])
                chunk_count += 1
                total_bytes += len(chunk["data"])

        logger.info(
            f"Edge TTS stream complete: {chunk_count} chunks, {total_bytes} bytes"
        )

    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration based on text length."""
        words = len(text.split())
        return max(0.5, (words / 150) * 60)

    def get_available_voices(self, language: str = "en") -> Dict[str, Any]:
        """Get list of available voices for a language."""
        voices_info = {
            "en": [
                {"name": "en-US-AriaNeural", "gender": "Female", "description": "US English, expressive"}
            ],
            "ta": [
                {"name": "ta-IN-PallaviNeural", "gender": "Female", "description": "Tamil, natural"}
            ],
            "hi": [
                {"name": "hi-IN-SwaraNeural", "gender": "Female", "description": "Hindi, natural"}
            ],
        }

        return {
            "provider": "edge_tts",
            "language": language,
            "voices": voices_info.get(language, voices_info["en"]),
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