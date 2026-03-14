"""
Audio processing manager that coordinates STT and TTS operations
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, Tuple
from .stt import get_stt_processor
from .tts import get_tts_processor

logger = logging.getLogger(__name__)

class AudioManager:
    def __init__(self):
        """Initialize audio manager with STT and TTS processors"""
        self.stt_processor = get_stt_processor()
        self.tts_processor = get_tts_processor()

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language from text by checking for Tamil Unicode characters.
        Returns 'ta' if Tamil script found, else 'en'."""
        for char in text:
            if '\u0B80' <= char <= '\u0BFF':
                return "ta"
        return "en"
        
    async def process_audio_to_text(
        self, 
        audio_data: bytes, 
        language: str = "auto"
    ) -> Dict[str, Any]:
        """
        Process audio input to text
        
        Args:
            audio_data: Raw audio bytes
            language: Expected language ("en", "ta")
            
        Returns:
            Transcription result with confidence and language detection
        """
        try:
            # Validate audio format first
            validation = await self.stt_processor.validate_audio_format(audio_data)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Invalid audio format: {validation.get('error', 'Unknown')}",
                    "text": "",
                    "confidence": 0.0
                }
            
            logger.info(f"Processing audio: {validation['format']}, {validation['size']} bytes")
            
            # Transcribe audio
            result = await self.stt_processor.transcribe_audio(audio_data, language)
            
            if result["success"]:
                logger.info(f"Transcription successful: '{result['text'][:50]}...'")
            else:
                logger.error(f"Transcription failed: {result.get('error', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0
            }

    async def process_text_to_audio(
        self, 
        text: str, 
        language: str = "en",
        emotion: str = "none",
        voice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process text to audio
        
        Args:
            text: Text to synthesize
            language: Target language ("en", "ta")
            emotion: Emotion for speech ("happy", "sad", "none") 
            voice: Specific voice name (optional)
            
        Returns:
            Audio synthesis result with metadata
        """
        try:
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "Empty text provided",
                    "audio_data": b"",
                    "format": "wav"
                }
            
            logger.info(f"Synthesizing text: '{text[:50]}...' (lang: {language}, emotion: {emotion})")
            
            # Synthesize speech
            result = await self.tts_processor.synthesize_speech(text, language, voice, emotion)
            
            if result["success"]:
                logger.info(f"TTS successful: {result['size']} bytes, {result['format']} format")
            else:
                logger.error(f"TTS failed: {result.get('error', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            logger.error(f"Text-to-audio processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": b"",
                "format": "wav"
            }

    async def process_audio_conversation(
        self, 
        audio_data: bytes,
        get_response_func,
        input_language: str = "auto",
        output_language: str = "en",
        emotion: str = "none"
    ) -> Dict[str, Any]:
        """
        Complete audio conversation flow: STT -> Agent -> TTS
        
        Args:
            audio_data: Input audio bytes
            get_response_func: Function to get agent response (async)
            input_language: Language for STT
            output_language: Language for TTS
            emotion: Emotion for TTS response
            
        Returns:
            Complete conversation result with both text and audio
        """
        try:
            # Step 1: Convert speech to text
            stt_result = await self.process_audio_to_text(audio_data, input_language)
            
            if not stt_result["success"]:
                return {
                    "success": False,
                    "error": f"STT failed: {stt_result.get('error', 'Unknown')}",
                    "input_text": "",
                    "response_text": "",
                    "audio_data": b"",
                    "stt_confidence": 0.0
                }
            
            input_text = stt_result["text"]
            detected_language = stt_result.get("language", "unknown")
            is_tamil = stt_result.get("is_tamil", False)
            
            # Create language context for agent
            language_context = {
                "language": detected_language,
                "is_tamil": is_tamil,
                "confidence": stt_result["confidence"]
            }
            
            # Step 2: Get agent response with language context
            logger.info(f"Getting agent response for: '{input_text}' (detected: {detected_language})")
            agent_result = await get_response_func(input_text, language_context)
            
            # Validate agent response format
            if not isinstance(agent_result, dict):
                logger.error(f"Invalid agent response type: {type(agent_result)}")
                return {
                    "success": False,
                    "error": "Invalid agent response format",
                    "input_text": input_text,
                    "response_text": "",
                    "audio_data": b"",
                    "stt_confidence": stt_result["confidence"]
                }
            
            # Extract response text with safety checks
            response_text = agent_result.get("response", "")
            response_emotion = agent_result.get("emotion", emotion)
            
            # Safety validation for response text
            if not response_text or not isinstance(response_text, str):
                logger.error(f"Invalid response text from agent: {type(response_text)} - {str(response_text)[:100]}")
                response_text = "I apologize, but I'm having trouble generating a response right now."
            
            # Additional safety check - ensure it's not malformed JSON
            response_text = response_text.strip()
            if response_text.startswith('{') or response_text.startswith('['):
                logger.warning(f"Response text appears to be JSON, extracting content: {response_text[:50]}...")
                # Try to extract readable content from potential JSON fragments
                if '"response"' in response_text:
                    try:
                        import re
                        match = re.search(r'"response":\s*"([^"]*)"', response_text)
                        if match:
                            response_text = match.group(1)
                        else:
                            response_text = "I apologize for the technical issue. Please try again."
                    except:
                        response_text = "I apologize for the technical issue. Please try again."
                else:
                    response_text = "I apologize for the technical issue. Please try again."
            
            logger.info(f"Final response text for TTS: '{response_text[:50]}...'")
            
            # Step 3: Convert response to speech
            # Auto-detect Tamil from response text instead of relying on output_language param
            resolved_output_language = self._detect_language(response_text)
            logger.info(f"TTS language auto-detected from response: {resolved_output_language}")

            tts_result = await self.process_text_to_audio(
                response_text, 
                resolved_output_language,
                response_emotion
            )
            
            return {
                "success": True,
                "input_text": input_text,
                "input_language": detected_language,
                "output_language": resolved_output_language,
                "response_text": response_text,
                "response_emotion": response_emotion,
                "audio_data": tts_result.get("audio_data", b""),
                "audio_format": tts_result.get("format", "wav"),
                "audio_duration": tts_result.get("duration", 0.0),
                "stt_confidence": stt_result["confidence"],
                "tts_success": tts_result["success"]
            }
            
        except Exception as e:
            logger.error(f"Audio conversation processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "input_text": "",
                "response_text": "",
                "audio_data": b"",
                "stt_confidence": 0.0
            }

    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported audio formats and capabilities"""
        return {
            "stt": {
                "supported_formats": ["wav", "mp3", "ogg", "flac", "m4a"],
                "languages": ["en", "ta"],
                "max_duration": 300,  # seconds
                "provider": "deepgram"
            },
            "tts": {
                "supported_formats": ["wav"],
                "languages": ["en", "ta"],
                "emotions": ["none", "happy", "sad"],
                "provider": "gemini_tts"
            },
            "conversation_flow": {
                "supports_realtime": False,
                "supports_streaming": False,
                "supports_emotion_detection": True,
                "api_key_rotation": True
            }
        }

    async def get_voice_options(self, language: str = "en") -> Dict[str, Any]:
        """Get available voice options for TTS"""
        return self.tts_processor.get_available_voices(language)

# Global audio manager instance
_audio_manager = None

def get_audio_manager() -> AudioManager:
    """Get global audio manager instance"""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioManager()
    return _audio_manager