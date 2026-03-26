"""
websocket_handler.py — AIVA WebSocket gateway.

Crash-proofing guarantees:
  • _safe_send_json() swallows disconnects on every outgoing JSON message.
  • handle_audio_base64_streaming() has a 30-second hard asyncio timeout.
  • TTS errors fall back to single-shot synthesis; if that also fails, the
    text response (already sent) is sufficient and we simply stop.
  • All entry points have a top-level try/except so no unhandled exception
    ever propagates to the main WebSocket loop.
"""

import json
import re
import traceback
import base64
import time
import asyncio
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from agent.groq_llama_agent import get_agent_response
from audio.manager import get_audio_manager

router = APIRouter()
audio_manager = get_audio_manager()

_ws_sessions: dict = {}


# ── Safe send helpers ─────────────────────────────────────────────────────────

async def _safe_send_json(ws: WebSocket, data: dict) -> bool:
    """Sends JSON without raising if the client has already disconnected."""
    try:
        await ws.send_json(data)
        return True
    except Exception:
        return False


# ── Session / agent helpers ───────────────────────────────────────────────────

async def call_agent_with_history(
    ws: WebSocket,
    query: str,
    language_context: Optional[dict] = None,
) -> dict:
    session_id = id(ws)
    if session_id not in _ws_sessions:
        _ws_sessions[session_id] = {"chat_history": [], "last_query": ""}

    session_data = _ws_sessions[session_id]

    rag_query = query
    if len(query.split()) <= 4 and session_data["last_query"]:
        rag_query = f"{session_data['last_query']} {query}"
    elif len(query.split()) > 4:
        session_data["last_query"] = query

    result = await get_agent_response(
        query, language_context, session_data["chat_history"], rag_query
    )

    if result and "response" in result:
        session_data["chat_history"].append({"role": "user",      "content": query})
        session_data["chat_history"].append({"role": "assistant", "content": result["response"]})
        if len(session_data["chat_history"]) > 6:
            session_data["chat_history"] = session_data["chat_history"][-6:]

    return result


# ── Response cache ────────────────────────────────────────────────────────────

_response_cache: dict = {}
_CACHE_MAX_SIZE = 256


def _get_cached_response(query: str) -> Optional[dict]:
    return _response_cache.get(query.strip().lower())


def _cache_response(query: str, result: dict):
    if len(_response_cache) >= _CACHE_MAX_SIZE:
        del _response_cache[next(iter(_response_cache))]
    _response_cache[query.strip().lower()] = result


# ── Language helpers ──────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Unicode-only detection (works for Tamil script and Hindi; not Tanglish)."""
    for char in text:
        if '\u0B80' <= char <= '\u0BFF':
            return "ta"
        if '\u0900' <= char <= '\u097F':
            return "hi"
    return "en"


def _normalise_lang(raw: Optional[str]) -> str:
    """Map any language hint to en/ta/hi; unknown → 'auto'."""
    if not raw:
        return "auto"
    raw = raw.strip().lower()
    mapping = {
        "en": "en", "en-us": "en", "en-in": "en", "english": "en",
        "ta": "ta", "ta-in": "ta", "tamil": "ta",
        "hi": "hi", "hi-in": "hi", "hindi": "hi",
    }
    return mapping.get(raw, "auto")


# ── WebSocket entry point ─────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = id(ws)
    _ws_sessions[session_id] = {"chat_history": [], "last_query": ""}
    print("[WS] Client connected")

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.receive":
                if "text" in message:
                    await handle_text_message(ws, message["text"])
                elif "bytes" in message:
                    await handle_binary_message(ws, message["bytes"])
            elif message["type"] == "websocket.disconnect":
                print("[WS] Client disconnected (clean)")
                break

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        traceback.print_exc()
        await _safe_send_json(ws, {
            "type": "error",
            "response": f"Server error: {str(e)}",
            "emotion": "sad",
        })
    finally:
        _ws_sessions.pop(session_id, None)


# ── Message routers ───────────────────────────────────────────────────────────

async def handle_text_message(ws: WebSocket, data: str):
    try:
        payload = json.loads(data)
        await handle_json_message(ws, payload)
    except json.JSONDecodeError:
        query = data.strip()
        if not query:
            await _safe_send_json(ws, {"type": "text_response", "response": "Please send a valid query.", "emotion": "none"})
            return
        result = await call_agent_with_history(ws, query)
        result["type"] = "text_response"
        await _safe_send_json(ws, result)
    except Exception as e:
        await _safe_send_json(ws, {"type": "error", "response": str(e), "emotion": "sad"})


async def handle_json_message(ws: WebSocket, payload: dict):
    message_type = payload.get("type", "text")
    handlers = {
        "text":                 handle_text_query,
        "audio":                handle_audio_query,
        "audio_base64":         handle_audio_base64_query,
        "get_audio_info":       lambda w, p: handle_audio_info_request(w),
        "get_voices":           handle_voices_request,
        "audio_base64_streaming": handle_audio_base64_streaming,
        "audio_streaming":      handle_audio_streaming,
        "audio_tts_streaming":  handle_tts_streaming,
        "test_immediate":       handle_test_immediate,
    }
    handler = handlers.get(message_type)
    if handler:
        await handler(ws, payload)
    else:
        await _safe_send_json(ws, {"type": "error", "response": f"Unknown type: {message_type}", "emotion": "sad"})


# ── Text query ────────────────────────────────────────────────────────────────

async def handle_text_query(ws: WebSocket, payload: dict):
    query = payload.get("query", "").strip()
    if not query:
        await _safe_send_json(ws, {"type": "text_response", "response": "Please send a valid query.", "emotion": "none"})
        return

    t0         = time.time()
    enable_tts = payload.get("enable_tts", False)
    raw_lang   = payload.get("language")
    query_lang = _normalise_lang(raw_lang)
    if query_lang == "auto":
        query_lang = detect_language(query)

    lang_ctx = {"language": query_lang, "is_tamil": query_lang == "ta", "confidence": 1.0}

    cached = _get_cached_response(query)
    if cached:
        result = cached.copy()
    else:
        result = await call_agent_with_history(ws, query, lang_ctx)
        _cache_response(query, result)

    # Use is_tamil from context — Tanglish is all-Latin
    is_tamil_resp = query_lang == "ta"

    if enable_tts and result.get("response"):
        tts_lang = "ta" if is_tamil_resp else detect_language(result["response"])
        try:
            tts_result = await asyncio.wait_for(
                audio_manager.process_text_to_audio(result["response"], tts_lang, result.get("emotion", "none")),
                timeout=15.0,
            )
            if tts_result["success"]:
                result.update({
                    "type":           "text_with_audio_response",
                    "audio_data":     base64.b64encode(tts_result["audio_data"]).decode("utf-8"),
                    "audio_format":   tts_result["format"],
                    "audio_duration": tts_result.get("duration", 0.0),
                })
            else:
                result["type"] = "text_response"
        except asyncio.TimeoutError:
            result["type"] = "text_response"
    else:
        result["type"] = "text_response"

    print(f"[WS] Text pipeline {(time.time()-t0)*1000:.0f}ms")
    await _safe_send_json(ws, result)


# ── Binary audio ──────────────────────────────────────────────────────────────

async def handle_binary_message(ws: WebSocket, binary_data: bytes):
    print(f"[WS] Binary audio {len(binary_data)}B")
    try:
        result = await audio_manager.process_audio_conversation(
            binary_data,
            lambda q, lc=None: call_agent_with_history(ws, q, lc),
            input_language="en",
            output_language="en",
        )
        if result["success"]:
            payload = {
                "type":           "audio_conversation_response",
                "success":        True,
                "input_text":     result["input_text"],
                "response_text":  result["response_text"],
                "emotion":        result["response_emotion"],
                "audio_data":     base64.b64encode(result["audio_data"]).decode("utf-8"),
                "audio_format":   result["audio_format"],
                "audio_duration": result["audio_duration"],
                "stt_confidence": result["stt_confidence"],
            }
        else:
            payload = {
                "type":           "audio_conversation_response",
                "success":        False,
                "error":          result.get("error", "Unknown error"),
                "input_text":     result.get("input_text", ""),
                "response_text":  "",
                "audio_data":     "",
                "stt_confidence": result.get("stt_confidence", 0.0),
            }
        await _safe_send_json(ws, payload)
    except Exception as e:
        await _safe_send_json(ws, {"type": "error", "response": str(e), "emotion": "sad"})


# ── Base64 audio (non-streaming) ──────────────────────────────────────────────

async def handle_audio_base64_query(ws: WebSocket, payload: dict):
    try:
        raw = payload.get("audio_data", "")
        if not raw:
            await _safe_send_json(ws, {"type": "error", "response": "No audio data provided", "emotion": "sad"})
            return

        audio_data = base64.b64decode(raw)
        in_lang    = payload.get("input_language",  "en")
        out_lang   = payload.get("output_language", "en")

        result = await audio_manager.process_audio_conversation(
            audio_data,
            lambda q, lc=None: call_agent_with_history(ws, q, lc),
            input_language=in_lang,
            output_language=out_lang,
        )

        if result["success"]:
            resp = {
                "type":           "audio_conversation_response",
                "success":        True,
                "input_text":     result["input_text"],
                "input_language": result.get("input_language", in_lang),
                "response_text":  result["response_text"],
                "emotion":        result["response_emotion"],
                "audio_data":     base64.b64encode(result["audio_data"]).decode("utf-8"),
                "audio_format":   result["audio_format"],
                "audio_duration": result["audio_duration"],
                "stt_confidence": result["stt_confidence"],
            }
        else:
            resp = {
                "type":           "audio_conversation_response",
                "success":        False,
                "error":          result.get("error", "Processing failed"),
                "input_text":     result.get("input_text", ""),
                "response_text":  "",
                "audio_data":     "",
                "stt_confidence": result.get("stt_confidence", 0.0),
            }
        await _safe_send_json(ws, resp)

    except Exception as e:
        await _safe_send_json(ws, {"type": "error", "response": str(e), "emotion": "sad"})


# ── Streaming audio (main path) ───────────────────────────────────────────────

async def handle_audio_streaming(ws: WebSocket, payload: dict):
    try:
        raw = payload.get("audio_data", b"")
        if not raw:
            await _safe_send_json(ws, {"type": "error", "response": "No audio data", "emotion": "sad"})
            return
        await handle_audio_base64_streaming(ws, {
            "audio_data":     base64.b64encode(raw).decode("utf-8"),
            "input_language": payload.get("input_language", "auto"),
            "output_language": payload.get("output_language", "en"),
            "language":       payload.get("language"),
        })
    except Exception as e:
        await _safe_send_json(ws, {"type": "streaming_error", "error": str(e), "stage": "general"})


async def handle_audio_base64_streaming(ws: WebSocket, payload: dict):
    """
    Main voice pipeline with 30-second hard timeout.
    Flow: STT → Agent (cached) → send text immediately → stream TTS.
    Every send is crash-safe. Client disconnect is handled silently.
    Tamil mode always uses Sarvam STT + TTS and enforces Tanglish responses.
    """
    try:
        await asyncio.wait_for(
            _audio_pipeline(ws, payload),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        print("[WS] ⏰ 30s timeout — pipeline aborted")
        await _safe_send_json(ws, {
            "type":  "streaming_error",
            "error": "Request timed out. Please try again.",
            "stage": "timeout",
        })
    except (WebSocketDisconnect, RuntimeError):
        print("[WS] Client disconnected during pipeline")
    except Exception as e:
        print(f"[WS] Pipeline error: {e}")
        await _safe_send_json(ws, {
            "type":  "streaming_error",
            "error": "An unexpected error occurred. Please try again.",
            "stage": "general",
        })


async def _audio_pipeline(ws: WebSocket, payload: dict):
    """Inner pipeline — run inside a 30-second asyncio.wait_for."""

    # ── Decode audio ──────────────────────────────────────────────────────
    raw_b64 = payload.get("audio_data", "")
    if not raw_b64:
        await _safe_send_json(ws, {"type": "error", "response": "No audio data provided", "emotion": "sad"})
        return

    try:
        audio_data = base64.b64decode(raw_b64)
    except Exception:
        await _safe_send_json(ws, {"type": "streaming_error", "error": "Invalid audio encoding", "stage": "decode"})
        return

    # Validate minimum size
    if len(audio_data) < 1000:
        await _safe_send_json(ws, {"type": "streaming_error", "error": "Audio clip too short", "stage": "decode"})
        return

    raw_lang  = payload.get("language") or payload.get("input_language", "auto")
    user_lang = _normalise_lang(raw_lang)

    print(f"[WS] 🎤 Audio {len(audio_data)}B lang={user_lang}")
    mgr = get_audio_manager()
    t_pipeline = time.perf_counter()

    # ── STT ───────────────────────────────────────────────────────────────
    t_stt      = time.perf_counter()
    stt_result = await mgr.process_audio_to_text(audio_data, user_lang)
    stt_ms     = (time.perf_counter() - t_stt) * 1000
    print(f"[WS] STT {stt_ms:.0f}ms | provider={stt_result.get('provider','?')}")

    if not stt_result.get("success") or not stt_result.get("text", "").strip():
        err = stt_result.get("error") or "No speech detected — please speak clearly and try again."
        await _safe_send_json(ws, {"type": "streaming_error", "error": err, "stage": "stt"})
        return

    input_text   = stt_result["text"].strip()
    lang_code    = stt_result.get("language", "en")
    is_tamil     = bool(stt_result.get("is_tamil", False))
    # If user explicitly selected Tamil mode, always enforce it
    if user_lang == "ta":
        is_tamil  = True
        lang_code = "ta"

    lang_ctx = {"language": lang_code, "is_tamil": is_tamil, "confidence": stt_result.get("confidence", 0.9)}
    print(f"[WS] STT: '{input_text[:60]}' lang={lang_code} tamil={is_tamil}")

    # ── Immediate ACK ─────────────────────────────────────────────────────
    await _safe_send_json(ws, {
        "type":       "streaming_status",
        "stage":      "processing",
        "input_text": input_text,
        "message":    "Processing your query...",
    })

    # ── Agent (LRU cache) ─────────────────────────────────────────────────
    cached = _get_cached_response(input_text)
    if cached:
        agent_result = cached.copy()
        print("[WS] ⚡ Cache HIT")
    else:
        t_agent      = time.perf_counter()
        agent_result = await call_agent_with_history(ws, input_text, lang_ctx)
        agent_ms     = (time.perf_counter() - t_agent) * 1000
        _cache_response(input_text, agent_result)
        print(f"[WS] Agent {agent_ms:.0f}ms")

    if not isinstance(agent_result, dict) or not agent_result.get("response"):
        await _safe_send_json(ws, {
            "type":  "streaming_error",
            "error": "Agent could not generate a response. Please try again.",
            "stage": "agent",
        })
        return

    response_text    = str(agent_result.get("response", "")).strip()
    response_emotion = agent_result.get("emotion", "none")
    if response_emotion not in ("happy", "sad", "none"):
        response_emotion = "none"

    # Tanglish is all-Latin — Unicode scan returns 'en'; use is_tamil flag
    tts_lang = "ta" if is_tamil else detect_language(response_text)
    print(f"[WS] Resp: '{response_text[:60]}' tts_lang={tts_lang}")

    # ── Send text IMMEDIATELY ─────────────────────────────────────────────
    await _safe_send_json(ws, {
        "type":             "streaming_text_response",
        "success":          True,
        "input_text":       input_text,
        "input_language":   lang_code,
        "response_text":    response_text,
        "emotion":          response_emotion,
        "stt_confidence":   stt_result.get("confidence", 0.9),
        "is_tamil":         is_tamil,
        "audio_processing": "in_progress",
    })
    print("[WS] ✅ Text dispatched")

    # ── TTS stream ────────────────────────────────────────────────────────
    t_tts = time.perf_counter()
    try:
        await _safe_send_json(ws, {
            "type":     "audio_stream_start",
            "format":   "wav" if tts_lang == "ta" else "mp3",
            "language": tts_lang,
        })
        await mgr.stream_tts_to_websocket(
            text=response_text,
            language=tts_lang,
            websocket=ws,
        )
        tts_ms    = (time.perf_counter() - t_tts) * 1000
        total_ms  = (time.perf_counter() - t_pipeline) * 1000
        await _safe_send_json(ws, {
            "type":              "audio_stream_end",
            "audio_processing":  "complete",
            "duration_ms":       tts_ms,
            "total_pipeline_ms": total_ms,
        })
        print(f"[WS] ✅ Pipeline {total_ms:.0f}ms (STT {stt_ms:.0f}ms | TTS {tts_ms:.0f}ms)")

    except (WebSocketDisconnect, RuntimeError):
        print("[WS] Client disconnected during TTS stream")

    except Exception as tts_err:
        # TTS stream failed — attempt single-shot fallback
        print(f"[WS] TTS stream failed ({tts_err}), single-shot fallback")
        try:
            fb = await asyncio.wait_for(
                mgr.process_text_to_audio(response_text, tts_lang, response_emotion),
                timeout=12.0,
            )
            if fb.get("success"):
                await _safe_send_json(ws, {
                    "type":             "streaming_audio_chunk",
                    "chunk_id":         0,
                    "total_chunks":     1,
                    "audio_data":       base64.b64encode(fb["audio_data"]).decode("utf-8"),
                    "audio_format":     fb.get("format", "mp3"),
                    "is_final":         True,
                    "audio_processing": "complete",
                })
            else:
                await _safe_send_json(ws, {
                    "type":             "streaming_audio_error",
                    "error":            "TTS unavailable",
                    "audio_processing": "failed",
                })
        except asyncio.TimeoutError:
            await _safe_send_json(ws, {
                "type":             "streaming_audio_error",
                "error":            "TTS timed out",
                "audio_processing": "failed",
            })
        except Exception:
            pass  # Text already shown — audio is best-effort


# ── TTS-only streaming ────────────────────────────────────────────────────────

async def handle_tts_streaming(ws: WebSocket, payload: dict):
    try:
        text = payload.get("text", "").strip()
        if not text:
            await _safe_send_json(ws, {"type": "error", "response": "No text provided for TTS", "emotion": "sad"})
            return

        language  = _normalise_lang(payload.get("language", "en"))
        if language == "auto":
            language = "en"
        emotion   = payload.get("emotion", "none")
        sentences = split_text_into_sentences(text)
        mgr       = get_audio_manager()

        for i, sentence in enumerate(sentences):
            try:
                result = await mgr.process_text_to_audio(sentence, language, emotion)
                if result["success"]:
                    await _safe_send_json(ws, {
                        "type":           "streaming_tts_chunk",
                        "chunk_id":       i,
                        "total_chunks":   len(sentences),
                        "text_chunk":     sentence,
                        "audio_data":     base64.b64encode(result["audio_data"]).decode("utf-8"),
                        "audio_format":   result.get("format", "mp3"),
                        "is_final":       (i == len(sentences) - 1),
                        "chunk_duration": result.get("duration", 0.0),
                    })
                else:
                    await _safe_send_json(ws, {
                        "type": "streaming_tts_error",
                        "chunk_id": i,
                        "error": result.get("error", "TTS failed"),
                        "text_chunk": sentence,
                    })
            except Exception as e:
                await _safe_send_json(ws, {
                    "type": "streaming_tts_error",
                    "chunk_id": i,
                    "error": str(e),
                    "text_chunk": sentence,
                })

        await _safe_send_json(ws, {
            "type":                  "streaming_tts_complete",
            "total_chunks_processed": len(sentences),
        })

    except Exception as e:
        await _safe_send_json(ws, {"type": "streaming_error", "error": str(e), "stage": "tts_streaming"})


def split_text_into_sentences(text: str):
    text = text.replace("Mr.", "Mr").replace("Dr.", "Dr").replace("etc.", "etc")
    text = text.replace("A.M.", "AM").replace("P.M.", "PM")
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        s = s.replace("Mr", "Mr.").replace("Dr", "Dr.").replace("etc", "etc.")
        s = s.replace("AM", "A.M.").replace("PM", "P.M.")
        if len(s.split()) < 4 and result:
            result[-1] += " " + s
        else:
            result.append(s)
    return result or [text]


# ── Misc handlers ─────────────────────────────────────────────────────────────

async def handle_test_immediate(ws: WebSocket, payload: dict):
    msg = payload.get("message", "Test message")
    await _safe_send_json(ws, {"type": "test_immediate_response", "message": f"Immediate: {msg}", "timestamp": time.time()})
    for i in range(3):
        await asyncio.sleep(1)
        await _safe_send_json(ws, {"type": "test_progress", "step": i + 1, "timestamp": time.time()})
    await _safe_send_json(ws, {"type": "test_final_response", "message": f"Final: {msg}", "timestamp": time.time()})


async def handle_audio_info_request(ws: WebSocket):
    await _safe_send_json(ws, {"type": "audio_info_response", "info": audio_manager.get_supported_formats()})


async def handle_voices_request(ws: WebSocket, payload: dict):
    language = payload.get("language", "en")
    voices   = await audio_manager.get_voice_options(language)
    await _safe_send_json(ws, {"type": "voices_response", "voices": voices})


async def handle_audio_query(ws: WebSocket, payload: dict):
    """Alias: audio → audio_base64."""
    await handle_audio_base64_query(ws, payload)
