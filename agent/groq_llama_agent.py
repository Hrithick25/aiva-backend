"""
Groq Llama AI Agent with RAG Integration
Optimized for minimum latency: no chat history, smart context injection, lean prompt.
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from groq import Groq

from rag_faiss.retriever import retrieve as query_knowledge_base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base system prompt — lean, no filler, no emoji, no follow-up questions.
# Injected on EVERY call. Keep this as short as possible.
# ---------------------------------------------------------------------------
_BASE_PROMPT = """You are AIVA, the AI assistant for Sri Eshwar College of Engineering (SECE).
Answer ONLY from the provided Context. Do NOT hallucinate. Be concise.

RULES:
1. If the answer is not in the Context, say "I don't have that information."
2. For Tamil queries, respond in Tanglish (Tamil + English).
3. For fee queries: "For fee details, please contact the reception at Sri Eshwar College of Engineering."
4. NO emojis anywhere in the response.
5. NO follow-up questions at the end.
6. NO filler sections: no "What this means", no "This information matters", no "Overall,".
7. Keep the response short and directly relevant to what was asked.

FORMAT:
[Direct 1-2 line answer]

- [Key point 1]
- [Key point 2]
- [Key point 3]
(Add more only if genuinely needed)

RESPOND ONLY as valid JSON: {"response": "<answer>", "emotion": "<happy|sad|none>"}"""

# ---------------------------------------------------------------------------
# Course/department context — injected ONLY when the query is about
# courses, departments, or programs. Avoids token waste on every call.
# ---------------------------------------------------------------------------
_COURSES_CONTEXT = """
SECE PROGRAMS (list exactly these, no additions):
B.Tech: Computer Science and Engineering | Cyber Security | Artificial Intelligence and Machine Learning | Artificial Intelligence and Data Science | Information Technology | Computer Science and Business Systems
B.E.: Electronics and Communication Engineering | Electrical and Electronics Engineering | Mechanical Engineering | Computer and Communication Engineering
M.E.: Computer Science and Engineering | VLSI Design | Engineering Design
NOTE: No MBA or MCA programs exist at SECE."""

# Keywords that trigger course context injection
_COURSE_KEYWORDS = frozenset([
    "course", "courses", "department", "departments", "program", "programs",
    "btech", "b.tech", "be ", "b.e.", "mtech", "m.tech", "me ", "m.e.",
    "branch", "branches", "specialization", "degree", "list"
])


def _needs_course_context(query: str) -> bool:
    """Return True if the query is about courses/departments."""
    q = query.lower()
    return any(kw in q for kw in _COURSE_KEYWORDS)


# ---------------------------------------------------------------------------
# Cached Groq client — avoids creating a new client per request
# ---------------------------------------------------------------------------
_groq_client: Optional[Groq] = None
_groq_client_key: Optional[str] = None


def _get_groq_client() -> Groq:
    global _groq_client, _groq_client_key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in environment")
    if _groq_client is None or api_key != _groq_client_key:
        _groq_client = Groq(api_key=api_key)
        _groq_client_key = api_key
    return _groq_client


# ---------------------------------------------------------------------------
# Main agent function — no chat history, direct RAG + LLM call
# ---------------------------------------------------------------------------
async def get_agent_response(
    user_query: str,
    language_context: Optional[Dict] = None,
    # Kept for API compatibility but ignored — no history needed
    chat_history: Optional[list] = None,
    rag_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a response from the Groq Llama agent with RAG context.

    Optimized for low latency:
    - No chat history passed to LLM (saves 50-200 tokens per call)
    - Course list injected only when relevant (saves ~150 tokens)
    - RAG timeout kept tight at 1.5s
    - Native JSON mode (no parsing overhead)
    - temperature=0.1, top_p=0.8 for fast, deterministic output
    - max_tokens=350 (enough for full course list + hostel info)
    """
    try:
        # 1. Language instruction (tiny, 5 tokens)
        lang_instruction = (
            "Respond in Tanglish (Tamil + English)."
            if (language_context and language_context.get("is_tamil"))
            else "Respond in English."
        )

        # 2. RAG retrieval (parallel-safe, 1.5s timeout)
        context = "No specific context found."
        try:
            loop = asyncio.get_running_loop()
            rag_start = time.time()
            rag_results = await asyncio.wait_for(
                loop.run_in_executor(None, query_knowledge_base, user_query),
                timeout=1.5,
            )
            rag_ms = (time.time() - rag_start) * 1000
            logger.info(f"RAG took {rag_ms:.0f}ms")

            if rag_results and isinstance(rag_results, dict):
                ctx = rag_results.get("context", "")
                if ctx:
                    context = ctx
                    logger.info(f"RAG sources: {rag_results.get('sources', [])}")
        except asyncio.TimeoutError:
            logger.warning("RAG timeout")
        except Exception as e:
            logger.warning(f"RAG error: {e}")

        # 3. Conditionally inject course list (saves ~150 tokens on non-course queries)
        extra_context = _COURSES_CONTEXT if _needs_course_context(user_query) else ""

        # 4. Build minimal prompt
        prompt = (
            f"{_BASE_PROMPT}\n\n"
            f"{lang_instruction}\n"
            f"{extra_context}\n"
            f"Context: {context}\n\n"
            f"Question: {user_query}"
        )

        logger.info(f"Prompt length: {len(prompt)} chars")

        # 5. LLM call
        client = _get_groq_client()
        llm_start = time.time()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # low = fast + deterministic
            max_tokens=350,    # tight ceiling; prompt governs brevity
            top_p=0.8,         # slightly tighter than 0.85 for speed
            stream=False,
            response_format={"type": "json_object"},  # native JSON — no parsing failures
        )
        llm_ms = (time.time() - llm_start) * 1000
        logger.info(f"LLM took {llm_ms:.0f}ms")

        text = response.choices[0].message.content.strip()

        # 6. Parse JSON (native json_object mode makes this nearly always succeed)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed, attempting extraction")
            parsed = {"response": text, "emotion": "none"}

        if "response" not in parsed or not parsed["response"]:
            parsed["response"] = "I don't have that information."
        if parsed.get("emotion") not in ("happy", "sad", "none"):
            parsed["emotion"] = "none"

        logger.info(f"Response ({len(parsed['response'])} chars): '{parsed['response'][:80]}...'")
        return parsed

    except Exception as error:
        logger.error(f"Agent error: {error}")
        return {
            "response": "I'm experiencing technical difficulties. Please try again.",
            "emotion": "sad",
        }