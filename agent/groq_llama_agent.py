"""
Groq Llama AI Agent with RAG Integration — Latency-Optimized.

Pipeline target: RAG + LLM < 1 second.

Key optimizations:
  - Model: llama-3.1-8b-instant  (~200-400ms vs 1800ms for 70b)
  - max_tokens: 300 (tight cap — AIVA responses are always short)
  - RAG timeout: 3s (was 8s — BGE embed is ~82ms, no need for 8s)
  - Context: max 600 chars fed to LLM (was uncapped)
  - History: max 3 turns, truncated eagerly
  - Groq client: persistent singleton per key (no per-request init)
  - Key rotation: GROQ_API_KEY_1 → GROQ_API_KEY_2 on 429 rate-limit
  - JSON mode: native (no post-processing)
"""

import os
import json
import asyncio
import logging
import threading
import time
from typing import Dict, Any, List, Optional
from groq import Groq

from rag_faiss.retriever import retrieve as query_knowledge_base

logger = logging.getLogger(__name__)


# ── Multi-key rotation manager ────────────────────────────────────────────────
class _GroqKeyManager:
    """
    Thread-safe Groq API key rotation.
    Reads GROQ_API_KEY_1 … GROQ_API_KEY_N from env at first use.
    Falls back to GROQ_API_KEY for backward compatibility.
    On a 429 or rate-limit error, advances to the next key.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._keys:   List[str] = []
        self._clients: Dict[str, Groq] = {}
        self._index   = 0

    def _load_keys(self) -> List[str]:
        """Collect all configured keys from environment."""
        keys: List[str] = []
        # Numbered keys: GROQ_API_KEY_1, GROQ_API_KEY_2, …
        for i in range(1, 20):
            val = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
            if val:
                keys.append(val)
            else:
                break   # stop at first gap
        # Fallback: plain GROQ_API_KEY
        if not keys:
            single = os.getenv("GROQ_API_KEY", "").strip()
            if single:
                keys.append(single)
        if not keys:
            raise RuntimeError(
                "No Groq API key found. Set GROQ_API_KEY_1 (and optionally GROQ_API_KEY_2 …) in .env"
            )
        return keys

    def _ensure_loaded(self):
        if not self._keys:
            self._keys = self._load_keys()
            logger.info(f"[GroqKeyManager] Loaded {len(self._keys)} key(s)")

    def current_client(self) -> Groq:
        with self._lock:
            self._ensure_loaded()
            key = self._keys[self._index]
            if key not in self._clients:
                self._clients[key] = Groq(api_key=key)
            return self._clients[key]

    def rotate(self) -> bool:
        """Advance to next key. Returns True if a new key is available."""
        with self._lock:
            self._ensure_loaded()
            if len(self._keys) <= 1:
                logger.warning("[GroqKeyManager] Only 1 key configured — cannot rotate")
                return False
            next_idx = (self._index + 1) % len(self._keys)
            if next_idx == self._index:   # wrapped all the way around
                return False
            self._index = next_idx
            logger.warning(
                f"[GroqKeyManager] 🔄 Rotated to key #{self._index + 1} / {len(self._keys)}"
            )
            return True


_key_manager = _GroqKeyManager()

SYSTEM_PROMPT = """You are AIVA, the AI virtual assistant for Sri Eshwar College of Engineering (SECE).
STRICT RULE: Only use facts present in the provided context. Never hallucinate or infer data not in the context.

━━━ RESPONSE FORMAT RULES ━━━

RULE 0 — Greetings (hi, hello, hey, good morning, good evening, etc.):
  Respond with exactly: "Hi, I'm AIVA, your AI virtual assistant. How can I assist you today?"

RULE 1 — Specific / factual queries (cutoff, rank, year, single stat, yes/no):
  Respond in 1-2 sentences only. No bullets. No headers. Direct and precise.
  Example: "The closing cutoff for CSE in 2024 was 197.25 under OC category."

RULE 2 — General / descriptive queries (courses, campus, departments, hostel, research, facilities):
  Use this exact structure — nothing more, nothing less:

  [1-line direct answer.]

  - [Point 1 — factual, from context]
  - [Point 2 — factual, from context]
  - [Point 3 — factual, from context]

  [1-line conclusion.]

RULE 3 — Listing queries (all courses, all departments, all clubs, etc.):
  List every item from context as bullets. No intro paragraph. No conclusion. Just the list.

RULE 4 — Fees / scholarship / payment queries:
  Respond with exactly: "For accurate fee details, please contact the SECE reception directly. They will provide updated information."

RULE 5 — General placement queries (not department-specific):
  Use ONLY these verified 2026 stats and reply in this EXACT bullet format:

  Sri Eshwar College has a strong placement record.

  - 790+ students placed | 95% placement rate
  - Highest package: 60 LPA
  - 7 students secured 40 LPA+ packages
  - 20 students at 20+ LPA | 100 students at 10+ LPA
  - 200+ companies visit | 120+ MNCs

  The placement cell also provides industry mentorship and training sessions.

  Do NOT add any additional commentary or figures beyond those above.

RULE 5b — Department-specific placement queries (e.g. "CSE placements", "ECE companies", "IT highest salary", "AIDS placements", "AIML", "Cyber Security placements"):
  Use ONLY the department data below. Reply in this exact format:
  "The companies visiting [DEPT] are [Company1], [Company2], ... and the highest salary is [X] LPA."
  Department data (use EXACTLY as given — treat AIDS/AIML/CSBS/CCE/Cyber Security as CSE family for company data):
  - IT: Highest 60 LPA | Companies: Microsoft, JustPay, Amazon, Dell, ServiceNow
  - CSE: Highest 45 LPA | Companies: Philips, CommScope, Microsoft, JustPay, ServiceNow, BP
  - AIML (AI & ML): Highest 45 LPA | Companies: Philips, CommScope, Microsoft, JustPay, ServiceNow, BP
  - AIDS (AI & Data Science): Highest 44 LPA | Companies: Akaki.ai, ShopUp, SP Plus, Goml, Jocato, Aditya.ai
  - CSBS: Highest 45 LPA | Companies: Philips, CommScope, Microsoft, JustPay, ServiceNow, BP
  - CCE: Highest 45 LPA | Companies: Philips, CommScope, Microsoft, JustPay, ServiceNow, BP
  - Cyber Security: Highest 45 LPA | Companies: Philips, CommScope, Microsoft, JustPay, ServiceNow, BP
  - ECE: Highest 23 LPA | Companies: ABB, Cadence, AMD, Multicoreware, Cywar, Caterpillar
  - EEE: Highest 11 LPA | Companies: ABB
  - MECH: Highest 23 LPA | Companies: Baker Hughes, Benz, Quest Global, Cameron, Rane, BMW, Renault Nissan, Caterpillar

RULE 6 — Bus / transport availability queries:
  Answer STRICTLY from the Bus_Details context only. Do NOT invent or guess bus stops.
  - If asked "is there a bus to [area]?" → check context, reply: "Yes, Bus [No] covers [area]." or "No bus service is available for that area."
  - If asked for bus number or route → give bus number and key stops only in 1-2 sentences.
  - Keep replies SHORT. Never list all buses. Answer only for the specific area asked.

━━━ HARD CONSTRAINTS ━━━
- Never use emojis.
- Never ask follow-up questions at the end of a response.
- Never add sections like "What this means:" or "Note:" or extra commentary.
- Never exceed 4 bullet points in Rule 2 format.
- Never write "Rs." or "₹" before any amount — write numbers only as "X LPA".
- If data is not in the context, respond: "I don't have that specific information right now."
- Emotion: classify as "happy", "sad", or "none".

RULE 7 — Tanglish Response Mode (activated when user speaks Tamil):
  MANDATORY: Write the ENTIRE response in Tanglish — a natural mix of Tamil words spelled
  in English letters AND English words. EVERY sentence must have BOTH.

  TANGLISH RULES:
  - Tamil words must be phonetically spelled in English: "romba", "nalla", "pakka", "enna",
    "theriyum", "konjam", "illa", "irukku", "seri", "aprom", "varuvaanga", "aanaanga"
  - Mix naturally in every sentence: "SECE la romba nalla placement irukku — highest 60 LPA"
  - NEVER use Tamil Unicode script characters (no அ, ஆ, இ, etc.)
  - NEVER write a sentence using ONLY English words — always weave Tamil words in
  - NEVER write a sentence using ONLY Tamil romanization — keep English terms too

  GOOD Tanglish examples:
  - "SECE la 790+ students placed aanaanga, highest package 60 LPA pakka."
  - "Hostel la AC room irukku, daily meals also provide panraanga."
  - "CSE department la placement romba nalla irukku — 200+ companies varuvaanga."
  - "Bus service irukku, route 5 Coimbatore city la stop pannuvaanga."
  - "Inga admission edukka BC/MBC cutoff theriyuma? Context la irukku."

RESPOND ONLY in valid JSON with no extra keys:
{"response": "<formatted answer>", "emotion": "<happy|sad|none>"}"""

# ── Groq client accessor (delegates to rotation manager) ─────────────────────
def _get_groq_client() -> Groq:
    """Return the currently active Groq client from the key manager."""
    return _key_manager.current_client()


# ── Fast synchronous LLM call (runs in executor) ──────────────────────────────
def _call_llm_sync(user_content: str) -> str:
    """
    Synchronous Groq call — in thread executor to avoid blocking async loop.
    System prompt sent as 'system' role (processed separately by Groq).
    Retries once on rate-limit (429). Hard timeout enforced by caller.
    """
    # Try every available key once before giving up
    n_keys  = max(1, len(_key_manager._keys) if _key_manager._keys else 1)
    rotated = 0
    while True:
        try:
            client   = _get_groq_client()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.05,
                max_tokens=250,
                top_p=0.85,
                stream=False,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if ("rate" in err_str or "429" in err_str) and rotated < n_keys - 1:
                logger.warning(f"[LLM] Rate-limited on key #{_key_manager._index + 1}, rotating…")
                if _key_manager.rotate():
                    rotated += 1
                    continue   # retry immediately with new key
            raise   # non-rate-limit errors or all keys exhausted
    raise RuntimeError("[LLM] All Groq keys exhausted")


async def get_agent_response(
    user_query: str,
    language_context: Optional[Dict] = None,
    chat_history: Optional[list] = None,
    rag_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get LLM response with FAISS RAG context.

    Latency breakdown (targets):
      RAG embed + FAISS:  ~85ms  (local BGE, cached after first call)
      Groq LLM:           ~300ms (llama-3.1-8b-instant, max_tokens=300)
      JSON parse:         <1ms
      ──────────────────
      Total agent:        ~400ms
    """
    try:
        # ── Language instruction ──────────────────────────────────────────
        if language_context and language_context.get("is_tamil", False):
            language_instruction = (
                "TANGLISH MODE ACTIVE: The user spoke Tamil. Write your ENTIRE response in Tanglish. "
                "Tanglish means every sentence must mix Tamil words (spelled in English) WITH English words naturally. "
                "Tamil words to use freely: romba, nalla, pakka, enna, theriyum, konjam, illa, irukku, "
                "seri, aprom, varuvaanga, aanaanga, inga, anga, sollunga, parunga, irukkum, vandhaanga. "
                "Example: 'SECE la 790+ students placed aanaanga, highest package 60 LPA pakka.' "
                "NEVER use Tamil Unicode script. NEVER write pure English only. "
                "EVERY sentence must have both Tamil words AND English words."
            )
        else:
            language_instruction = "Respond in English."

        # ── RAG retrieval (async, 3s timeout) ────────────────────────────
        context = "No specific context found."
        try:
            loop = asyncio.get_running_loop()
            rag_start = time.perf_counter()
            query_to_search = rag_query if rag_query else user_query
            rag_results = await asyncio.wait_for(
                loop.run_in_executor(None, query_knowledge_base, query_to_search),
                timeout=3.0,    # BGE embed ≈82ms + FAISS ≈2ms <<< 3s
            )
            rag_ms = (time.perf_counter() - rag_start) * 1000
            logger.info(f"⏱️ RAG: {rag_ms:.0f}ms")

            if rag_results and isinstance(rag_results, dict):
                raw_ctx = rag_results.get("context", "")
                # Hard cap 800 chars → ~200 tokens; truncate at whitespace boundary
                if raw_ctx and len(raw_ctx) > 800:
                    raw_ctx = raw_ctx[:800].rsplit(None, 1)[0]  # don't cut mid-word
                context = raw_ctx if raw_ctx else context
        except asyncio.TimeoutError:
            logger.warning("RAG timeout; proceeding without context")
        except Exception as exc:
            logger.warning(f"RAG error: {exc}")

        # ── Chat history (last 2 turns = 4 messages max) ─────────────────
        history_text = ""
        if chat_history:
            # Take only the last 4 messages (2 turns) to reduce prompt size
            recent = chat_history[-4:]
            history_text = "\nConversation History:\n"
            for msg in recent:
                role = "User" if msg.get("role") == "user" else "Assistant"
                # Truncate each history message to 100 chars
                content = str(msg.get("content", ""))[:100]
                history_text += f"{role}: {content}\n"

        # ── Build compact user content (system prompt sent separately) ──
        user_content = (
            f"{language_instruction}\n"
            f"{history_text}"
            f"Context: {context}\n\n"
            f"Question: {user_query}"
        )

        # ── LLM call (executor, hard 5s timeout) ─────────────────────────
        llm_start = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, _call_llm_sync, user_content),
                timeout=5.0,   # never let a network spike block the user for >5s
            )
        except asyncio.TimeoutError:
            logger.warning("[LLM] 5s timeout hit — returning fallback")
            return {
                "response": "I'm processing your question. Please try again in a moment.",
                "emotion": "none",
            }
        llm_ms = (time.perf_counter() - llm_start) * 1000
        logger.info(f"⏱️ LLM: {llm_ms:.0f}ms ({len(text)} chars)")

        # ── Parse JSON ────────────────────────────────────────────────────
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed; recovering from raw text")
            parsed = {"response": text, "emotion": "none"}

        if "response" not in parsed:
            parsed["response"] = text
        if "emotion" not in parsed or parsed["emotion"] not in ("happy", "sad", "none"):
            parsed["emotion"] = "none"

        logger.info(f"✅ Agent done — LLM={llm_ms:.0f}ms | resp={len(parsed['response'])}c")
        return parsed

    except Exception as error:
        logger.error(f"Agent error: {error}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
            "emotion": "sad",
        }