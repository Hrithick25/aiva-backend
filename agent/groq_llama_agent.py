"""
Groq Llama AI Agent with RAG Integration
Handles conversational responses with adaptive response length
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
- For Tamil queries, respond in Tanglish (Tamil words written in English).
- Emotion: classify as "happy", "sad", or "none".

RESPOND ONLY in valid JSON with no extra keys:
{"response": "<formatted answer>", "emotion": "<happy|sad|none>"}"""




# Cached Groq client (avoids creating a new client per request)
_groq_client = None
_groq_client_key = None


def _get_groq_client() -> Groq:
    """Get cached Groq client, only recreating when the key changes."""
    global _groq_client, _groq_client_key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in environment")
    
    if _groq_client is None or api_key != _groq_client_key:
        _groq_client = Groq(api_key=api_key)
        _groq_client_key = api_key
    return _groq_client


async def get_agent_response(user_query: str, language_context: Optional[Dict] = None, chat_history: Optional[list] = None, rag_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Get response from Groq Llama agent with RAG integration.
    
    OPTIMIZED:
    - Compressed prompt (~250 tokens vs ~500)
    - Native JSON mode (no parsing failures)
    - Lower temperature (0.1) for speed + consistency
    - Adaptive response length (concise for facts, full for lists)
    """
    try:
        # Determine language instruction
        if language_context and language_context.get("is_tamil", False):
            language_instruction = "User asked in Tamil. Respond in Tanglish."
        else:
            language_instruction = "Respond in English."
        
        # Get relevant context from knowledge base (with timeout)
        try:
            loop = asyncio.get_running_loop()
            rag_start = time.time()
            query_to_search = rag_query if rag_query else user_query
            rag_results = await asyncio.wait_for(
                loop.run_in_executor(None, query_knowledge_base, query_to_search),
                timeout=8.0  # must be comfortably above embed read timeout (10s) minus network jitter
            )
            rag_ms = (time.time() - rag_start) * 1000
            logger.info(f"⏱️ RAG retrieval took {rag_ms:.0f}ms")
            
            if rag_results and isinstance(rag_results, dict):
                context = rag_results.get("context", "")
                sources = rag_results.get("sources", [])
                logger.info(f"RAG retrieved {len(sources)} sources: {sources}")
                context = context if context else "No specific context found."
            else:
                context = "No specific context found."
                
        except asyncio.TimeoutError:
            logger.warning(f"RAG retrieval timeout for query: {user_query}")
            context = "Knowledge base timeout - using general knowledge."
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            context = "Knowledge base temporarily unavailable."
        
        history_text = ""
        if chat_history:
            history_text = "\nConversation History:\n"
            for msg in chat_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_text += f"{role}: {msg.get('content')}\n"
        
        # Build compact prompt
        prompt = f"""{SYSTEM_PROMPT}

{language_instruction}
{history_text}
Context: {context}

Question: {user_query}"""

        logger.info(f"Context length: {len(context)} chars, Prompt length: {len(prompt)} chars")

        # Get Groq Llama client
        client = _get_groq_client()
        
        # OPTIMIZED: Native JSON mode + lower temperature + adaptive max_tokens
        llm_start = time.time()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # UPGRADED: smarter model, better reasoning
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,      # lower = more factual and consistent
            max_tokens=600,        # increased from 400 for richer answers
            top_p=0.85,
            stream=False,
            response_format={"type": "json_object"},
        )
        llm_ms = (time.time() - llm_start) * 1000
        logger.info(f"⏱️ LLM generation took {llm_ms:.0f}ms")
        
        text = response.choices[0].message.content.strip()
        logger.info(f"Groq raw response ({len(text)} chars): '{text[:120]}...'")
        
        # JSON mode ensures valid JSON, but still handle edge cases gracefully
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed despite json_object mode: {e}")
            
            # Fallback: try to extract response text
            partial = ""
            if '"response"' in text:
                try:
                    start_idx = text.find('"response"')
                    # Find the value after the key
                    colon_idx = text.find(':', start_idx)
                    if colon_idx >= 0:
                        remaining = text[colon_idx+1:].strip()
                        if remaining.startswith('"'):
                            end_idx = remaining.find('"', 1)
                            if end_idx > 0:
                                partial = remaining[1:end_idx]
                except Exception:
                    pass
            
            parsed = {
                "response": partial or "I'm having trouble processing your request. Please try again.",
                "emotion": "none"
            }
        
        # Ensure required keys exist with valid values
        if "response" not in parsed:
            parsed["response"] = text if text else "I don't have that information."
        if "emotion" not in parsed or parsed["emotion"] not in ("happy", "sad", "none"):
            parsed["emotion"] = "none"
        
        logger.info(f"✅ Agent response ({len(parsed['response'])} chars): '{parsed['response'][:80]}...' (emotion: {parsed['emotion']})")
        
        return parsed
        
    except Exception as error:
        logger.error(f"Groq Llama agent error: {error}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
            "emotion": "sad"
        }