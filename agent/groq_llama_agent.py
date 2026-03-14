"""
Groq Llama-4 Scout AI Agent (Replacing Gemini)
Handles conversational responses with RAG integration
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from groq import Groq

from rag_faiss.retriever import retrieve as query_knowledge_base

logger = logging.getLogger(__name__)

# System prompt for Groq Llama agent
SYSTEM_PROMPT = """You are AIVA, a helpful AI assistant for Sri Eshwar College. Give accurate, specific answers using the provided context.

RESPONSE FORMAT: Always respond in JSON:
{
    "response": "Your concise answer (2-3 sentences max)",
    "emotion": "none" or "happy" or "sad"
}

GUIDELINES:
- Keep responses SHORT and specific (2-3 sentences). Avoid long paragraphs.
- When context has exact numbers/marks/fees, include them.
- Be conversational and friendly.
- If you don't know, say so honestly.
- For Tamil queries: respond ONLY in Tamil script (தமிழ்). Do NOT use English or Tanglish."""


# Cached Groq client (avoids creating a new client per request)
_groq_client = None
_groq_client_key = None


def _get_groq_client() -> Groq:
    """Get cached Groq client, only recreating when the key changes.
    
    OPTIMIZED: Previously created a new Groq() client on every single request.
    Now caches the client and reuses the connection.
    """
    global _groq_client, _groq_client_key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in environment")
    
    if _groq_client is None or api_key != _groq_client_key:
        _groq_client = Groq(api_key=api_key)
        _groq_client_key = api_key
    return _groq_client


async def get_agent_response(user_query: str, language_context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get response from Groq Llama-4 Scout agent with RAG integration
    
    Args:
        user_query: User's question
        language_context: Optional language detection info
        
    Returns:
        Response with text and emotion
    """
    try:
        # Determine language instruction
        if language_context and language_context.get("is_tamil", False):
            language_instruction = "User asked in Tamil. You MUST respond in Tamil script (தமிழ்) only. Do NOT use English."
        else:
            language_instruction = "User asked in English. Respond in clear, concise English."
        
        # Get relevant context from knowledge base (with timeout)
        # OPTIMIZED: Use get_running_loop() (not deprecated get_event_loop())
        try:
            loop = asyncio.get_running_loop()
            
            # For Tamil queries, create an English search version for better RAG results
            # (embeddings were created from English documents)
            rag_query = user_query
            is_tamil_query = any('\u0B80' <= c <= '\u0BFF' for c in user_query)
            if is_tamil_query:
                # Append English keywords alongside Tamil to improve embedding match
                # The Gemini embedding model handles multilingual input well
                rag_query = f"[Tamil query] {user_query}"
                logger.info(f"Tamil query detected, using bilingual RAG query")
            
            rag_results = await asyncio.wait_for(
                loop.run_in_executor(None, query_knowledge_base, rag_query),
                timeout=3.0  # Increased from 1.5s — REST API embedding is slower than SDK
            )
            
            # Extract context properly from RAG results dictionary
            if rag_results and isinstance(rag_results, dict):
                context = rag_results.get("context", "")
                sources = rag_results.get("sources", [])
                logger.info(f"RAG retrieved {len(sources)} sources: {sources}")
                context = context if context else "No specific context found."
            else:
                context = "No specific context found."
                logger.warning(f"RAG returned unexpected format: {type(rag_results)}")
                
        except asyncio.TimeoutError:
            logger.warning(f"RAG retrieval timeout for query: {user_query}")
            context = "Knowledge base timeout - using general knowledge."
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            context = "Knowledge base temporarily unavailable."
        
        # Build prompt
        prompt = f"""{SYSTEM_PROMPT}

LANGUAGE INSTRUCTION: {language_instruction}

Context from knowledge base: {context}

User Question: {user_query}

Response (in JSON format):"""

        # Debug logging
        logger.info(f"Context length: {len(context)} chars")
        logger.debug(f"Full context: {context[:200]}...")
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # Get Groq Llama client
        client = _get_groq_client()
        
        # Generate response with optimized settings for speed
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fastest Groq model for immediate responses
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for faster, more focused responses
            max_tokens=150,   # Short responses for fast TTS (~2-3 sentences)
            top_p=0.9,
            stream=False
        )
        
        text = response.choices[0].message.content.strip()
        
        logger.info(f"Groq Llama raw response: '{text[:100]}...'")
        
        # Clean up response format
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            # Parse JSON response
            parsed = json.loads(text)
            
            # Validate required keys
            if "response" not in parsed or "emotion" not in parsed:
                raise json.JSONDecodeError("Missing required keys", text, 0)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from Groq Llama (length={len(text)}): {e}")
            logger.debug(f"Raw response: '{text[:100]}...'")
            
            # Enhanced error recovery
            partial_response = ""
            
            if '"response"' in text:
                try:
                    # Extract response content
                    start_patterns = ['"response": "', '"response":"', ': "']
                    for pattern in start_patterns:
                        start_idx = text.find(pattern)
                        if start_idx >= 0:
                            start_idx += len(pattern)
                            remaining = text[start_idx:]
                            
                            # Find natural end points
                            end_patterns = ['",', '"', '\\n', "'}"]
                            min_end = len(remaining)
                            
                            for end_pattern in end_patterns:
                                end_pos = remaining.find(end_pattern)
                                if end_pos >= 0 and end_pos < min_end:
                                    min_end = end_pos
                            
                            if min_end < len(remaining):
                                partial_response = remaining[:min_end].strip()
                                break
                    
                    # Clean extracted response
                    if partial_response:
                        partial_response = partial_response.replace('\\"', '"')
                        partial_response = partial_response.replace('\\n', ' ')
                        partial_response = partial_response.strip()
                        
                except Exception as extraction_error:
                    logger.debug(f"Response extraction failed: {extraction_error}")
            
            # Final fallback
            if not partial_response or len(partial_response.strip()) < 3:
                partial_response = "I apologize, but I'm having trouble processing your request right now."
            
            parsed = {
                "response": partial_response,
                "emotion": "none"
            }
            
            logger.info(f"Recovered Groq response: '{partial_response[:50]}...'")
        
        # Ensure required keys exist with valid values
        if "response" not in parsed:
            parsed["response"] = text if text else "I don't have that information."
        if "emotion" not in parsed or parsed["emotion"] not in ("happy", "sad", "none"):
            parsed["emotion"] = "none"
        
        logger.info(f"Groq Llama response: '{parsed['response'][:50]}...' (emotion: {parsed['emotion']})")
        
        return parsed
        
    except Exception as error:
        logger.exception("Groq Llama agent error")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
            "emotion": "sad"
        }