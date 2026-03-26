"""
gemini_agent.py — DEPRECATED: Redirects to groq_llama_agent.

This file is kept only for backward compatibility with any imports.
All AI responses now go through groq_llama_agent.py (Groq Llama-3.3-70b).
Gemini dependency has been completely removed from this project.
"""
import logging
from agent.groq_llama_agent import get_agent_response  # noqa: F401 — re-export

logger = logging.getLogger(__name__)
logger.warning(
    "[DEPRECATED] gemini_agent.py is no longer active. "
    "Import from agent.groq_llama_agent instead."
)
