"""
llm_provider.py
----------------
Centralised LLM provider for IDAMP. Returns None gracefully when no API
key is configured so every agent can fall back to rule-based logic.

Supports Groq (preferred) and Google Gemini. Provider is selected by the
LLM_PROVIDER env var; if neither key is set, returns None.
"""

import logging

logger = logging.getLogger("llm_provider")


def get_chat_model():
    """Return a configured LangChain chat model, or None if no key is set.

    Caller pattern:
        model = llm_provider.get_chat_model()
        if model is None:
            return rule_based_fallback(...)
        agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    """
    from core.config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL

    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            logger.debug("LLM provider: Groq (%s)", GROQ_MODEL)
            return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        except Exception as e:
            logger.warning("Groq init failed: %s — falling back to Gemini", e)

    if GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.debug("LLM provider: Gemini (%s)", GEMINI_MODEL)
            return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)

    logger.info("No LLM API key configured — rule-based fallback mode active.")
    return None


def is_configured() -> bool:
    """True if at least one LLM provider has a key set."""
    from core.config import LLM_PROVIDER, GROQ_API_KEY, GOOGLE_API_KEY
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return True
    if GOOGLE_API_KEY:
        return True
    return False
