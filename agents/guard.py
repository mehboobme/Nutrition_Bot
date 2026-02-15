"""Guardrails module for input safety filtering using Llama Guard."""
import os
import logging
from typing import Optional
from langchain_groq import ChatGroq as Groq

logger = logging.getLogger(__name__)

# Default model for Llama Guard
DEFAULT_LLAMA_GUARD_MODEL = "llama-guard-3-8b"


def _get_llama_guard_client() -> Optional[Groq]:
    """Lazily initialize the Llama Guard client."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set - guardrails will be disabled")
        return None
    return Groq(api_key=groq_api_key, model=DEFAULT_LLAMA_GUARD_MODEL)


def filter_input_with_llama_guard(
    user_input: str, 
    model: str = DEFAULT_LLAMA_GUARD_MODEL
) -> Optional[str]:
    """
    Filters user input using Llama Guard to ensure it is safe.

    Args:
        user_input: The input provided by the user.
        model: The Llama Guard model to be used for filtering.

    Returns:
        The safety classification result, or None if filtering fails.
    """
    client = _get_llama_guard_client()
    if client is None:
        logger.warning("Llama Guard client not available, skipping filter")
        return "SAFE"  # Fail open if guard not configured
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model=model,
        )
        result = response.choices[0].message.content.strip()
        logger.debug(f"Llama Guard result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error with Llama Guard: {e}")
        return None