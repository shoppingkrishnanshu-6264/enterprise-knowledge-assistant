"""
Centralized Groq client construction for the Enterprise Knowledge Assistant.

Why this exists:
Every module that talked to Groq directly (graph.py, sql_tool.py) called
os.getenv("GROQ_API_KEY") and passed it straight into ChatGroq(...). If that
env var ever picks up a stray non-ASCII character (e.g. from a copy-paste
that grabs invisible/hidden characters), httpx fails deep inside header
construction with a cryptic UnicodeEncodeError that has nothing to do with
your actual code logic. This module validates the key once, in one place,
and fails with a clear, actionable error instead.
"""

import os
from functools import lru_cache
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def _load_and_validate_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file at the project root."
        )

    key = key.strip()

    non_ascii = [(i, c) for i, c in enumerate(key) if ord(c) > 127]
    if non_ascii:
        positions = ", ".join(f"{i}:{c!r}" for i, c in non_ascii)
        raise ValueError(
            f"GROQ_API_KEY contains non-ASCII character(s) at {positions}. "
            "This usually happens from a copy-paste that grabs a hidden/invisible "
            "character. Re-copy the key directly from the Groq console (use the "
            "copy icon, don't manually select text) and replace it in .env."
        )

    return key


@lru_cache(maxsize=None)
def get_groq_client(model: str, temperature: float = 0) -> ChatGroq:
    """
    Returns a cached ChatGroq client for the given (model, temperature) pair.
    Caching avoids re-validating the key and re-instantiating a client on
    every call, while still giving each distinct model/temperature its own
    client instance.
    """
    api_key = _load_and_validate_api_key()
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)
