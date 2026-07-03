from __future__ import annotations

import os


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def generate_with_llm(_: str) -> str | None:
    # MVP keeps generation deterministic. This hook documents where an
    # OpenAI-compatible client can enrich wording in production.
    return None

