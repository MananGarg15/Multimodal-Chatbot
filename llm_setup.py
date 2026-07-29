"""
Replaces Day4LLMCalling/callLlms.py's static Llms class and its match/case
source-selection block. Same 4 sources, same 'gpt-oss:20b' aliasing per
source, but returns a langchain_openai.ChatOpenAI instance instead of a raw
openai.OpenAI client + chat.completions.create(...) call.

Why ChatOpenAI instead of the raw client: it gives us .bind_tools(...) and
.stream(...) on the *same* object, which is the actual fix for the
streaming-vs-tools split between callModel (tools, no stream) and
callModelGenerator (stream, no tools) in the old code.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

DEFAULT_MODEL_ALIASES = {
    "ollama": "gpt-oss:20b",
    "gemini": "gemini-3-flash-preview",
    "openRouter": "openrouter/free",
    "openai": "gpt-5.4-nano",
}

_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openRouter": "https://openrouter.ai/api/v1",
    "openai": None,  # default OpenAI base url
}

_API_KEY_ENV = {
    "ollama": None,  # ollama ignores the key
    "gemini": "GOOGLE_API_KEY",
    "openRouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def resolve_model_name(source: str, model: str) -> str:
    """Mirrors the `if model == 'gpt-oss:20b': model = <alias>` branches that
    were duplicated across callModel/callModelGenerator in callLlms.py."""
    if model == "gpt-oss:20b" and source in DEFAULT_MODEL_ALIASES:
        return DEFAULT_MODEL_ALIASES[source]
    return model


def get_llm(source: str, model: str, temperature: float = 1.0, max_tokens: int = 4096):
    if source not in _BASE_URLS:
        raise ValueError(f"please select a valid source: {source!r}")

    resolved_model = resolve_model_name(source, model)
    api_key_env = _API_KEY_ENV[source]
    api_key = os.getenv(api_key_env) if api_key_env else "ollama"  # dummy non-empty key for local ollama

    kwargs = dict(model=resolved_model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    base_url = _BASE_URLS[source]
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


# A single OpenAI-native client for the two things only OpenAI's API supports:
# image generation and TTS. Neither gemini/ollama/openRouter proxy these
# endpoints in the old code either (artist()/talker() always used
# Llms.openai directly, never Llms.gemini etc.), so this mirrors that.
from openai import OpenAI  # noqa: E402

openai_native_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
