"""Build an OpenAI-compatible client for OpenAI, Azure OpenAI, xAI Grok, or any compatible endpoint."""
from __future__ import annotations

import os

from openai import OpenAI

#: Provider presets — each maps to (api_key_env, base_url, api_kind, omit_responses_reasoning).
#: Pass these values to make_client() and TranslationAgent accordingly.
PROVIDER_PRESETS: dict[str, dict] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "api_kind": "responses",
        "omit_responses_reasoning": False,
    },
    "azure": {
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "base_url": None,          # caller must supply --base-url
        "api_kind": "responses",
        "omit_responses_reasoning": False,
    },
    "grok": {
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "api_kind": "chat",
        "omit_responses_reasoning": True,
    },
}


def make_client(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
) -> OpenAI:
    """
    Return a configured OpenAI client.

    Works for:
      OpenAI direct — api_key_env="OPENAI_API_KEY", base_url=None
      Azure OpenAI  — api_key_env="AZURE_OPENAI_API_KEY",
                      base_url="https://<resource>.cognitiveservices.azure.com/openai/v1/"
      xAI Grok      — api_key_env="XAI_API_KEY", base_url="https://api.x.ai/v1"
      Any compatible endpoint — set base_url accordingly.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{api_key_env}' is not set. Export it before running."
        )
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
