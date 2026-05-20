"""LLM calling layer for KTBench — generate kernel translations from any OpenAI-compatible API."""
from ktbench.llm.client import make_client
from ktbench.llm.agent import TranslationAgent

__all__ = ["make_client", "TranslationAgent"]
