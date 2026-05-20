"""LLM utility functions: retry backoff, usage serialization, and candidate extraction."""
from __future__ import annotations

import random
import re
from typing import Any


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Parse Retry-After from OpenAI-compatible errors (header or message body)."""
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
            if headers is not None:
                raw = headers.get("retry-after") or headers.get("Retry-After")
                if raw is not None:
                    return float(raw)
    except (TypeError, ValueError):
        pass
    m = re.search(r"retry[_\s-]*after[:\s]+(\d+)", str(exc), re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def llm_retry_delay_s(
    attempt_idx: int,
    exc: BaseException,
    *,
    base: float = 2.0,
    max_s: float = 180.0,
) -> float:
    """Exponential backoff with jitter; honors Retry-After header when present."""
    ra = extract_retry_after_seconds(exc)
    if ra is not None and ra > 0:
        return min(ra + random.uniform(0.0, 1.5), max_s)
    exp = min(base ** float(attempt_idx), max_s)
    jitter = random.uniform(0.0, min(3.0, 0.25 * exp))
    return min(exp + jitter, max_s)


def llm_usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """Convert an OpenAI SDK usage object to a JSON-safe dict.

    Handles both Chat Completions (prompt_tokens / completion_tokens) and
    Responses API (input_tokens / output_tokens) shapes.
    """
    if usage is None:
        return None

    def _coerce(v: Any) -> Any:
        if v is None or isinstance(v, (bool, str, int, float)):
            return v
        if isinstance(v, dict):
            return {str(k): _coerce(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_coerce(x) for x in v]
        if hasattr(v, "model_dump"):
            try:
                return _coerce(v.model_dump(mode="python"))
            except TypeError:
                return _coerce(v.model_dump())
        return str(v)

    if hasattr(usage, "model_dump"):
        try:
            raw = usage.model_dump(mode="python")
        except TypeError:
            raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = dict(usage)
    else:
        return None

    out = _coerce(raw)
    return out if isinstance(out, dict) else None


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_candidate_from_text(text: str) -> str | None:
    """Return the last fenced Python block from model output if it contains ModelNew."""
    if not text or not text.strip():
        return None
    matches = list(_CODE_FENCE_RE.finditer(text))
    if not matches:
        return None
    code = matches[-1].group(1).strip()
    if "ModelNew" not in code:
        return None
    return code
