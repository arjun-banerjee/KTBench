"""
Shared helpers for KTBench harness integrations.

All harnesses (Inspect AI, LangGraph, smolagents, AutoGen) expect tools to return
str, not a custom type.  fmt_result() serializes ToolResult at the tool boundary.
extract_final_score() recovers the submit_kernel metadata from a message history.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure both src/ (ktbench package) and repo root (tools package) are importable
# from any working directory.
_REPO = Path(__file__).parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.tools import ToolResult  # noqa: E402  (after path setup)


def fmt_result(result: ToolResult) -> str:
    """Serialize a ToolResult to a string the LLM can read."""
    if not result.metadata:
        return result.output
    return result.output + "\n\n[meta] " + json.dumps(result.metadata, default=str)


def extract_final_score(messages) -> dict | None:
    """Scan a message history for the last submit_kernel [meta] block."""
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or str(msg)
        if isinstance(content, list):
            # Inspect AI (and some other harnesses) may store content as a list
            # of content blocks; flatten to a single string.
            content = " ".join(
                (getattr(b, "text", "") or str(b)) for b in content
            )
        if "submit_kernel" in content and "[meta]" in content:
            try:
                meta_str = content.split("[meta]")[-1].strip()
                return json.loads(meta_str)
            except (json.JSONDecodeError, ValueError):
                pass
    return None
