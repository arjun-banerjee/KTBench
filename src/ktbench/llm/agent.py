"""TranslationAgent — call an LLM to translate a kernel from source DSL to target DSL.

Supports the OpenAI Responses API (OpenAI, Azure) and the Chat Completions API
(Grok, any OpenAI-compatible endpoint). The caller constructs the client via
make_client() and passes it in; this keeps the agent agnostic to provider wiring.

Typical use:
    from ktbench.llm import make_client, TranslationAgent
    from ktbench.problem import load_problem

    problem = load_problem("problems/softmax_h200_to_triton")
    client  = make_client(api_key_env="OPENAI_API_KEY")
    agent   = TranslationAgent(client=client, model="gpt-4o", problem=problem)
    src     = agent.generate()   # returns candidate ModelNew source string
"""
from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from ktbench.problem import Problem
from ktbench.prompt import build_prompt
from ktbench.llm.utils import extract_candidate_from_text, llm_retry_delay_s, llm_usage_to_dict

_SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer specializing in DSL translation. "
    "You will be given a working kernel implementation in one DSL and asked to "
    "translate it to another DSL and hardware target.\n\n"
    "Output ONLY a fenced Python code block containing the complete ModelNew "
    "class implementation. Do not include prose or explanation after the code block.\n\n"
    "The class must define `forward(self, *inputs)` with the same input/output "
    "signature as the source kernel. Do not hardcode shapes or values — inputs "
    "are freshly randomized each run."
)


def _strip_status(obj: Any) -> None:
    """Recursively remove 'status' keys from Responses-API items before echoing.

    Azure's /openai/v1/ preview returns 400 if 'status' is present in the input
    array; the public OpenAI API ignores it. Strip it to keep both happy.
    """
    if isinstance(obj, dict):
        obj.pop("status", None)
        for v in obj.values():
            _strip_status(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_status(v)


class TranslationAgent:
    """Single-shot LLM agent for kernel translation.

    Args:
        client:                     Pre-configured OpenAI client (from make_client()).
        model:                      Model name, e.g. "gpt-4o", "gpt-4.1", "grok-3".
        problem:                    KTBench Problem to translate.
        api_kind:                   "responses" (default, supports reasoning models) or
                                    "chat" (Chat Completions — use for Grok and other
                                    endpoints that don't speak the Responses API).
        reasoning_effort:           "minimal" | "low" | "medium" | "high" | None.
                                    Passed to the Responses API reasoning.effort field.
        omit_responses_reasoning:   Set True for providers that reject the reasoning
                                    parameter (e.g. xAI Grok on the Responses API).
        max_api_retries:            Retry budget for transient API errors or empty responses.
        verbose:                    Print progress to stdout.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        problem: Problem,
        api_kind: str = "responses",
        reasoning_effort: str | None = None,
        omit_responses_reasoning: bool = False,
        max_api_retries: int = 3,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.problem = problem
        self.api_kind = api_kind
        self.reasoning_effort = reasoning_effort
        self.omit_responses_reasoning = omit_responses_reasoning
        self.max_api_retries = max(1, max_api_retries)
        self.verbose = verbose
        self.last_usage: dict | None = None

    def generate(self) -> str:
        """Call the LLM and return a ModelNew candidate source string.

        Retries up to max_api_retries times on API errors or if the model
        response contains no extractable ModelNew code block.

        Raises:
            RuntimeError: if no candidate is extracted after all retries.
        """
        prompt = build_prompt(self.problem)
        last_exc: BaseException | None = None

        for attempt in range(self.max_api_retries):
            try:
                if self.api_kind == "responses":
                    text, usage = self._call_responses(prompt)
                else:
                    text, usage = self._call_chat(prompt)
                self.last_usage = usage

                code = extract_candidate_from_text(text)
                if code:
                    if self.verbose:
                        print(
                            f"[TranslationAgent] attempt {attempt + 1}: "
                            f"extracted {len(code)} chars"
                        )
                    return code

                if self.verbose:
                    print(
                        f"[TranslationAgent] attempt {attempt + 1}: "
                        "no ModelNew code block in response — retrying"
                    )
                last_exc = RuntimeError("No ModelNew code block found in response")

            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self.max_api_retries:
                    break
                delay = llm_retry_delay_s(attempt, exc)
                if self.verbose:
                    print(
                        f"[TranslationAgent] attempt {attempt + 1} API error: "
                        f"{exc}; retrying in {delay:.1f}s"
                    )
                time.sleep(delay)

        raise RuntimeError(
            f"Translation failed after {self.max_api_retries} attempt(s)"
        ) from last_exc

    # ------------------------------------------------------------------
    # Internal call helpers
    # ------------------------------------------------------------------

    def _call_responses(self, prompt: str) -> tuple[str, dict | None]:
        """Call the OpenAI Responses API. Returns (response_text, usage_dict)."""
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": _SYSTEM_PROMPT,
            "input": [{"role": "user", "content": prompt}],
        }
        if not self.omit_responses_reasoning:
            r_params: dict[str, Any] = {"summary": "auto"}
            if self.reasoning_effort:
                r_params["effort"] = self.reasoning_effort
            create_kwargs["reasoning"] = r_params

        response = self.client.responses.create(**create_kwargs)

        chunks: list[str] = []
        for item in response.output:
            d = item.model_dump() if hasattr(item, "model_dump") else item
            _strip_status(d)
            if isinstance(d, dict) and d.get("type") == "message":
                for part in d.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        t = part.get("text")
                        if isinstance(t, str):
                            chunks.append(t)

        usage = llm_usage_to_dict(getattr(response, "usage", None))
        return "\n".join(chunks), usage

    def _call_chat(self, prompt: str) -> tuple[str, dict | None]:
        """Call the OpenAI Chat Completions API. Returns (response_text, usage_dict)."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = llm_usage_to_dict(getattr(response, "usage", None))
        return text, usage
