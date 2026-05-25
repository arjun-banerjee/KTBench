"""TranslationAgent — call an LLM to translate a kernel, with optional tool calling.

Single-shot mode (default):
    Build prompt → LLM call → extract ModelNew → return source string.

Tool-calling mode (pass tools=[...]):
    Multi-turn loop where the model can call compile_kernel, run_correctness,
    get_gpu_specs, static_check, and submit_kernel before committing to a final
    answer. Loop ends when submit_kernel is called or max_turns is reached.

Both the OpenAI Responses API and Chat Completions API are supported.

Usage (single-shot):
    from ktbench.llm import make_client, TranslationAgent
    client = make_client(api_key_env="OPENAI_API_KEY")
    agent  = TranslationAgent(client=client, model="gpt-4o", problem=problem)
    src    = agent.generate()

Usage (multi-turn with tools):
    from tools import get_tools, ToolContext
    ctx    = ToolContext(problem=problem, device=0)
    tools  = get_tools()
    agent  = TranslationAgent(client=client, model="o3", problem=problem, tools=tools, tool_ctx=ctx)
    result = agent.generate()   # returns TranslationResult (from submit_kernel)
"""
from __future__ import annotations

import json
import time
import traceback
from typing import Any, TYPE_CHECKING

from openai import OpenAI

from ktbench.problem import Problem
from ktbench.prompt import build_prompt
from ktbench.llm.utils import (
    extract_candidate_from_text,
    llm_retry_delay_s,
    llm_usage_to_dict,
)

if TYPE_CHECKING:
    from tools.tools import Tool, ToolContext, ToolResult

_SYSTEM_PROMPT_SINGLE_SHOT = (
    "You are an expert GPU kernel engineer specializing in DSL translation. "
    "You will be given a working kernel implementation in one DSL and asked to "
    "translate it to another DSL and hardware target.\n\n"
    "Output ONLY a fenced Python code block containing the complete ModelNew "
    "class implementation. Do not include prose or explanation after the code block.\n\n"
    "The class must define `forward(self, *inputs)` with the same input/output "
    "signature as the source kernel. Do not hardcode shapes or values — inputs "
    "are freshly randomized each run."
)

_SYSTEM_PROMPT_TOOLS_TMPL = (
    "You are an expert GPU kernel engineer specializing in DSL translation. "
    "Translate the given source kernel to the target DSL and hardware, then verify "
    "and optimize it using the available tools.\n\n"
    "You have {max_turns} turns. Each turn is one response from you, optionally "
    "with tool calls. submit_kernel records your final result and ends the session — "
    "call it once when you are satisfied.\n\n"
    "Iteration loop: write a kernel → compile_kernel → run_correctness → fix or "
    "optimize → repeat → submit_kernel. There is no required order and no required "
    "minimum number of iterations.\n\n"
    "Scoring: correctness × stress_pass_rate × SOL (Speed-of-Light — hardware "
    "utilization on a 0–1 scale). SOL is bounded by physics; you cannot game it "
    "by slowing a baseline.\n\n"
    "Rules:\n"
    "- Do not hardcode shapes or values — inputs are freshly randomized each run.\n"
    "- The kernel must actually execute on the GPU — pure PyTorch fallbacks or "
    "dead CUDA extensions that never launch will be rejected by the utilization gate.\n"
    "- Call submit_kernel exactly once when done — it ends the session."
)

_NO_TOOL_CALLS_NUDGE = (
    "You did not call any tools. Please use compile_kernel, run_correctness, "
    "or submit_kernel to make progress."
)

_TURN_WARNING = (
    "Heads up: you have {turns_left} turn(s) left. "
    "If your kernel is correct, consider calling submit_kernel soon."
)

_FINAL_TURN_NUDGE = (
    "This is your last turn. Call submit_kernel now with your best kernel implementation."
)


def _strip_status(obj: Any) -> None:
    """Remove 'status' keys from Responses-API output items before echoing back.

    Azure's /openai/v1/ preview rejects 'status' in the input array; the public
    OpenAI API ignores it. Strip it to keep both working.
    """
    if isinstance(obj, dict):
        obj.pop("status", None)
        for v in obj.values():
            _strip_status(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_status(v)


class TranslationAgent:
    """LLM agent for kernel translation — single-shot or multi-turn with tools.

    Args:
        client:                     Pre-configured OpenAI client (from make_client()).
        model:                      Model name, e.g. "gpt-4o", "o3", "grok-3".
        problem:                    KTBench Problem to translate.
        tools:                      Optional list of Tool instances. When provided,
                                    the agent enters a multi-turn tool-calling loop
                                    and generate() returns a TranslationResult.
                                    When None, single-shot mode returns a str.
        tool_ctx:                   ToolContext required when tools is not None.
        api_kind:                   "responses" (default) or "chat".
        reasoning_effort:           Passed to Responses API reasoning.effort.
        omit_responses_reasoning:   Skip reasoning param (for providers that reject it).
        max_turns:                  Max LLM turns in tool-calling mode (default 10).
        max_api_retries:            API retry budget per turn.
        verbose:                    Print progress.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        problem: Problem,
        tools: list["Tool"] | None = None,
        tool_ctx: "ToolContext | None" = None,
        api_kind: str = "responses",
        reasoning_effort: str | None = None,
        omit_responses_reasoning: bool = False,
        max_turns: int = 10,
        max_api_retries: int = 3,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.problem = problem
        self.tools = tools or []
        self.tool_ctx = tool_ctx
        self.tool_map: dict[str, "Tool"] = {t.name: t for t in self.tools}
        self.api_kind = api_kind
        self.reasoning_effort = reasoning_effort
        self.omit_responses_reasoning = omit_responses_reasoning
        self.max_turns = max(1, max_turns)
        self.max_api_retries = max(1, max_api_retries)
        self.verbose = verbose
        self.last_usage: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self):
        """
        Generate a kernel translation.

        Returns:
            str — candidate ModelNew source (single-shot mode, tools=None)
            TranslationResult — final evaluation result (tool-calling mode)
        """
        if not self.tools:
            return self._single_shot()
        return self._tool_loop()

    # ------------------------------------------------------------------
    # Single-shot (no tools)
    # ------------------------------------------------------------------

    def _single_shot(self) -> str:
        prompt = build_prompt(self.problem)
        last_exc: BaseException | None = None

        for attempt in range(self.max_api_retries):
            try:
                text, usage = self._call_llm(prompt, system=_SYSTEM_PROMPT_SINGLE_SHOT)
                self.last_usage = usage
                code = extract_candidate_from_text(text)
                if code:
                    if self.verbose:
                        print(f"[TranslationAgent] extracted {len(code)} chars")
                    return code
                if self.verbose:
                    print(f"[TranslationAgent] attempt {attempt + 1}: no ModelNew found — retrying")
                last_exc = RuntimeError("No ModelNew code block found in response")
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self.max_api_retries:
                    break
                delay = llm_retry_delay_s(attempt, exc)
                if self.verbose:
                    print(f"[TranslationAgent] attempt {attempt + 1} error: {exc}; retrying in {delay:.1f}s")
                time.sleep(delay)

        raise RuntimeError(f"Translation failed after {self.max_api_retries} attempt(s)") from last_exc

    # ------------------------------------------------------------------
    # Multi-turn tool-calling loop
    # ------------------------------------------------------------------

    def _tool_loop(self):
        """Multi-turn agent loop; returns the TranslationResult from submit_kernel."""
        if self.tool_ctx is None:
            raise ValueError("tool_ctx is required when tools are provided")

        prompt = build_prompt(self.problem)
        gpu_specs_text = self._prefetch_gpu_specs()

        if self.api_kind == "responses":
            return self._tool_loop_responses(prompt, gpu_specs_text)
        return self._tool_loop_chat(prompt, gpu_specs_text)

    def _prefetch_gpu_specs(self) -> str:
        """Call get_gpu_specs once before turn 1 and return its output text."""
        gpu_tool = self.tool_map.get("get_gpu_specs")
        if gpu_tool is None:
            return ""
        try:
            result = gpu_tool.execute(self.tool_ctx)
            return result.output
        except Exception:
            return ""

    def _execute_tool(self, tool_name: str, args: dict) -> "ToolResult":
        from tools.tools import ToolResult as TR
        tool = self.tool_map.get(tool_name)
        if tool is None:
            names = ", ".join(self.tool_map)
            return TR(
                tool_name=tool_name,
                success=False,
                output=f"Unknown tool '{tool_name}'. Available: {names}.",
                metadata={"error": "unknown_tool"},
            )
        try:
            return tool.execute(self.tool_ctx, **args)
        except Exception as exc:
            return TR(
                tool_name=tool_name,
                success=False,
                output=f"{tool_name} FAILED: unexpected error.\n{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                metadata={"error": str(exc)},
            )

    def _tool_loop_responses(self, prompt: str, gpu_specs_text: str = ""):
        """Responses-API tool-calling loop."""
        instructions = _SYSTEM_PROMPT_TOOLS_TMPL.format(max_turns=self.max_turns)
        input_items: list[dict] = [{"role": "user", "content": prompt}]
        if gpu_specs_text:
            input_items.append({"role": "user", "content": f"[Hardware specs pre-fetched]\n{gpu_specs_text}"})
        tool_schemas = [t.to_responses_schema() for t in self.tools]

        for turn_idx in range(self.max_turns):
            if self.verbose:
                print(f"[TranslationAgent] turn {turn_idx + 1}/{self.max_turns}")

            turns_left = self.max_turns - turn_idx
            if turns_left == 1:
                input_items.append({"role": "user", "content": _FINAL_TURN_NUDGE})
            elif turns_left <= max(3, self.max_turns // 10):
                input_items.append({"role": "user", "content": _TURN_WARNING.format(turns_left=turns_left)})

            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": instructions,
                "input": input_items,
                "tools": tool_schemas,
            }
            if not self.omit_responses_reasoning:
                r: dict[str, Any] = {"summary": "auto"}
                if self.reasoning_effort:
                    r["effort"] = self.reasoning_effort
                create_kwargs["reasoning"] = r

            # LLM call with retries
            response = None
            for attempt in range(self.max_api_retries):
                try:
                    response = self.client.responses.create(**create_kwargs)
                    break
                except Exception as exc:
                    if attempt + 1 >= self.max_api_retries:
                        raise
                    delay = llm_retry_delay_s(attempt, exc)
                    if self.verbose:
                        print(f"  LLM error: {exc}; retry in {delay:.1f}s")
                    time.sleep(delay)

            self.last_usage = llm_usage_to_dict(getattr(response, "usage", None))

            # Serialize and echo output items
            response_items: list[dict] = []
            for item in response.output:
                d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
                _strip_status(d)
                response_items.append(d)
            input_items.extend(response_items)

            # Find function calls
            fn_calls = [it for it in response_items if it.get("type") == "function_call"]
            if self.verbose:
                print(f"  {len(fn_calls)} tool call(s)")

            if not fn_calls:
                if turn_idx >= self.max_turns - 1:
                    break
                input_items.append({"role": "user", "content": _NO_TOOL_CALLS_NUDGE})
                continue

            final_result = None
            for fc in fn_calls:
                name = fc.get("name", "")
                raw_args = fc.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    args = {}

                tool_result = self._execute_tool(name, args)
                if self.verbose:
                    print(f"  {name}: {'OK' if tool_result.success else 'FAIL'}")

                input_items.append({
                    "type": "function_call_output",
                    "call_id": fc["call_id"],
                    "output": tool_result.output,
                })

                if name == "submit_kernel":
                    final_result = tool_result
                    break

            if final_result is not None:
                return self._extract_translation_result(final_result)

        # Fell through max_turns with no submit — return None
        return None

    def _tool_loop_chat(self, prompt: str, gpu_specs_text: str = ""):
        """Chat Completions tool-calling loop."""
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT_TOOLS_TMPL.format(max_turns=self.max_turns)},
            {"role": "user", "content": prompt},
        ]
        if gpu_specs_text:
            messages.append({"role": "user", "content": f"[Hardware specs pre-fetched]\n{gpu_specs_text}"})
        tool_schemas = [t.to_chat_schema() for t in self.tools]

        for turn_idx in range(self.max_turns):
            if self.verbose:
                print(f"[TranslationAgent] turn {turn_idx + 1}/{self.max_turns}")

            turns_left = self.max_turns - turn_idx
            if turns_left == 1:
                messages.append({"role": "user", "content": _FINAL_TURN_NUDGE})
            elif turns_left <= max(3, self.max_turns // 10):
                messages.append({"role": "user", "content": _TURN_WARNING.format(turns_left=turns_left)})

            # LLM call with retries
            response = None
            for attempt in range(self.max_api_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="auto",
                    )
                    break
                except Exception as exc:
                    if attempt + 1 >= self.max_api_retries:
                        raise
                    delay = llm_retry_delay_s(attempt, exc)
                    if self.verbose:
                        print(f"  LLM error: {exc}; retry in {delay:.1f}s")
                    time.sleep(delay)

            self.last_usage = llm_usage_to_dict(getattr(response, "usage", None))
            choice = response.choices[0]
            asst = choice.message
            raw_tool_calls = list(asst.tool_calls or [])

            # Echo assistant message
            asst_msg: dict[str, Any] = {"role": "assistant", "content": asst.content or ""}
            if raw_tool_calls:
                asst_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in raw_tool_calls
                ]
            messages.append(asst_msg)

            if self.verbose:
                print(f"  {len(raw_tool_calls)} tool call(s)")

            if not raw_tool_calls:
                if turn_idx >= self.max_turns - 1:
                    break
                messages.append({"role": "user", "content": _NO_TOOL_CALLS_NUDGE})
                continue

            final_result = None
            for tc in raw_tool_calls:
                name = tc.function.name or ""
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else {}
                except json.JSONDecodeError:
                    args = {}

                tool_result = self._execute_tool(name, args)
                if self.verbose:
                    print(f"  {name}: {'OK' if tool_result.success else 'FAIL'}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result.output,
                })

                if name == "submit_kernel":
                    final_result = tool_result
                    break

            if final_result is not None:
                return self._extract_translation_result(final_result)

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, system: str) -> tuple[str, dict | None]:
        """Single LLM call (no tools). Returns (text, usage)."""
        if self.api_kind == "responses":
            return self._call_responses(prompt, system)
        return self._call_chat(prompt, system)

    def _call_responses(self, prompt: str, system: str) -> tuple[str, dict | None]:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": [{"role": "user", "content": prompt}],
        }
        if not self.omit_responses_reasoning:
            r: dict[str, Any] = {"summary": "auto"}
            if self.reasoning_effort:
                r["effort"] = self.reasoning_effort
            create_kwargs["reasoning"] = r

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

    def _call_chat(self, prompt: str, system: str) -> tuple[str, dict | None]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = llm_usage_to_dict(getattr(response, "usage", None))
        return text, usage

    def _extract_translation_result(self, tool_result: "ToolResult"):
        """Pull the TranslationResult out of a submit_kernel ToolResult."""
        from ktbench.eval.harness import TranslationResult
        meta = tool_result.metadata
        if not meta:
            return None
        # metadata is result.summary() dict; reconstruct a minimal TranslationResult
        r = TranslationResult(
            problem_id=meta.get("problem_id", ""),
            tgt_dsl=meta.get("tgt_dsl", ""),
            tgt_hw=meta.get("tgt_hw", ""),
            compiled=meta.get("compiled", False),
            correctness_rate=meta.get("correctness_rate", 0.0),
            stress_pass_rate=meta.get("stress_pass_rate", 0.0),
            sol_score=meta.get("sol_score", 0.0),
            speedup_vs_ref=meta.get("speedup_vs_ref", -1.0),
            final_score=meta.get("final_score", 0.0),
        )
        return r
