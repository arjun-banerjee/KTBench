"""
KTBench <-> ensemble adapter.

Exposes KTBench problems through the ensemble framework's PluginTool / scenario
pattern so the same problem set can be driven by ensemble's multi-actor world
in addition to Inspect AI.

The integration is intentionally thin. KTBench's five Tool classes
(StaticCheckTool, CompileKernelTool, RunCorrectnessTool, GetGpuSpecsTool,
SubmitKernelTool) are wrapped as ensemble PluginTools that close over a
shared ToolContext, and a small world is registered under the name
"ktbench". A scenario in scenarios/translate_problem.py drives the agent
loop.

Why ensemble in addition to Inspect AI:
- ensemble owns the multi-actor pattern. A reviewer actor (the
  code_reviewer persona) can read the trace as the author builds the
  kernel and surface findings the static checker might miss. The
  research question this benchmark is best positioned to answer is
  "does adding a reviewer-actor improve the held-out pass rate", and
  that requires a harness that natively supports more than one actor.
- ensemble's trace format and viewer are richer (per-tool cost
  annotations, progress events, sandboxable dispatch). The KTBench
  harness's anti-hack and SOL machinery is orthogonal and reused
  as-is.

Configuration flows through env vars, read once at World("ktbench")
construction (see _setup() below):

    KTBENCH_PROBLEM_PATH   path to the problem dir; required
    KTBENCH_DEVICE         CUDA device index (default 0)
    KTBENCH_SEED           global RNG seed (default 0)
    KTBENCH_TIMING_TRIALS  perf timing trials (default 20)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ensemble import PluginPredicate, PluginTool, register_world

from ktbench.problem import Problem, load_problem
from ktbench.prompt import build_prompt
from tools.tools import (
    CompileKernelTool,
    GetGpuSpecsTool,
    RunCorrectnessTool,
    StaticCheckTool,
    SubmitKernelTool,
    ToolContext,
)
from integrations import fmt_result


PERSONAS_DIR = _REPO / "personas"


class KTBenchState:
    """Per-world state for the KTBench ensemble integration.

    Holds the loaded Problem, the shared ToolContext the five tools use,
    and a ledger of submission metadata so predicates have an O(1) view
    of the eval results without re-walking the trace.
    """

    def __init__(
        self,
        problem_path: str,
        device: int = 0,
        global_seed: int = 0,
        n_timing_trials: int = 20,
    ) -> None:
        self.problem_path = problem_path
        self.problem: Problem = load_problem(problem_path)
        self.ctx = ToolContext(
            problem=self.problem,
            device=device,
            global_seed=global_seed,
            n_timing_trials=n_timing_trials,
        )
        self.submissions: List[Dict[str, Any]] = []


def _payload(
    *,
    effect: Dict[str, Any],
    diff: Optional[Dict[str, Any]] = None,
) -> str:
    body: Dict[str, Any] = {"effect": effect}
    if diff is not None:
        body["diff"] = diff
    return json.dumps(body)


def _wrap_tool(
    tool_obj: Any,
    state: KTBenchState,
    *,
    capture_submit: bool = False,
) -> PluginTool:
    """Adapt a KTBench Tool to ensemble's PluginTool interface.

    The agent sees only ``effect`` (a structured envelope ensemble surfaces
    as the tool result). ``diff`` carries the structured metadata from
    submit_kernel so the trace viewer can show the per-stage scores and
    so post-hoc analysis tools have a machine-readable summary.
    """

    def fn(args_json: str) -> str:
        args = json.loads(args_json) if args_json else {}
        try:
            result = tool_obj.execute(state.ctx, **args)
        except Exception as e:
            effect = {
                "ok": False,
                "tool": tool_obj.name,
                "summary": f"{tool_obj.name} FAILED: {type(e).__name__}: {e}",
            }
            return _payload(effect=effect)

        summary = fmt_result(result)
        effect: Dict[str, Any] = {
            "ok": bool(result.success),
            "tool": result.tool_name,
            "summary": summary,
        }

        diff: Optional[Dict[str, Any]] = None
        if capture_submit:
            meta = dict(result.metadata or {})
            state.submissions.append(meta)
            diff = {
                "field": "ktbench_submissions",
                "old": None,
                "new": {
                    "tool": result.tool_name,
                    "success": bool(result.success),
                    **{
                        k: meta.get(k)
                        for k in (
                            "compiled",
                            "correctness_rate",
                            "stress_pass_rate",
                            "sol_score",
                            "speedup_vs_ref",
                            "final_score",
                            "bottleneck",
                        )
                        if k in meta
                    },
                },
            }

        return _payload(effect=effect, diff=diff)

    return PluginTool(
        name=tool_obj.name,
        description=tool_obj.description,
        parameters=tool_obj.input_schema,
        fn=fn,
    )


def build_all_tools(state: KTBenchState) -> List[PluginTool]:
    """Return the five KTBench tools wrapped as ensemble PluginTools."""
    return [
        _wrap_tool(StaticCheckTool(), state),
        _wrap_tool(CompileKernelTool(), state),
        _wrap_tool(RunCorrectnessTool(), state),
        _wrap_tool(GetGpuSpecsTool(), state),
        _wrap_tool(SubmitKernelTool(), state, capture_submit=True),
    ]


def build_predicates(state: KTBenchState) -> List[PluginPredicate]:
    """Grader predicates over KTBench eval results.

    Mirrors the inspect_ai scorer: pulls submit_kernel metadata from the
    state ledger and surfaces it as named questions a scenario's grader
    can compose. Predicate names are stable across scenarios so a sweep
    aggregator can find them by name.
    """

    def submit_called(trace_json: str, args_json: str) -> bool:
        return len(state.submissions) > 0

    def submit_passed(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("final_score") or 0) > 0 for r in state.submissions
        )

    def correctness_passed(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("correctness_rate") or 0) >= 1.0 for r in state.submissions
        )

    def stress_passed(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("stress_pass_rate") or 0) >= 0.9 for r in state.submissions
        )

    def utilization_passed(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("sol_score") or 0) > 0 for r in state.submissions
        )

    def static_check_failed(trace_json: str, args_json: str) -> bool:
        trace = json.loads(trace_json) if trace_json else []
        for ev in trace:
            payload = ev.get("payload") or {}
            if payload.get("kind") != "tool_result":
                continue
            if payload.get("name") != "static_check":
                continue
            res = payload.get("result") or {}
            effect = res.get("effect") if isinstance(res, dict) else None
            if isinstance(effect, dict) and effect.get("ok") is False:
                return True
        return False

    return [
        PluginPredicate(name="submit_called", fn=submit_called),
        PluginPredicate(name="submit_passed", fn=submit_passed),
        PluginPredicate(name="correctness_passed", fn=correctness_passed),
        PluginPredicate(name="stress_passed", fn=stress_passed),
        PluginPredicate(name="utilization_passed", fn=utilization_passed),
        PluginPredicate(name="static_check_failed", fn=static_check_failed),
    ]


def prompt_for_env() -> str:
    """Render the user-facing prompt for the problem named by KTBENCH_PROBLEM_PATH.

    A scenario calls this to set up the agent's first user message; the
    prompt is the same one Inspect AI sends, built from the problem's
    source.py + meta.toml + signature, with shapes deliberately omitted.
    """
    problem_path = os.environ.get(
        "KTBENCH_PROBLEM_PATH", "problems/softmax_h200_to_triton"
    )
    return build_prompt(load_problem(problem_path))


def _state_from_env() -> KTBenchState:
    return KTBenchState(
        problem_path=os.environ.get(
            "KTBENCH_PROBLEM_PATH", "problems/softmax_h200_to_triton"
        ),
        device=int(os.environ.get("KTBENCH_DEVICE", "0")),
        global_seed=int(os.environ.get("KTBENCH_SEED", "0")),
        n_timing_trials=int(os.environ.get("KTBENCH_TIMING_TRIALS", "20")),
    )


def _setup() -> tuple[List[PluginTool], List[PluginPredicate]]:
    state = _state_from_env()
    tools = build_all_tools(state)
    predicates = build_predicates(state)
    return tools, predicates


register_world("ktbench", setup=_setup, personas_dir=PERSONAS_DIR)


__all__ = ["KTBenchState", "PERSONAS_DIR", "prompt_for_env"]
