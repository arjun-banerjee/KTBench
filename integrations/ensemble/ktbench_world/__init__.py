"""
KTBench world for ensemble.

Wraps the five KTBench agent tools (static_check, compile_kernel,
run_correctness, get_gpu_specs, submit_kernel) as ensemble PluginTools
and registers the "ktbench" world via register_world. Scenarios
spawn agents against this world; each agent's tool calls dispatch to
the underlying KTBench Tool.execute() through the wrapper.

Three pieces worth knowing:

1.  **Sandboxed CUDA dispatch.** compile_kernel, run_correctness, and
    submit_kernel touch the GPU and so are vulnerable to CUDA-context
    poisoning when a candidate kernel does an illegal memory access or
    runs into a watchdog timeout. These tools are marked
    sandbox=True so each call dispatches to a fresh
    `python -m ensemble.tool_worker` subprocess; a fatal kernel crashes
    only that subprocess, and the next tool call gets a clean Python
    interpreter and a clean CUDA context. static_check and
    get_gpu_specs do not touch the GPU and run in-process.

2.  **Config-driven eval knobs.** Settings under [eval], [stress],
    and [antihack] in configs/eval_defaults.toml flow into the
    ToolContext at world construction. Override the path via
    KTBENCH_EVAL_CONFIG.

3.  **Problem binding via env var.** The world is constructed once
    per scenario; the problem is bound by reading KTBENCH_PROBLEM_PATH
    at tool-call time. This lets the sandboxed subprocess re-derive
    the same problem (env vars cross the subprocess boundary) without
    the parent having to pass it explicitly through every call.

Predicates surface submission state by walking the trace for
ktbench_submissions state-diff events; that survives the parent/
subprocess split.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Python 3.11+ has tomllib in stdlib; older needs tomli.
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover  - python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

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


PACKAGE_DIR = Path(__file__).resolve().parent.parent
PERSONAS_DIR = PACKAGE_DIR / "personas"
DEFAULT_EVAL_CONFIG = _REPO / "configs" / "eval_defaults.toml"

# Tools that touch the GPU. When sandboxed, each call dispatches to a
# fresh `python -m ensemble.tool_worker` subprocess so a CUDA-fatal
# candidate (illegal memory access, watchdog timeout) kills only the
# worker — the parent's CUDA context stays clean.
#
# Temporarily empty: in ensemble@e99b15d the sandbox worker does not
# inherit enough state to re-register the world, so dispatched calls
# fail with "tool 'compile_kernel' not registered by world 'ktbench'".
# Restore once the upstream subprocess propagation is fixed.
_SANDBOXED: set[str] = set()


def _load_eval_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read configs/eval_defaults.toml into a dict.

    Override the path via KTBENCH_EVAL_CONFIG. Missing file returns an
    empty dict so the world still constructs with library defaults.
    """
    target = path or Path(os.environ.get("KTBENCH_EVAL_CONFIG", str(DEFAULT_EVAL_CONFIG)))
    if not target.exists():
        return {}
    try:
        with target.open("rb") as f:
            return tomllib.load(f)
    except Exception as e:  # tomllib raises tomllib.TOMLDecodeError on bad input
        print(f"warning: failed to read {target}: {e}", file=sys.stderr)
        return {}


def _ctx_from_env_and_config(config: Dict[str, Any]) -> Optional[ToolContext]:
    """Build a ToolContext from the environment and the eval config.

    Returns None if KTBENCH_PROBLEM_PATH is unset, which is the
    "world is constructed but no problem is bound yet" state.
    """
    problem_path = os.environ.get("KTBENCH_PROBLEM_PATH")
    if not problem_path:
        return None

    eval_section = config.get("eval", {}) if isinstance(config, dict) else {}
    n_timing = int(os.environ.get("KTBENCH_TIMING_TRIALS",
                                  eval_section.get("num_timing_trials", 20)))
    device = int(os.environ.get("KTBENCH_DEVICE", "0"))
    seed_env = os.environ.get("KTBENCH_SEED")
    global_seed = int(seed_env) if seed_env else None
    verbose = os.environ.get("KTBENCH_VERBOSE", "").lower() in ("1", "true", "yes")

    problem = load_problem(problem_path)
    return ToolContext(
        problem=problem,
        device=device,
        global_seed=global_seed,
        n_timing_trials=n_timing,
        verbose=verbose,
    )


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
    config: Dict[str, Any],
    *,
    capture_submit: bool = False,
) -> PluginTool:
    """Wrap a KTBench Tool as an ensemble PluginTool.

    The wrapper builds a fresh ToolContext per call from env + config so
    sandboxed dispatches (which run in subprocesses that re-import this
    module) reconstruct the same context the parent would have built.
    The agent sees only ``effect``; ``diff`` carries the structured
    metadata for the trace viewer and for predicate evaluation.
    """

    def fn(args_json: str) -> str:
        args = json.loads(args_json) if args_json else {}
        ctx = _ctx_from_env_and_config(config)
        if ctx is None:
            return _payload(effect={
                "ok": False,
                "tool": tool_obj.name,
                "summary": (
                    f"{tool_obj.name} FAILED: KTBENCH_PROBLEM_PATH is unset. "
                    f"The scenario must set this env var before the agent's "
                    f"first tool call."
                ),
            })
        try:
            result = tool_obj.execute(ctx, **args)
        except Exception as e:
            return _payload(effect={
                "ok": False,
                "tool": tool_obj.name,
                "summary": f"{tool_obj.name} FAILED: {type(e).__name__}: {e}",
            })

        summary = fmt_result(result)
        effect: Dict[str, Any] = {
            "ok": bool(result.success),
            "tool": result.tool_name,
            "summary": summary,
        }

        diff: Optional[Dict[str, Any]] = None
        if capture_submit:
            meta = dict(result.metadata or {})
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

    pt = PluginTool(
        name=tool_obj.name,
        description=tool_obj.description,
        parameters=tool_obj.input_schema,
        fn=fn,
    )
    # Sandbox the CUDA-touching tools so a fatal kernel does not poison
    # the parent's CUDA context.
    if tool_obj.name in _SANDBOXED:
        pt.sandbox = True
        pt.sandbox_world = "ktbench"
    return pt


def build_all_tools(config: Dict[str, Any]) -> List[PluginTool]:
    return [
        _wrap_tool(StaticCheckTool(), config),
        _wrap_tool(CompileKernelTool(), config),
        _wrap_tool(RunCorrectnessTool(), config),
        _wrap_tool(GetGpuSpecsTool(), config),
        _wrap_tool(SubmitKernelTool(), config, capture_submit=True),
    ]


def _walk_submissions(trace_json: str) -> List[Dict[str, Any]]:
    """Pull every ktbench_submissions diff out of the trace.

    Sandboxed tools cannot share Python state with the parent, so
    predicates read the trace rather than an in-memory ledger. Each
    submission lands as a state_diff event with field=ktbench_submissions.
    """
    out: List[Dict[str, Any]] = []
    trace = json.loads(trace_json) if trace_json else []
    for ev in trace:
        payload = ev.get("payload") or {}
        if payload.get("kind") != "state_diff":
            continue
        diffs = payload.get("diff") or []
        items = diffs if isinstance(diffs, list) else [diffs]
        for item in items:
            if isinstance(item, dict) and item.get("field") == "ktbench_submissions":
                new = item.get("new")
                if isinstance(new, dict):
                    out.append(new)
    return out


def build_predicates() -> List[PluginPredicate]:
    """Grader predicates over KTBench submission metadata.

    Read from the trace rather than from an in-memory ledger because
    sandboxed tool dispatches do not share Python state with the
    parent. Same six names and same semantics as the in-process
    predicates the earlier integration exposed.
    """

    def submit_called(trace_json: str, args_json: str) -> bool:
        return len(_walk_submissions(trace_json)) > 0

    def submit_passed(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("final_score") or 0) > 0 for r in _walk_submissions(trace_json)
        )

    def correctness_passed(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("correctness_rate") or 0) >= 1.0 for r in _walk_submissions(trace_json)
        )

    def stress_passed(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("stress_pass_rate") or 0) >= 0.9 for r in _walk_submissions(trace_json)
        )

    def sol_above_threshold(trace_json: str, args_json: str) -> bool:
        return any(
            (r.get("sol_score") or 0) >= 0.5 for r in _walk_submissions(trace_json)
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
        PluginPredicate(name="sol_above_threshold", fn=sol_above_threshold),
        PluginPredicate(name="static_check_failed", fn=static_check_failed),
    ]


def prompt_for_path(problem_path: str) -> str:
    """Render the user-facing prompt for a problem at problem_path.

    Used by scenarios to deliver the source kernel + target HW context
    to the agent. The scenario should call this with its baked
    PROBLEM_PATH constant; the world's tool wrappers re-load the same
    problem at tool-call time from KTBENCH_PROBLEM_PATH.
    """
    return build_prompt(load_problem(problem_path))


def _setup() -> tuple[List[PluginTool], List[PluginPredicate]]:
    config = _load_eval_config()
    tools = build_all_tools(config)
    predicates = build_predicates()
    return tools, predicates


register_world("ktbench", setup=_setup, personas_dir=PERSONAS_DIR)


__all__ = ["PACKAGE_DIR", "PERSONAS_DIR", "prompt_for_path"]
