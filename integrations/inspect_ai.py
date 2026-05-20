"""
KTBench ↔ Inspect AI adapter.

Exposes KTBench problems as Inspect AI Tasks so you can benchmark any supported
model (OpenAI, Anthropic, Google, xAI) with a single command:

    inspect eval integrations/inspect_ai.py \\
        --model openai/o3 \\
        --model anthropic/claude-opus-4-7 \\
        --model google/gemini-2.5-pro \\
        -T problem_path=problems/softmax_h200_to_triton

Keys are read from env vars automatically by Inspect AI:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY

Python API (used by run_sweep.py):
    from integrations.inspect_ai import ktbench_task, run_ktbench_eval

    results = run_ktbench_eval(
        problem_paths=["problems/softmax_h200_to_triton"],
        models=["openai/o3", "anthropic/claude-opus-4-7"],
        device=0,
    )
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from inspect_ai import Task, eval as _inspect_eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import tool

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
from integrations import extract_final_score, fmt_result


# ---------------------------------------------------------------------------
# Tool adapters — closures over ToolContext, decorated with @tool for Inspect AI
# ---------------------------------------------------------------------------

def _make_tools(ctx: ToolContext) -> list:
    """Return Inspect AI tool callables bound to the given ToolContext."""
    _static    = StaticCheckTool()
    _compile   = CompileKernelTool()
    _correct   = RunCorrectnessTool()
    _specs     = GetGpuSpecsTool()
    _submit    = SubmitKernelTool()

    @tool
    def static_check():
        """Run static analysis to detect reward-hacking patterns before submitting."""
        async def execute(kernel_code: str) -> str:
            return fmt_result(_static.execute(ctx, kernel_code=kernel_code))
        return execute

    @tool
    def compile_kernel():
        """Compile the candidate kernel without running it to catch syntax and DSL errors."""
        async def execute(kernel_code: str) -> str:
            return fmt_result(_compile.execute(ctx, kernel_code=kernel_code))
        return execute

    @tool
    def run_correctness():
        """Run the candidate against the oracle for correctness; no timing."""
        async def execute(kernel_code: str) -> str:
            return fmt_result(_correct.execute(ctx, kernel_code=kernel_code))
        return execute

    @tool
    def get_gpu_specs():
        """Get peak throughput, memory bandwidth, and ridge point for the target GPU."""
        async def execute() -> str:
            return fmt_result(_specs.execute(ctx))
        return execute

    @tool
    def submit_kernel():
        """Submit the final kernel for full evaluation (correctness + timing + SOL score). Call once when done."""
        async def execute(kernel_code: str) -> str:
            return fmt_result(_submit.execute(ctx, kernel_code=kernel_code))
        return execute

    return [
        static_check(),
        compile_kernel(),
        run_correctness(),
        get_gpu_specs(),
        submit_kernel(),
    ]


# ---------------------------------------------------------------------------
# Scorer — reads submit_kernel metadata from the message history
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def ktbench_scorer():
    """Extract final_score from the submit_kernel tool result."""
    async def score(state: TaskState, target) -> Score:
        meta = extract_final_score(state.messages)
        if meta is None:
            return Score(
                value=0.0,
                explanation="submit_kernel was never called",
            )

        final_score  = float(meta.get("final_score", 0.0))
        sol_score    = float(meta.get("sol_score", 0.0))
        stress_rate  = float(meta.get("stress_pass_rate", 0.0))
        correct_rate = float(meta.get("correctness_rate", 0.0))

        return Score(
            value=final_score,
            explanation=(
                f"final={final_score:.4f} | "
                f"correctness={correct_rate:.0%} | "
                f"stress={stress_rate:.0%} | "
                f"sol={sol_score:.4f}"
            ),
        )

    return score


# ---------------------------------------------------------------------------
# Task factory — called by Inspect AI CLI via @task decorator
# ---------------------------------------------------------------------------

@task
def ktbench_eval(
    problem_path: str = "problems/softmax_h200_to_triton",
    device: int = 0,
    seed: int | None = None,
    n_timing: int = 20,
    max_messages: int = 30,
) -> Task:
    """
    Inspect AI Task for one KTBench problem.

    Parameters (pass with -T on the CLI):
        problem_path  — path to the problem directory
        device        — CUDA device index (default 0)
        seed          — random seed (default None = random)
        n_timing      — timing trials for submit_kernel (default 20)
        max_messages  — hard cap on total messages before the session ends (default 30)
    """
    problem = load_problem(problem_path)
    ctx = ToolContext(
        problem=problem,
        device=device,
        global_seed=seed,
        n_timing_trials=n_timing,
    )
    tools = _make_tools(ctx)

    return Task(
        dataset=[Sample(input=build_prompt(problem), id=problem.meta.name)],
        solver=[use_tools(*tools), generate()],
        scorer=ktbench_scorer(),
        max_messages=max_messages,
    )


# ---------------------------------------------------------------------------
# Python API — used by run_sweep.py
# ---------------------------------------------------------------------------

def run_ktbench_eval(
    problem_paths: list[str],
    models: list[str],
    device: int = 0,
    seed: int | None = None,
    n_timing: int = 20,
    max_messages: int = 30,
    **eval_kwargs,
):
    """
    Run KTBench problems against multiple models and return Inspect AI EvalLog list.

    Args:
        problem_paths: list of problem directory paths
        models:        list of model strings, e.g. ["openai/o3", "anthropic/claude-opus-4-7"]
        device:        CUDA device index
        seed:          global random seed
        n_timing:      timing trials per submission
        max_messages:  message cap per session
        **eval_kwargs: forwarded to inspect_ai.eval() (e.g. log_dir, max_tasks)

    Returns:
        list[EvalLog] — one per (problem × model) combination
    """
    tasks = [
        ktbench_eval(
            problem_path=p,
            device=device,
            seed=seed,
            n_timing=n_timing,
            max_messages=max_messages,
        )
        for p in problem_paths
    ]
    return _inspect_eval(tasks, model=models, **eval_kwargs)
