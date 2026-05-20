"""
KTBench world for Ensemble.

Wraps the five KTBench agent tools as Ensemble-callable actions and tracks
submission state so grader predicates can read it.

World state keys (readable in graders):
    submitted          bool   — submit_kernel was called
    correctness_passed bool   — correctness_rate == 1.0
    stress_passed      bool   — stress_pass_rate >= threshold
    sol_above_threshold bool  — sol_score >= 0.5
    final_score        float  — correctness × stress × SOL
    correctness_rate   float
    stress_pass_rate   float
    sol_score          float
    problem_id         str
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ensemble import World, tool  # type: ignore[import]

from tools.tools import (
    CompileKernelTool,
    GetGpuSpecsTool,
    RunCorrectnessTool,
    StaticCheckTool,
    SubmitKernelTool,
    ToolContext,
    get_tools,
)
from integrations import fmt_result


_SOL_THRESHOLD = 0.5   # minimum SOL to count as "passing" for the grader


class KTBenchWorld(World):
    """
    One KTBench evaluation environment.

    Call load_problem(problem_path) in the scenario before spawning actors.
    """

    name = "ktbench"

    def __init__(self, device: int = 0, **kw):
        super().__init__(**kw)
        self.device = device
        self._ctx: ToolContext | None = None
        self._tool_map = {}
        # Grader-readable state
        self.state = {
            "submitted":           False,
            "correctness_passed":  False,
            "stress_passed":       False,
            "sol_above_threshold": False,
            "final_score":         0.0,
            "correctness_rate":    0.0,
            "stress_pass_rate":    0.0,
            "sol_score":           0.0,
            "problem_id":          "",
        }

    def load_problem(self, problem_path: str) -> str:
        """
        Load a KTBench problem and return the initial prompt to send to the agent.
        Must be called before spawning actors.
        """
        from ktbench.problem import load_problem
        from ktbench.prompt import build_prompt

        problem = load_problem(problem_path)
        self._ctx = ToolContext(
            problem=problem,
            device=self.device,
        )
        self._tool_map = {t.name: t for t in get_tools()}
        self.state["problem_id"] = problem.meta.problem_id
        return build_prompt(problem)

    # ------------------------------------------------------------------ #
    # Tool implementations — called by Ensemble's tool runtime            #
    # ------------------------------------------------------------------ #

    @tool
    def static_check(self, kernel_code: str) -> str:
        assert self._ctx, "load_problem() must be called before tools"
        return fmt_result(self._tool_map["static_check"].execute(self._ctx, kernel_code=kernel_code))

    @tool
    def compile_kernel(self, kernel_code: str) -> str:
        assert self._ctx, "load_problem() must be called before tools"
        return fmt_result(self._tool_map["compile_kernel"].execute(self._ctx, kernel_code=kernel_code))

    @tool
    def run_correctness(self, kernel_code: str) -> str:
        assert self._ctx, "load_problem() must be called before tools"
        return fmt_result(self._tool_map["run_correctness"].execute(self._ctx, kernel_code=kernel_code))

    @tool
    def get_gpu_specs(self) -> str:
        assert self._ctx, "load_problem() must be called before tools"
        return fmt_result(self._tool_map["get_gpu_specs"].execute(self._ctx))

    @tool
    def submit_kernel(self, kernel_code: str) -> str:
        assert self._ctx, "load_problem() must be called before tools"
        result = self._tool_map["submit_kernel"].execute(self._ctx, kernel_code=kernel_code)

        # Update grader-readable state from submission metadata
        meta = result.metadata
        cr   = float(meta.get("correctness_rate",  0.0))
        sr   = float(meta.get("stress_pass_rate",  0.0))
        sol  = float(meta.get("sol_score",          0.0))
        fs   = float(meta.get("final_score",        0.0))

        self.state.update({
            "submitted":           True,
            "correctness_passed":  cr == 1.0,
            "stress_passed":       sr >= 0.9,
            "sol_above_threshold": sol >= _SOL_THRESHOLD,
            "final_score":         fs,
            "correctness_rate":    cr,
            "stress_pass_rate":    sr,
            "sol_score":           sol,
        })

        return fmt_result(result)
