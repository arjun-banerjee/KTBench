"""
KTBench agent tools — compile, check, correct, submit.

Design mirrors KernelBench's agent/tools.py but is self-contained:
no KernelBench imports anywhere; only ktbench.* and stdlib/torch.

Tool catalogue
--------------
  static_check     — static reward-hack pattern detector
  compile_kernel   — try to compile candidate; return compiler output
  run_correctness  — structured + stress correctness suite (no timing)
  get_gpu_specs    — hardware specs for the problem's target HW
  submit_kernel    — full eval: correctness + timing + SOL score

Output convention
-----------------
Every tool output string starts with:
    {tool_name} PASSED: ...
    {tool_name} FAILED: ...
"""
from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from ktbench.eval.antihack import static_check as _antihack_static_check
from ktbench.eval.correctness import run_structured_cases, run_stress_suite
from ktbench.eval.harness import TranslationResult, eval_translation
from ktbench.problem import Problem
from ktbench.registry.dsl import DSLLoadError, load_model_class
from ktbench.registry.hardware import get_hw_spec


# ---------------------------------------------------------------------------
# ToolContext — shared per-run state
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Shared per-agent-run state injected into every tool call."""
    problem: Problem          # KTBench Problem to evaluate against
    device: int = 0           # CUDA device index
    global_seed: int | None = None
    n_timing_trials: int = 20
    verbose: bool = False


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str           # LLM-readable string
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------

class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for tool inputs

    @abstractmethod
    def execute(self, ctx: ToolContext, **kwargs) -> ToolResult: ...

    def to_responses_schema(self) -> dict[str, Any]:
        """OpenAI Responses API tool schema (flat, used with client.responses.create)."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_chat_schema(self) -> dict[str, Any]:
        """OpenAI Chat Completions tool schema (nested, used with chat.completions.create)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# ---------------------------------------------------------------------------
# Shared schema fragment
# ---------------------------------------------------------------------------

_KERNEL_CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kernel_code": {
            "type": "string",
            "description": (
                "Complete Python source implementing ModelNew. Must be a full "
                "Python module — not raw CUDA/HIP/Triton source alone."
            ),
        }
    },
    "required": ["kernel_code"],
}


# ---------------------------------------------------------------------------
# 1. StaticCheckTool
# ---------------------------------------------------------------------------

class StaticCheckTool(Tool):
    name = "static_check"
    description = (
        "Run a static analysis pass that detects reward-hacking patterns: "
        "process manipulation (os.kill, subprocess), stack introspection "
        "(inspect.stack, sys._getframe), oracle file snooping, and ktbench "
        "eval imports. Run this before submit_kernel to catch violations early."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        result = _antihack_static_check(kernel_code, ctx.problem.meta.tgt_dsl)

        if result.passed and not result.warnings:
            output = "static_check PASSED: no violations detected."
        elif result.passed:
            lines = ["static_check PASSED (with warnings):"]
            for w in result.warnings:
                lines.append(f"  WARNING: {w}")
            output = "\n".join(lines)
        else:
            lines = ["static_check FAILED: strict violations found."]
            for e in result.errors:
                lines.append(f"  ERROR: {e}")
            for w in result.warnings:
                lines.append(f"  WARNING: {w}")
            output = "\n".join(lines)

        return ToolResult(
            tool_name=self.name,
            success=result.passed,
            output=output,
            metadata={"passed": result.passed, "errors": result.errors, "warnings": result.warnings},
        )


# ---------------------------------------------------------------------------
# 2. CompileKernelTool
# ---------------------------------------------------------------------------

class CompileKernelTool(Tool):
    name = "compile_kernel"
    description = (
        "Compile the candidate kernel without running it. Use this first to "
        "catch syntax, linker, and DSL-compilation errors before spending GPU "
        "time on correctness. Returns compiler output on failure."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        try:
            cls = load_model_class(kernel_code, ctx.problem.meta.tgt_dsl)
            # Instantiate once to surface __init__ errors
            model = cls()
            device = torch.device(f"cuda:{ctx.device}")
            if hasattr(model, "to"):
                model.to(device)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="compile_kernel PASSED: ModelNew compiled and instantiated successfully.",
                metadata={"compiled": True},
            )
        except DSLLoadError as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"compile_kernel FAILED: {e}",
                metadata={"compiled": False, "error": str(e)},
            )
        except Exception as e:
            detail = traceback.format_exc()
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"compile_kernel FAILED: {type(e).__name__}: {e}\n{detail}",
                metadata={"compiled": False, "error": str(e)},
            )


# ---------------------------------------------------------------------------
# 3. RunCorrectnessTool
# ---------------------------------------------------------------------------

class RunCorrectnessTool(Tool):
    name = "run_correctness"
    description = (
        "Run the candidate kernel against the oracle for correctness only — "
        "no timing. Tests structured cases (fixed shapes, random values) then "
        "stress trials (random shapes AND values). Use after compile_kernel "
        "succeeds. Returns pass/fail per case and the stress pass rate."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        device = torch.device(f"cuda:{ctx.device}")
        seed = ctx.global_seed or 0

        # Load candidate
        try:
            cand_cls = load_model_class(kernel_code, ctx.problem.meta.tgt_dsl)
        except DSLLoadError as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"run_correctness FAILED: compilation error.\n{e}",
                metadata={"compiled": False},
            )

        # Load reference (used as correctness baseline)
        try:
            oracle_cls = load_model_class(ctx.problem.reference_tgt_src, ctx.problem.effective_ref_dsl)
        except DSLLoadError as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"run_correctness FAILED: reference_tgt load error.\n{e}",
                metadata={"error": "reference_load_failed"},
            )

        # Structured cases
        try:
            cr = run_structured_cases(cand_cls, oracle_cls, ctx.problem, device, global_seed=seed)
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"run_correctness FAILED: structured cases raised an exception.\n{type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        if not cr.all_passed:
            failed = [c for c in cr.cases if not c.passed]
            lines = [
                f"run_correctness FAILED: {len(failed)}/{len(cr.cases)} structured cases failed."
            ]
            for c in failed[:3]:  # show at most 3 failures
                msg = c.error or str(c.details)
                lines.append(f"  [{c.case_id}] {msg}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="\n".join(lines),
                metadata={"compiled": True, "correctness": False, "detail": cr.summary()},
            )

        # Stress suite
        try:
            sr = run_stress_suite(cand_cls, oracle_cls, ctx.problem, device, global_seed=seed)
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"run_correctness FAILED: stress suite raised an exception.\n{type(e).__name__}: {e}",
                metadata={"compiled": True, "error": str(e)},
            )

        threshold = ctx.problem.test_suite.stress.pass_threshold
        lines = [
            f"run_correctness PASSED: all {len(cr.cases)} structured cases passed.",
            f"Stress: {sr.trials_passed}/{sr.trials_total} trials passed "
            f"({sr.pass_rate:.0%}, threshold {threshold:.0%}).",
        ]
        if sr.errors:
            lines.append(f"Stress errors (first {min(3, len(sr.errors))}): "
                         + "; ".join(sr.errors[:3]))

        stress_ok = sr.pass_rate >= threshold
        return ToolResult(
            tool_name=self.name,
            success=stress_ok,
            output="\n".join(lines),
            metadata={
                "compiled": True,
                "correctness": True,
                "stress_pass_rate": round(sr.pass_rate, 4),
                "stress_ok": stress_ok,
                "detail": {**cr.summary(), **sr.summary()},
            },
        )


# ---------------------------------------------------------------------------
# 4. GetGpuSpecsTool
# ---------------------------------------------------------------------------

class GetGpuSpecsTool(Tool):
    name = "get_gpu_specs"
    description = (
        "Return hardware specs for this problem's target GPU: peak TFLOPS, "
        "memory bandwidth, SM count, L2 size, and shared memory per SM. "
        "Use this once at the start to calibrate optimization targets. "
        "No arguments needed."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, ctx: ToolContext, **_) -> ToolResult:
        hw = get_hw_spec(ctx.problem.meta.tgt_hw)
        device = torch.device(f"cuda:{ctx.device}")

        try:
            device_name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            total_mem_gb = props.total_memory / 1024 ** 3
            sm_count = props.multi_processor_count
            shared_mem_per_sm_kb = props.max_shared_memory_per_block // 1024
        except Exception:
            device_name = hw.key
            total_mem_gb = sm_count = shared_mem_per_sm_kb = None

        lines = [
            f"GPU specs for {device_name} (problem target: {hw.key}):",
            f"  Peak FP16 throughput : {hw.fp16_tflops} TFLOP/s",
            f"  DRAM bandwidth       : {hw.dram_bw_gbs} GB/s",
        ]
        if total_mem_gb is not None:
            lines.append(f"  Total memory         : {total_mem_gb:.1f} GB")
        if sm_count is not None:
            lines.append(f"  SM count             : {sm_count}")
        if shared_mem_per_sm_kb is not None:
            lines.append(f"  Shared mem / SM      : {shared_mem_per_sm_kb} KB")

        # Compute roofline ridge point
        ridge = (hw.fp16_tflops * 1e12) / (hw.dram_bw_gbs * 1e9)
        lines.append(f"  Ridge point          : {ridge:.1f} FLOP/byte")
        lines.append(
            "  (Kernels with arithmetic intensity > ridge point are compute-bound; "
            "< ridge point are memory-bound.)"
        )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
            metadata={
                "hw_key": hw.key,
                "fp16_tflops": hw.fp16_tflops,
                "dram_bw_gbs": hw.dram_bw_gbs,
                "ridge_point_flop_per_byte": round(ridge, 2),
            },
        )


# ---------------------------------------------------------------------------
# 5. SubmitKernelTool
# ---------------------------------------------------------------------------

class SubmitKernelTool(Tool):
    """
    Full evaluation: static check → compile → correctness → stress → timing → SOL score.

    Anti-reward-hacking: speedup_vs_ref is NOT shown (SOL score is shown instead —
    it is bounded by hardware physics and cannot be inflated by a slow reference).
    """
    name = "submit_kernel"
    description = (
        "Submit the final kernel for full evaluation: static check, correctness, "
        "stress testing, and performance measurement. Call this when you are "
        "confident the kernel is correct and optimized. Returns the final score "
        "(correctness × stress × SOL). Only call this once — it ends the session."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        try:
            result: TranslationResult = eval_translation(
                kernel_code,
                ctx.problem,
                device=ctx.device,
                global_seed=ctx.global_seed,
                n_timing_trials=ctx.n_timing_trials,
                verbose=ctx.verbose,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"submit_kernel FAILED: unexpected error during evaluation.\n{type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        # Static block
        if result.static_check and not result.static_check.passed:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="submit_kernel FAILED: static analysis blocked the submission.\n"
                       + "\n".join(f"  ERROR: {e}" for e in result.static_check.errors),
                metadata=result.summary(),
            )

        # Compile fail
        if not result.compiled:
            err = result.compile_error or "unknown compilation error"
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"submit_kernel FAILED: kernel did not compile.\n{err}",
                metadata=result.summary(),
            )

        # Correctness fail
        if result.correctness_rate < 1.0:
            detail = ""
            if result.correctness:
                failed = [c for c in result.correctness.cases if not c.passed]
                detail = "\n".join(
                    f"  [{c.case_id}] {c.error or c.details}" for c in failed[:3]
                )
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    f"submit_kernel FAILED: correctness check failed "
                    f"({result.correctness_rate:.0%} cases passed).\n{detail}"
                ),
                metadata=result.summary(),
            )

        # Utilization gate
        if result.utilization_blocked:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    f"submit_kernel FAILED: utilization gate — kernel appears to be a no-op "
                    f"({result.utilization_reason})."
                ),
                metadata=result.summary(),
            )

        # Performance measurement failure (sol_score < 0 means timing did not complete)
        if result.sol_score < 0:
            err = result.compile_error or "performance measurement failed"
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"submit_kernel FAILED: performance measurement did not complete.\n{err}",
                metadata=result.summary(),
            )

        # Success — report score and SOL; do NOT reveal speedup_vs_ref
        stress_rate = result.stress_pass_rate
        lines = [
            "submit_kernel PASSED.",
            f"  Correctness    : {result.correctness_rate:.0%}",
            f"  Stress pass    : {stress_rate:.0%} "
            f"({result.stress.trials_passed}/{result.stress.trials_total} trials)"
            if result.stress else "",
            f"  SOL score      : {result.sol_score:.4f}",
            f"  Final score    : {result.final_score:.4f}  "
            f"(correctness × stress × SOL)",
        ]
        if result.excessive_speedup_flagged:
            lines.append(f"  NOTE: {result.excessive_speedup_reason}")

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(l for l in lines if l),
            metadata=result.summary(),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TOOLS: list[Tool] = [
    StaticCheckTool(),
    CompileKernelTool(),
    RunCorrectnessTool(),
    GetGpuSpecsTool(),
    SubmitKernelTool(),
]

TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}

# Tools excluded from the default set (none currently; profile_kernel would go here
# if/when it is added, since it requires ncu in PATH).
_OPT_IN_TOOLS: set[str] = set()


def get_tools(tool_names: list[str] | None = None) -> list[Tool]:
    """
    Return Tool instances for the given names.

    tool_names=None  → all tools except opt-in tools (currently none excluded).
    submit_kernel is always included so the agent has a way to record a final result.
    """
    if tool_names is None:
        return [t for t in ALL_TOOLS if t.name not in _OPT_IN_TOOLS]
    wanted = set(tool_names) | {"submit_kernel"}
    return [t for t in ALL_TOOLS if t.name in wanted]
