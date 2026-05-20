"""
Example multi-actor KTBench scenario: author + reviewer on
softmax_a100_to_h100.

The author writes the kernel with the full tool kit (static_check,
compile_kernel, run_correctness, get_gpu_specs, submit_kernel). A
separate reviewer agent runs static_check and run_correctness on the
author's submissions and audits the trace. Both share the same
KTBench evaluation state (one ToolContext per world, one Problem)
so the reviewer's claims are grounded in the same test suite the
author's submit_kernel will run against.

The multi-actor pattern is the research arm of the benchmark.
Compare results from this scenario against the single-agent
scenarios/softmax_a100_to_h100.py on the same model + problem cell
to measure whether the reviewer adds correctness uplift.

Run:
    KTBENCH_PROBLEM_PATH=problems/softmax_a100_to_h100 \\
    ensemble run ktbench.judge_softmax_a100_to_h100 \\
        --world ktbench \\
        --manifest integrations/ensemble \\
        --backend anthropic
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO / "integrations" / "ensemble") not in sys.path:
    sys.path.insert(0, str(_REPO / "integrations" / "ensemble"))

import ktbench_world  # noqa: F401  registers the world
from ktbench_world import PERSONAS_DIR, prompt_for_path
from ensemble import scenario
from ensemble.persona import load_persona


PROBLEM_PATH = "problems/softmax_a100_to_h100"
MAX_TURNS    = 40


def _log_agent_prompt(world, agent_id: str, persona_name: str, model: str) -> None:
    try:
        spec = load_persona(PERSONAS_DIR / f"{persona_name}.toml")
        world._native.log_note(
            f"agent_spawn: id={agent_id} persona={spec.name} model={model}\n"
            f"system_prompt:\n{spec.system_prompt}"
        )
    except Exception:
        pass


def _persona_system_prompt(name: str) -> str:
    persona_path = PERSONAS_DIR / f"{name}.toml"
    if not persona_path.exists():
        return ""
    return load_persona(persona_path).system_prompt


@scenario("ktbench.judge_softmax_a100_to_h100", world="ktbench")
async def judge_softmax_a100_to_h100(world):
    os.environ["KTBENCH_PROBLEM_PATH"] = PROBLEM_PATH

    author_model = os.environ.get("KTBENCH_AUTHOR_MODEL",
                                  os.environ.get("KTBENCH_MODEL", "claude-opus-4-7"))
    reviewer_model = os.environ.get("KTBENCH_REVIEWER_MODEL", author_model)
    author_persona = os.environ.get("KTBENCH_AUTHOR_PERSONA", "normal_translation")

    problem_prompt = prompt_for_path(PROBLEM_PATH)

    # Author: full tool kit, baseline (or override) persona.
    author_system = (
        _persona_system_prompt(author_persona) + "\n\n---\n\n" + problem_prompt
    ).lstrip()
    author = world.spawn_agent(
        id="author",
        model=author_model,
        system_prompt=author_system,
        tools=[
            "static_check",
            "compile_kernel",
            "run_correctness",
            "get_gpu_specs",
            "submit_kernel",
        ],
    )
    _log_agent_prompt(world, "author", author_persona, author_model)

    # Reviewer: read-only tool subset, code_reviewer persona.
    # No compile_kernel or submit_kernel — the reviewer audits, the
    # author commits. The reviewer's run_correctness uses the same
    # test suite as the author's eventual submit_kernel.
    reviewer_system = (
        _persona_system_prompt("code_reviewer") + "\n\n---\n\n" + problem_prompt
    ).lstrip()
    reviewer = world.spawn_agent(
        id="reviewer",
        model=reviewer_model,
        system_prompt=reviewer_system,
        tools=["static_check", "run_correctness"],
    )
    _log_agent_prompt(world, "reviewer", "code_reviewer", reviewer_model)

    # Seed the conversation so the author knows a reviewer is watching.
    reviewer.say(
        "author",
        "I'm reviewing this translation. Walk through your approach as you go; "
        "I'll run static_check and run_correctness against your submissions "
        "and flag anything that looks like the kernel is sidestepping the eval.",
    )

    yield world.until(world.turn_count > MAX_TURNS)

    # Same six cells as the single-agent scenario, so per-cell
    # comparisons against the baseline are direct.
    yield {
        "submitted":             1.0 if world.evaluate_predicate("submit_called") else 0.0,
        "submit_passed":         1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "correctness_passed":    1.0 if world.evaluate_predicate("correctness_passed") else 0.0,
        "stress_passed":         1.0 if world.evaluate_predicate("stress_passed") else 0.0,
        "sol_above_threshold":   1.0 if world.evaluate_predicate("sol_above_threshold") else 0.0,
        "static_check_failed":   1.0 if world.evaluate_predicate("static_check_failed") else 0.0,
    }
