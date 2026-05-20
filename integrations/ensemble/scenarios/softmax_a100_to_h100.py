"""KTBench scenario: softmax_a100_to_h100.

To add a new problem, copy this file, change ``PROBLEM_PATH`` and
the ``@scenario`` name. The world's tool wrappers pick up the
problem from ``KTBENCH_PROBLEM_PATH``; the rest stays the same.

Run::

    KTBENCH_PROBLEM_PATH=problems/softmax_a100_to_h100 \\
    KTBENCH_MODEL=gpt-5.5 \\
    ensemble run ktbench.softmax_a100_to_h100 \\
        --world ktbench --backend openai
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
MAX_TURNS    = int(os.environ.get("KTBENCH_MAX_TURNS", "80"))


def _persona_system_prompt(name: str) -> str:
    persona_path = PERSONAS_DIR / f"{name}.toml"
    if not persona_path.exists():
        return ""
    return load_persona(persona_path).system_prompt


@scenario("ktbench.softmax_a100_to_h100", world="ktbench")
async def softmax_a100_to_h100(world):
    os.environ["KTBENCH_PROBLEM_PATH"] = PROBLEM_PATH

    model        = os.environ.get("KTBENCH_MODEL", "claude-opus-4-7")
    persona_name = os.environ.get("KTBENCH_PERSONA", "normal_translation")
    effort       = os.environ.get("KTBENCH_REASONING_EFFORT")

    problem_prompt = prompt_for_path(PROBLEM_PATH)
    system_prompt  = (_persona_system_prompt(persona_name) + "\n\n---\n\n" + problem_prompt).lstrip()

    world.log_event("problem_prompt", {"text": problem_prompt})

    params = {"reasoning_effort": effort} if effort else None
    world.spawn_agent(
        id="kernel_engineer",
        model=model,
        system_prompt=system_prompt,
        tools=world.tool_names(),
        params=params,
    )

    # Scheduler quiesces without an inbound message at startup. A
    # harness user delivers a one-shot kickoff and then stays silent.
    # The kickoff intentionally does NOT prescribe a workflow - the
    # model discovers its own path.
    harness = world.spawn_user(id="harness", persona="ktbench_harness", model="user-model")
    harness.say(
        "kernel_engineer",
        "Begin. Your score is 0 unless submit_kernel is called - that is "
        "the only event that counts.",
    )

    yield world.until_predicate("submit_called") | (world.turn_count > MAX_TURNS)

    yield {
        "submitted":             1.0 if world.evaluate_predicate("submit_called") else 0.0,
        "submit_passed":         1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "correctness_passed":    1.0 if world.evaluate_predicate("correctness_passed") else 0.0,
        "stress_passed":         1.0 if world.evaluate_predicate("stress_passed") else 0.0,
        "sol_above_threshold":   1.0 if world.evaluate_predicate("sol_above_threshold") else 0.0,
        "static_check_failed":   1.0 if world.evaluate_predicate("static_check_failed") else 0.0,
    }
