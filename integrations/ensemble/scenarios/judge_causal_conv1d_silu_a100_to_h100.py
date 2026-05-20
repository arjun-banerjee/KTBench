"""KTBench multi-actor scenario: judge_causal_conv1d_silu_a100_to_h100

Author + reviewer on the same problem. Compared cell-for-cell
against the single-agent scenario for the same model + problem,
this is the headline measurement of whether the reviewer actor
moves the score.

Run::

    KTBENCH_PROBLEM_PATH=problems/causal_conv1d_silu_a100_to_h100 \
    KTBENCH_AUTHOR_MODEL=gpt-5.5 \
    ensemble run ktbench.judge_causal_conv1d_silu_a100_to_h100 --world ktbench --backend openai
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


PROBLEM_PATH = "problems/causal_conv1d_silu_a100_to_h100"
MAX_TURNS    = int(os.environ.get("KTBENCH_MAX_TURNS", "80"))


def _persona_system_prompt(name: str) -> str:
    persona_path = PERSONAS_DIR / f"{name}.toml"
    if not persona_path.exists():
        return ""
    return load_persona(persona_path).system_prompt


@scenario("ktbench.judge_causal_conv1d_silu_a100_to_h100", world="ktbench")
async def judge_causal_conv1d_silu_a100_to_h100(world):
    os.environ["KTBENCH_PROBLEM_PATH"] = PROBLEM_PATH

    author_model   = os.environ.get("KTBENCH_AUTHOR_MODEL",
                                    os.environ.get("KTBENCH_MODEL", "gpt-5.5"))
    reviewer_model = os.environ.get("KTBENCH_REVIEWER_MODEL", author_model)
    author_persona = os.environ.get("KTBENCH_AUTHOR_PERSONA", "normal_translation")
    effort         = os.environ.get("KTBENCH_REASONING_EFFORT")

    problem_prompt = prompt_for_path(PROBLEM_PATH)
    world.log_event("problem_prompt", {"text": problem_prompt})

    params = {"reasoning_effort": effort} if effort else None

    world.spawn_agent(
        id="author",
        model=author_model,
        system_prompt=(_persona_system_prompt(author_persona) + "\n\n---\n\n" + problem_prompt).lstrip(),
        tools=["static_check", "compile_kernel", "run_correctness",
               "get_gpu_specs", "submit_kernel"],
        params=params,
    )
    reviewer = world.spawn_agent(
        id="reviewer",
        model=reviewer_model,
        system_prompt=(_persona_system_prompt("code_reviewer") + "\n\n---\n\n" + problem_prompt).lstrip(),
        tools=["static_check", "run_correctness"],
        params=params,
    )

    reviewer.say(
        "author",
        "I'm reviewing this translation. Walk through your approach as you go; "
        "I'll run static_check and run_correctness against your submissions and "
        "flag anything that looks like the kernel is sidestepping the eval.",
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
