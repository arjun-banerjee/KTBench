"""KTBench scenario: swiglu_activation_a100_to_h100.

Translation of a naive A100 SwiGLU CUDA kernel to a Hopper-friendly
H100 CUDA kernel. Follows the same shape as scenarios/softmax_a100_to_h100.py.

Run:
    KTBENCH_PROBLEM_PATH=problems/swiglu_activation_a100_to_h100 \\
    KTBENCH_MODEL=gpt-5.5 \\
    ensemble run ktbench.swiglu_activation_a100_to_h100 \\
        --world ktbench \\
        --manifest integrations/ensemble \\
        --backend openai
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


PROBLEM_PATH = "problems/swiglu_activation_a100_to_h100"
MAX_TURNS    = int(os.environ.get("KTBENCH_MAX_TURNS", "30"))


def _log_agent_prompt(world, agent_id: str, persona_name: str, model: str) -> None:
    try:
        persona_path = PERSONAS_DIR / f"{persona_name}.toml"
        spec = load_persona(persona_path)
        note = (
            f"agent_spawn: id={agent_id} persona={spec.name} model={model}\n"
            f"system_prompt:\n{spec.system_prompt}"
        )
        world._native.log_note(note)
    except Exception:
        pass


@scenario("ktbench.swiglu_activation_a100_to_h100", world="ktbench")
async def swiglu_activation_a100_to_h100(world):
    os.environ["KTBENCH_PROBLEM_PATH"] = PROBLEM_PATH

    model = os.environ.get("KTBENCH_MODEL", "claude-opus-4-7")
    persona_name = os.environ.get("KTBENCH_PERSONA", "normal_translation")

    persona_path = PERSONAS_DIR / f"{persona_name}.toml"
    persona_system = ""
    if persona_path.exists():
        persona_system = load_persona(persona_path).system_prompt
    problem_prompt = prompt_for_path(PROBLEM_PATH)
    full_system = (persona_system + "\n\n---\n\n" + problem_prompt).lstrip()

    world.spawn_agent(
        id="kernel_engineer",
        model=model,
        system_prompt=full_system,
        tools=[
            "static_check",
            "compile_kernel",
            "run_correctness",
            "get_gpu_specs",
            "submit_kernel",
        ],
    )
    _log_agent_prompt(world, "kernel_engineer", persona_name, model)
    world._native.log_note(f"problem_prompt:\n{problem_prompt}")

    harness = world.spawn_user(id="harness", persona="ktbench_harness", model="user-model")
    harness.say(
        "kernel_engineer",
        "Begin. Use the tools to iterate (compile_kernel, run_correctness, get_gpu_specs) "
        "and call submit_kernel once with your final ModelNew.",
    )

    yield world.until(world.turn_count > MAX_TURNS)

    yield {
        "submitted":             1.0 if world.evaluate_predicate("submit_called") else 0.0,
        "submit_passed":         1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "correctness_passed":    1.0 if world.evaluate_predicate("correctness_passed") else 0.0,
        "stress_passed":         1.0 if world.evaluate_predicate("stress_passed") else 0.0,
        "sol_above_threshold":   1.0 if world.evaluate_predicate("sol_above_threshold") else 0.0,
        "static_check_failed":   1.0 if world.evaluate_predicate("static_check_failed") else 0.0,
    }
