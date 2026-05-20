"""
Example KTBench scenario: softmax_a100_to_h100.

To add a new problem, copy this file, change PROBLEM_PATH and the
@scenario name. Everything else stays the same; the world's
configuration (num_timing_trials, device, sandbox) flows from env +
configs/eval_defaults.toml automatically.

Run:
    KTBENCH_PROBLEM_PATH=problems/softmax_a100_to_h100 \\
    ensemble run ktbench.softmax_a100_to_h100 \\
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
MAX_TURNS    = 30


def _log_agent_prompt(world, agent_id: str, persona_name: str, model: str) -> None:
    """Write the resolved persona's system prompt to the trace.

    Ensemble does not emit a spawn event with the system prompt today,
    so the trace viewer cannot show it without this note. Best-effort:
    a missing persona file leaves the trace without the spawn note
    rather than failing the run.
    """
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


@scenario("ktbench.softmax_a100_to_h100", world="ktbench")
async def softmax_a100_to_h100(world):
    # Bind the problem for the world's tool wrappers. The wrappers read
    # KTBENCH_PROBLEM_PATH at tool-call time and re-derive the same
    # ToolContext inside the sandbox, so this env var has to be set
    # before the agent's first tool call. Setting it here in the
    # scenario means a CLI invocation does not have to remember to.
    os.environ["KTBENCH_PROBLEM_PATH"] = PROBLEM_PATH

    model = os.environ.get("KTBENCH_MODEL", "claude-opus-4-7")
    persona_name = os.environ.get("KTBENCH_PERSONA", "normal_translation")

    # Persona's baseline framing + the problem-specific prompt go in
    # together as the agent's system context. spawn_agent takes a
    # single system_prompt; the persona text covers role + scoring,
    # the prompt below adds the source kernel and target HW.
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

    yield world.until(world.turn_count > MAX_TURNS)

    # Six grader cells, all in [0, 1]. The world's predicates pull
    # submission metadata from ktbench_submissions state-diff events
    # on the trace, which survives the parent/subprocess split caused
    # by sandbox dispatch.
    yield {
        "submitted":             1.0 if world.evaluate_predicate("submit_called") else 0.0,
        "submit_passed":         1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "correctness_passed":    1.0 if world.evaluate_predicate("correctness_passed") else 0.0,
        "stress_passed":         1.0 if world.evaluate_predicate("stress_passed") else 0.0,
        "sol_above_threshold":   1.0 if world.evaluate_predicate("sol_above_threshold") else 0.0,
        "static_check_failed":   1.0 if world.evaluate_predicate("static_check_failed") else 0.0,
    }
