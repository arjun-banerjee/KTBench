"""One-agent KTBench translation rollout, ensemble-driven.

The agent gets the problem prompt as the first user message, the persona's
system prompt as its system context, and the five KTBench tools wrapped
as PluginTools. It iterates until submit_kernel returns or the turn
budget is exhausted. Grader cells come from KTBench's eval pipeline
(correctness, stress, SOL) plus participation signals.

Configuration via env vars; defaults match the inspect_ai integration
so the same problem path / device / seed flags work from either harness.

    KTBENCH_PROBLEM_PATH   problem directory (required)
    KTBENCH_DEVICE         CUDA device index (default 0)
    KTBENCH_SEED           global RNG seed (default 0)
    KTBENCH_TIMING_TRIALS  perf timing trials (default 20)
    KTBENCH_MODEL          LLM identifier (default claude-sonnet-4-5)
    KTBENCH_PERSONA        persona TOML name (default normal_translation)
    KTBENCH_MAX_TURNS      turn budget for the until predicate (default 30)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import integrations.ensemble  # noqa: F401  registers the ktbench world
from integrations.ensemble import PERSONAS_DIR, prompt_for_env
from ensemble import scenario
from ensemble.persona import load_persona


def _log_agent_prompt(world, agent_id: str, persona_name: str, model: str) -> None:
    """Write the resolved persona's system prompt to the trace.

    Mirrors the popcorn_world helper. Ensemble does not currently emit
    a spawn event with the agent's system prompt, so the trace viewer
    cannot show it without this note. Best-effort: a misconfigured
    persona path leaves the trace without the spawn note rather than
    failing the run.
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


@scenario("ktbench.translate_problem", world="ktbench")
async def translate_problem(world):
    model = os.environ.get("KTBENCH_MODEL", "claude-sonnet-4-5")
    persona_name = os.environ.get("KTBENCH_PERSONA", "normal_translation")
    max_turns = int(os.environ.get("KTBENCH_MAX_TURNS", "30"))

    # The world's setup already loaded the problem. The scenario reads
    # the prompt back so it can hand it to the agent as a user message;
    # this duplicates the problem load but the cost is just a few TOML
    # reads.
    problem_prompt = prompt_for_env()

    # Pull the persona's system prompt so spawn_agent can layer it on
    # top of the model's default behaviour. The persona's prompt is the
    # baseline framing (translation task, correctness contract, scoring
    # objective); the user message below carries the per-problem
    # specifics (source kernel, target HW, signature).
    persona_path = PERSONAS_DIR / f"{persona_name}.toml"
    persona_system_prompt = None
    if persona_path.exists():
        persona_system_prompt = load_persona(persona_path).system_prompt

    # Combine the persona's baseline framing with the problem-specific
    # prompt. Ensemble's spawn_agent takes a single system_prompt; the
    # persona's text covers the role / contract / scoring, and the
    # problem prompt below it adds the source kernel and target HW.
    # Splitting the two into separate messages would require a seeded
    # user message, which ensemble does not expose at the scenario
    # layer today.
    full_system = (
        (persona_system_prompt or "") + "\n\n---\n\n" + problem_prompt
    )
    agent = world.spawn_agent(
        id="translator",
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
    _log_agent_prompt(world, "translator", persona_name, model)
    world._native.log_note(f"problem_prompt:\n{problem_prompt}")

    yield world.until(world.turn_count > max_turns)

    yield {
        "submitted": 1.0 if world.evaluate_predicate("submit_called") else 0.0,
        "submit_passed": 1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "correctness_passed": 1.0 if world.evaluate_predicate("correctness_passed") else 0.0,
        "stress_passed": 1.0 if world.evaluate_predicate("stress_passed") else 0.0,
        "utilization_passed": 1.0 if world.evaluate_predicate("utilization_passed") else 0.0,
        "static_check_failed": 1.0 if world.evaluate_predicate("static_check_failed") else 0.0,
    }
