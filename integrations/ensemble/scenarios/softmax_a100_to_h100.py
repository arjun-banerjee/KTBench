"""
Example KTBench scenario: softmax_a100_to_h100.

To add a new problem, copy this file, change PROBLEM_PATH and the
@scenario name. Everything else stays the same.

Run with:
    ensemble run ktbench.softmax_a100_to_h100 \
        --world ktbench \
        --manifest integrations/ensemble \
        --backend anthropic          # or openai, vllm
"""

from ensemble import scenario  # type: ignore[import]

PROBLEM_PATH = "problems/softmax_a100_to_h100"
MAX_TURNS    = 30
DEVICE       = 0


@scenario("ktbench.softmax_a100_to_h100", world="ktbench")
async def softmax_a100_to_h100(world):
    # Load the problem — returns the prompt to send to the agent.
    prompt = world.load_problem(PROBLEM_PATH)

    # Harness: scripted sender; no LLM needed.
    harness = world.spawn_user(
        id="harness",
        persona="ktbench_harness",
        model=None,   # scripted, not LLM-driven
    )

    # Agent: the frontier model being evaluated.
    agent = world.spawn_agent(
        id="kernel_engineer",
        model="claude-opus-4-7",          # override via --model on the CLI
        tools=[
            "static_check",
            "compile_kernel",
            "run_correctness",
            "get_gpu_specs",
            "submit_kernel",
        ],
    )

    # Send the problem prompt to kick off the session.
    harness.say("kernel_engineer", prompt)

    # Run until submit_kernel is called or the turn limit is reached.
    yield world.until(
        world.state["submitted"] or world.turn_count >= MAX_TURNS
    )

    # ---- Scores (all in [0, 1]) ----------------------------------------
    s = world.state
    yield {
        "final_score":      s["final_score"],        # primary ranking key
        "correctness":      s["correctness_rate"],
        "stress":           s["stress_pass_rate"],
        "sol":              s["sol_score"],
        "submitted":        1.0 if s["submitted"] else 0.0,
        "turns_used":       world.turn_count / MAX_TURNS,
    }
