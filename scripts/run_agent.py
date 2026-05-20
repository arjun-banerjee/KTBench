"""
run_agent.py — Use an LLM to translate a KTBench problem kernel.

Single-shot mode (default):
  The agent builds a prompt, calls the LLM once, extracts ModelNew, and
  optionally evaluates the result.

Multi-turn tool-calling mode (--multi-turn):
  The agent enters a loop where the model can call compile_kernel,
  run_correctness, get_gpu_specs, static_check, and submit_kernel.
  The session ends when submit_kernel is called or max_turns is reached.

Provider presets
----------------
  openai   OPENAI_API_KEY         Responses API (supports o3/o4-mini reasoning)
  azure    AZURE_OPENAI_API_KEY   Responses API; --base-url required
  grok     XAI_API_KEY            Chat Completions (https://api.x.ai/v1)

Multi-turn example (OpenAI):
  python scripts/run_agent.py \\
    --problem problems/softmax_h200_to_triton \\
    --model o3 --reasoning-effort medium \\
    --multi-turn --max-turns 8 --device 0 --verbose

Single-shot + eval example:
  python scripts/run_agent.py \\
    --problem problems/softmax_h200_to_triton \\
    --model gpt-4.1 --provider azure \\
    --base-url https://my-resource.cognitiveservices.azure.com/openai/v1/ \\
    --out candidate.py --eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure both src/ (ktbench package) and repo root (tools package) are importable
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from ktbench.llm.client import make_client, PROVIDER_PRESETS
from ktbench.llm.agent import TranslationAgent
from ktbench.problem import load_problem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a KTBench kernel translation via an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Problem
    parser.add_argument("--problem", required=True,
                        help="Path to problem directory")

    # Model / provider
    parser.add_argument("--model", required=True,
                        help="Model name, e.g. gpt-4o, o3, gpt-4.1, grok-3")
    parser.add_argument("--provider", default="openai",
                        choices=list(PROVIDER_PRESETS.keys()) + ["custom"],
                        help="Provider preset (default: openai)")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None,
                        help="API base URL (required for azure)")
    parser.add_argument("--api-kind", default=None, choices=["responses", "chat"])
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--omit-responses-reasoning", action="store_true")

    # Mode
    parser.add_argument("--multi-turn", action="store_true",
                        help="Enable multi-turn tool-calling mode")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max LLM turns in multi-turn mode (default: 10)")
    parser.add_argument("--tools", default=None,
                        help="Comma-separated tool names for multi-turn mode "
                             "(default: all). E.g. compile_kernel,run_correctness,submit_kernel")
    parser.add_argument("--retries", type=int, default=3)

    # Output
    parser.add_argument("--out", default=None,
                        help="Write candidate source to this file")

    # Evaluation (single-shot mode only)
    parser.add_argument("--eval", action="store_true",
                        help="Run the full evaluator after single-shot generation")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n-timing", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json-out", default=None)

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve provider                                                     #
    # ------------------------------------------------------------------ #
    preset = PROVIDER_PRESETS.get(args.provider, {})
    api_key_env           = args.api_key_env           or preset.get("api_key_env", "OPENAI_API_KEY")
    base_url              = args.base_url               or preset.get("base_url")
    api_kind              = args.api_kind               or preset.get("api_kind", "responses")
    omit_responses_reason = args.omit_responses_reasoning or preset.get("omit_responses_reasoning", False)

    if args.provider == "azure" and not base_url:
        parser.error("--provider azure requires --base-url")

    try:
        client = make_client(api_key_env=api_key_env, base_url=base_url)
    except RuntimeError as e:
        sys.exit(f"[run_agent] {e}")

    problem = load_problem(args.problem)
    print(
        f"[run_agent] Problem : {problem.meta.name}",
        f"\n[run_agent] Model   : {args.model}  ({args.provider}, {api_kind})",
        f"\n[run_agent] Mode    : {'multi-turn (tools)' if args.multi_turn else 'single-shot'}",
        sep="",
    )

    # ------------------------------------------------------------------ #
    # Build agent                                                          #
    # ------------------------------------------------------------------ #
    tools = None
    tool_ctx = None

    if args.multi_turn:
        from tools.tools import get_tools, ToolContext
        tool_names = [t.strip() for t in args.tools.split(",")] if args.tools else None
        tools = get_tools(tool_names)
        tool_ctx = ToolContext(
            problem=problem,
            device=args.device,
            global_seed=args.seed,
            n_timing_trials=args.n_timing,
            verbose=args.verbose,
        )
        print(f"[run_agent] Tools   : {[t.name for t in tools]}")

    agent = TranslationAgent(
        client=client,
        model=args.model,
        problem=problem,
        tools=tools,
        tool_ctx=tool_ctx,
        api_kind=api_kind,
        reasoning_effort=args.reasoning_effort,
        omit_responses_reasoning=omit_responses_reason,
        max_turns=args.max_turns,
        max_api_retries=args.retries,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #
    print("[run_agent] Running...")
    result = agent.generate()

    if agent.last_usage:
        u = agent.last_usage
        total = u.get("total_tokens") or (
            (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0)
        )
        print(f"[run_agent] Tokens  : {total}")

    # ------------------------------------------------------------------ #
    # Multi-turn result                                                    #
    # ------------------------------------------------------------------ #
    if args.multi_turn:
        if result is None:
            print("[run_agent] Agent did not call submit_kernel — no final result.")
            sys.exit(1)

        print("\n--- Multi-turn Result ---")
        summary = result.summary() if hasattr(result, "summary") else vars(result)
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, indent=2))
            print(f"[run_agent] Result JSON: {args.json_out}")

        sys.exit(0 if getattr(result, "final_score", 0) > 0 else 1)

    # ------------------------------------------------------------------ #
    # Single-shot result                                                   #
    # ------------------------------------------------------------------ #
    candidate_src: str = result  # type: ignore[assignment]

    if args.out:
        Path(args.out).write_text(candidate_src)
        print(f"[run_agent] Candidate: {args.out}")
    else:
        print("\n--- Candidate ---\n")
        print(candidate_src)

    if args.eval:
        from ktbench.eval.harness import eval_translation
        print(f"\n[run_agent] Evaluating on cuda:{args.device}...")
        eval_result = eval_translation(
            candidate_src,
            problem,
            device=args.device,
            global_seed=args.seed,
            n_timing_trials=args.n_timing,
            verbose=args.verbose,
        )
        print("\n--- Evaluation Result ---")
        summary = eval_result.summary()
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\n  final_score: {eval_result.final_score}")

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, indent=2))
            print(f"[run_agent] Result JSON: {args.json_out}")

        sys.exit(0 if eval_result.correctness_rate == 1.0 else 1)


if __name__ == "__main__":
    main()
