"""
run_agent.py — Use an LLM to translate a KTBench problem kernel.

The agent builds a prompt from the problem, calls the LLM, extracts ModelNew,
optionally writes the candidate to a file, and optionally runs the full evaluator.

Provider presets
----------------
  openai   OPENAI_API_KEY         Responses API (supports o3/o4-mini reasoning)
  azure    AZURE_OPENAI_API_KEY   Responses API; --base-url required
  grok     XAI_API_KEY            Chat Completions (https://api.x.ai/v1)

Azure example:
  python scripts/run_agent.py \\
    --problem problems/softmax_h200_to_triton \\
    --model gpt-4.1 \\
    --provider azure \\
    --base-url https://my-resource.cognitiveservices.azure.com/openai/v1/ \\
    --out candidate.py --eval

OpenAI example:
  python scripts/run_agent.py \\
    --problem problems/softmax_h200_to_triton \\
    --model o3 --reasoning-effort medium \\
    --out candidate.py --eval --verbose

Grok example:
  python scripts/run_agent.py \\
    --problem problems/softmax_h200_to_triton \\
    --model grok-3 --provider grok \\
    --out candidate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
                        help="Path to problem directory (e.g. problems/softmax_h200_to_triton)")

    # Model / provider
    parser.add_argument("--model", required=True,
                        help="Model name, e.g. gpt-4o, o3, gpt-4.1, grok-3")
    parser.add_argument("--provider", default="openai",
                        choices=list(PROVIDER_PRESETS.keys()) + ["custom"],
                        help="Provider preset (default: openai). Use 'custom' with "
                             "--api-key-env / --base-url / --api-kind.")
    parser.add_argument("--api-key-env", default=None,
                        help="Env var holding the API key (overrides provider preset)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (required for azure; optional for custom)")
    parser.add_argument("--api-kind", default=None, choices=["responses", "chat"],
                        help="API style: 'responses' (OpenAI/Azure) or 'chat' "
                             "(Grok/compatible). Overrides provider preset.")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort for Responses API reasoning models.")
    parser.add_argument("--omit-responses-reasoning", action="store_true",
                        help="Skip the 'reasoning' parameter (for providers that reject it).")

    # Retries
    parser.add_argument("--retries", type=int, default=3,
                        help="API retry budget (default: 3)")

    # Output
    parser.add_argument("--out", default=None,
                        help="Write candidate source to this file (default: print to stdout)")

    # Optional evaluation
    parser.add_argument("--eval", action="store_true",
                        help="Run the full KTBench evaluator on the generated candidate")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index for evaluation (default: 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for evaluation")
    parser.add_argument("--json-out", default=None,
                        help="Write evaluation result JSON to this file")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve provider settings                                            #
    # ------------------------------------------------------------------ #
    preset = PROVIDER_PRESETS.get(args.provider, {})

    api_key_env           = args.api_key_env           or preset.get("api_key_env", "OPENAI_API_KEY")
    base_url              = args.base_url               or preset.get("base_url")
    api_kind              = args.api_kind               or preset.get("api_kind", "responses")
    omit_responses_reason = args.omit_responses_reasoning or preset.get("omit_responses_reasoning", False)

    if args.provider == "azure" and not base_url:
        parser.error("--provider azure requires --base-url")

    # ------------------------------------------------------------------ #
    # Build client + agent                                                 #
    # ------------------------------------------------------------------ #
    try:
        client = make_client(api_key_env=api_key_env, base_url=base_url)
    except RuntimeError as e:
        sys.exit(f"[run_agent] {e}")

    problem = load_problem(args.problem)

    print(
        f"[run_agent] Problem:  {problem.meta.name}",
        f"\n[run_agent] Model:    {args.model}  ({args.provider}, {api_kind})",
        f"\n[run_agent] Calling LLM...",
        sep="",
    )

    agent = TranslationAgent(
        client=client,
        model=args.model,
        problem=problem,
        api_kind=api_kind,
        reasoning_effort=args.reasoning_effort,
        omit_responses_reasoning=omit_responses_reason,
        max_api_retries=args.retries,
        verbose=args.verbose,
    )

    try:
        candidate_src = agent.generate()
    except RuntimeError as e:
        sys.exit(f"[run_agent] Generation failed: {e}")

    if agent.last_usage:
        usage = agent.last_usage
        total = usage.get("total_tokens") or (
            (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
        )
        print(f"[run_agent] Tokens used: {total}")

    # ------------------------------------------------------------------ #
    # Write / print candidate                                              #
    # ------------------------------------------------------------------ #
    if args.out:
        Path(args.out).write_text(candidate_src)
        print(f"[run_agent] Candidate written to: {args.out}")
    else:
        print("\n--- Candidate ---\n")
        print(candidate_src)

    # ------------------------------------------------------------------ #
    # Optional evaluation                                                  #
    # ------------------------------------------------------------------ #
    if args.eval:
        from ktbench.eval.harness import eval_translation

        print(f"\n[run_agent] Evaluating on device cuda:{args.device}...")
        result = eval_translation(
            candidate_src,
            problem,
            device=args.device,
            global_seed=args.seed,
            verbose=args.verbose,
        )

        print("\n--- Evaluation Result ---")
        summary = result.summary()
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\n  final_score: {result.final_score}")

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, indent=2))
            print(f"[run_agent] Result JSON written to: {args.json_out}")

        sys.exit(0 if result.correctness_rate == 1.0 else 1)


if __name__ == "__main__":
    main()
