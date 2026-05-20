# Ensemble integration: in-depth guide

This document is the contract documentation for the ensemble integration. It covers what each piece does, how the pieces compose into a working agent run, and what you change to extend each axis (new persona, new actor, new scenario, new problem, new metric).

The intended reader is someone who has read the top-level `README.md` and the `plan.md` and now wants to write code against the integration. Examples come from the existing files; cross-links go to specific section headers rather than to whole files.

The ensemble integration lives in three files: `integrations/ensemble.py`, `scenarios/translate_problem.py`, and `scenarios/judge_translate.py`. The rest of the repo (problem layout, eval harness, scoring, anti-hack machinery) is shared with the Inspect AI integration and is documented in `plan.md` and `harness_plan.md`.

## Why ensemble at all

KTBench's eval harness is harness-agnostic. The five tools the model calls (`static_check`, `compile_kernel`, `run_correctness`, `get_gpu_specs`, `submit_kernel`) have a `Tool.execute(ctx, **kwargs) -> ToolResult` interface; anything that can drive a tool-calling loop against that signature can be a KTBench harness. Inspect AI does this with its `@tool` decorator, an async closure per tool, and a `Task` that runs `use_tools()` then `generate()`. The ensemble integration does the same shape using ensemble's `PluginTool` wrappers and a scenario that calls `spawn_agent`.

The ensemble path adds two affordances Inspect AI does not have:

1. **Multi-actor scenarios.** A scenario can spawn two or more agents (or a mix of agents and simulated users) that share the same world and tool registry but each carry their own context window, persona, and tool restriction. The `code_reviewer` persona reads the trace as the author builds the kernel and flags suspicious submissions. Comparing single-agent runs against author + reviewer runs on the same problem cell measures whether the multi-actor pattern improves the held-out pass rate. This is the central research question this integration is designed to answer.

2. **A trace recorder + viewer.** Every event (tool call, tool result, state diff, cost annotation, grader output, system note) lands in a JSONL trace that ensemble's static viewer can render. The viewer polls the trace file every two seconds, so traces are readable as they grow. The publisher at `scripts/publish_traces.py` walks `traces/`, parses each into a summary record, and writes a `runs.json` plus a per-run viewer page to gh-pages.

If you only need single-shot or multi-turn Inspect AI runs across providers, stay on the Inspect AI integration. Reach for the ensemble integration when you want a reviewer actor in the loop or when you want the leaderboard.

## How the pieces compose

The agent loop on the ensemble path looks like this:

1. `ensemble run ktbench.translate_problem --world ktbench` (or the equivalent Python invocation) hands control to ensemble's scenario runner.
2. Ensemble constructs `World('ktbench')`, which fires `integrations.ensemble._setup()`. The factory reads env vars, builds a `KTBenchState` (which loads the problem and constructs a `ToolContext`), wraps the five KTBench tools as `PluginTool`s, and returns the tools plus six predicates.
3. The scenario function in `scenarios/translate_problem.py` runs. It reads env vars to pick a persona, a model, a turn budget. It loads the persona's system prompt, concatenates it with `build_prompt(problem)`, and calls `world.spawn_agent(...)` with the combined system prompt and the five tool names.
4. The agent loop turns. Each turn, the LLM produces text plus zero or more tool calls. Each tool call dispatches to the matching `PluginTool.fn` wrapper, which calls into the KTBench tool's `.execute(ctx, **kwargs)` and serialises the `ToolResult` into the JSON envelope ensemble expects (`{"effect": ..., "diff": ...}`).
5. The agent's `submit_kernel` call runs the full KTBench eval pipeline (static check, compile, structured correctness, stress, performance, score). The result lands in `state.submissions` as a Python dict, and a state-diff event with the per-stage metadata fires on the trace.
6. The scenario's `until` predicate fires when the agent has spent its turn budget or has submitted. Grader predicates evaluate against `state.submissions` and return six numeric cells. The scenario returns the cells as a dict; ensemble emits a grader event on the trace.
7. The publisher (when run after the fact) walks `traces/`, parses each trace, builds `runs.json`, and publishes to gh-pages.

The seam that keeps this clean is `ToolContext`. It carries the problem, the device, the seed, and the timing-trial count, and every tool's `execute()` takes it as the first positional. Same `ToolContext` for every tool call in a session, fresh one per `World('ktbench')` construction. The five tools never see ensemble; ensemble never sees the eval harness internals.

## The shared state object

`KTBenchState` in `integrations/ensemble.py` is the object every `PluginTool` and every predicate closes over. It is constructed once per `World('ktbench')` and carries three things:

- `problem: Problem` — the loaded problem object. Read by every tool that needs the source kernel, the oracle, the test suite, or the target hardware spec.
- `ctx: ToolContext` — the shared context object every KTBench `Tool.execute()` takes. Bundles the problem with the device index, the global seed, and `n_timing_trials`.
- `submissions: list[dict]` — append-only ledger of submit_kernel metadata. Populated by the `capture_submit=True` branch of `_wrap_tool`. The grader predicates read this directly instead of walking the trace, which means they are O(1) per call.

The state is per-world, not per-agent. In `judge_translate` both the author and the reviewer share the same `KTBenchState`. That is deliberate: the reviewer's `run_correctness` call must use the same test suite + seed the author's `submit_kernel` will use, otherwise the reviewer's "I ran correctness and it passes" claim has no relationship to the author's eventual submit-time result.

If you need per-actor state that other actors do not see, put it in the actor's `hidden_state` on the persona TOML rather than in the shared `KTBenchState`.

## The tool wrappers

Every KTBench `Tool` is wrapped via `_wrap_tool(tool_obj, state, *, capture_submit=False)` in `integrations/ensemble.py`. The wrapper does five things per call:

1. Parse `args_json` to a Python dict (ensemble's JSON-string ABI).
2. Call `tool_obj.execute(state.ctx, **args)`. Any exception turns into a structured error envelope so the agent gets a useful failure message instead of a Python traceback.
3. Run `fmt_result(result)` to produce the LLM-facing summary string (the tool's `.output` plus a `[meta] {...}` JSON tail).
4. If `capture_submit=True` (only `submit_kernel`), append the result's metadata to `state.submissions` and emit a structured state-diff event on the trace under the `ktbench_submissions` field so post-hoc analysis tools and the publisher can read it.
5. Return a JSON envelope ensemble understands.

The five tool names (`static_check`, `compile_kernel`, `run_correctness`, `get_gpu_specs`, `submit_kernel`) come from the underlying `Tool.name` attribute and are stable across both harnesses. A scenario that wants to give the agent a different tool subset passes the names it wants to `spawn_agent(tools=[...])`; ensemble's per-agent tool filtering enforces the restriction.

If you add a new KTBench tool, the wrapper handles it automatically — you only need to add it to `build_all_tools(state)` in the integration.

## The predicates

Six grader predicates are registered in `build_predicates(state)`. Each is a closure over `state.submissions` (and in one case the trace).

- `submit_called` — true if any submission landed in the ledger.
- `submit_passed` — true if any submission has `final_score > 0`.
- `correctness_passed` — true if any submission has `correctness_rate >= 1.0`.
- `stress_passed` — true if any submission has `stress_pass_rate >= 0.9`.
- `utilization_passed` — true if any submission has `sol_score > 0`.
- `static_check_failed` — walks the trace for any `static_check` tool result with `effect.ok == false`. Walks the trace rather than reading state because the static checker is not a submission; multiple static checks can happen per run, and the predicate fires on any one failing.

The thresholds (1.0 for correctness, 0.9 for stress, > 0 for SOL) match KTBench's eval defaults. Changing them is a one-line edit in `build_predicates`.

Predicates take `(trace_json: str, args_json: str)` and return `bool`. The trace argument is the deserialised list of events; the args argument is the JSON-serialised dict the grader passed (empty for unqualified predicates, `{"user_id": "alice"}` for per-user predicates). KTBench predicates do not currently use args; if you add per-actor predicates, that is where the dispatch happens.

## The scenarios

Two scenarios ship with the integration:

`scenarios/translate_problem.py` is the single-agent baseline. One actor, the five tools, the persona's framing layered on top of `build_prompt(problem)`. The scenario reads `KTBENCH_MODEL`, `KTBENCH_PERSONA`, `KTBENCH_MAX_TURNS` from the environment. Six grader cells are returned.

`scenarios/judge_translate.py` is the multi-actor arm. Two actors: an `author` with the full tool kit and an author persona (default `normal_translation`), and a `reviewer` with the `code_reviewer` persona and a read-only tool subset (`static_check`, `run_correctness` only — not `compile_kernel`, not `submit_kernel`, not `get_gpu_specs`). The reviewer is seeded with a `.say()` to the author so the conversation opens with both roles announced. Both actors share the same `KTBenchState`. Same six grader cells.

Both scenarios call `_log_agent_prompt(...)` after each `spawn_agent` so the trace carries the resolved system prompt for each actor. This is a workaround for the fact that ensemble does not emit a spawn event with the system prompt today; the workaround is a `world._native.log_note(...)` with the persona's full system prompt as the body. The trace viewer renders system notes as a panel, and `publish_traces.py` parses the first line to extract the persona and model for the leaderboard.

## Extending the integration

The integration is intentionally small so extensions land in obvious places. Five common extension paths follow.

### Extending: add a new persona

Personas live as TOML files under `personas/`. Each file declares a name, mode, optional style, optional hidden-state schema, and a `system_prompt.template` body.

```toml
# personas/aggressive_translator.toml
[persona]
name = "aggressive_translator"
mode = "prompted"
description = "Translator that prioritises Hopper-specific optimisations over a literal port."

[persona.style]
tone = "ambitious"
verbosity = "low"

[persona.hidden_state.schema]
target_sol_fraction = { type = "number", default = 0.5 }

[persona.system_prompt]
template = """
<verbatim baseline from normal_translation.toml>

## How you work

You aim to clear 50% SOL on the target. Reach for wgmma, TMA, async copy,
and shared-memory layout rewrites; transliteration is a fallback, not the
goal. Track your target_sol_fraction in hidden state and iterate until
you beat it. The correctness contract from the baseline above is
non-negotiable.
"""
```

Rules of thumb for new personas:

1. **Compose over a baseline.** Either `normal` (write-task framing) or `normal_translation` (translation framing) is the baseline. Copy the verbatim baseline section into the new persona's `system_prompt.template`, then add a `## How you work` section with the intervention. The `methodical_engineer`, `speed_obsessed`, and `code_reviewer` files are working templates.

2. **Decide the role.** Most personas extend the existing role (translator, reviewer, etc.). If you are creating a fundamentally new role — say, an `optimiser` that only modifies an existing submission rather than authoring from scratch — that's a role change. Document it at the top of the file and budget for a corresponding scenario change to match.

3. **Hidden state is for things the grader can read later.** A `verdict` field on `code_reviewer` is consumed by the grader at end of run. A `target_speedup` on `speed_obsessed` is a knob the agent reads via the persona resolver. If a field has neither role, it's noise.

4. **Test the persona load.** Once the file is in place, the simplest verification is `python -c "from ensemble.persona import load_persona; print(load_persona('personas/<name>.toml').system_prompt[:200])"`. If that returns the right text, the persona is wired.

Personas are auto-discovered by ensemble's persona resolver when `PERSONAS_DIR` (in `integrations/ensemble.py`) is set. No registry edit required for a new file.

### Extending: add a multi-actor scenario

The pattern is in `scenarios/judge_translate.py`. The three moving parts are:

1. **Spawn each actor with its own system prompt.** Each `spawn_agent(id=..., persona=..., model=..., system_prompt=..., tools=[...])` call adds an actor to the world. Tools is the per-actor allow-list; passing `["static_check", "run_correctness"]` restricts the reviewer to the read-only subset.

2. **Seed the conversation if the actors need to be aware of each other.** Use `actor.say(target_id, text)` to deliver a seeded message. The judge scenario opens with the reviewer announcing itself to the author so the author's first turn happens with the reviewer's presence acknowledged.

3. **Decide whose state matters.** All actors share the same `KTBenchState`, so submissions from any actor land on the same ledger. If you want predicates to disambiguate "did the author pass" vs "did the reviewer pass", add per-actor predicates that read `args["user_id"]` and filter the ledger by submitter.

A worked example for a different multi-actor shape — say, two parallel authors trying different optimisation strategies, with no reviewer:

```python
# scenarios/two_author_translate.py
@scenario("ktbench.two_author_translate", world="ktbench")
async def two_author_translate(world):
    problem_prompt = prompt_for_env()
    author_a_persona = _persona_system_prompt("normal_translation")
    author_b_persona = _persona_system_prompt("speed_obsessed")

    a = world.spawn_agent(
        id="author_a",
        model=os.environ.get("KTBENCH_MODEL_A", "claude-sonnet-4-5"),
        system_prompt=author_a_persona + "\n\n---\n\n" + problem_prompt,
        tools=["static_check", "compile_kernel", "run_correctness",
               "get_gpu_specs", "submit_kernel"],
    )
    b = world.spawn_agent(
        id="author_b",
        model=os.environ.get("KTBENCH_MODEL_B", "claude-sonnet-4-5"),
        system_prompt=author_b_persona + "\n\n---\n\n" + problem_prompt,
        tools=["static_check", "compile_kernel", "run_correctness",
               "get_gpu_specs", "submit_kernel"],
    )

    yield world.until(world.turn_count > int(os.environ.get("KTBENCH_MAX_TURNS", "40")))

    # Per-author grader cells require per-actor predicates; the
    # default predicates aggregate across submissions.
    yield {
        "any_submit_passed": 1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "any_correctness_passed": 1.0 if world.evaluate_predicate("correctness_passed") else 0.0,
    }
```

That scenario gives you two parallel authors with different personas on the same problem. The grader cells aggregate across the two; to separate them you'd add `args["actor_id"]` to the wrapped `submit_kernel` and predicates that filter `state.submissions` by actor.

Each new scenario needs the `@scenario("ktbench.<name>", world="ktbench")` decorator. Once decorated and imported, it registers in ensemble's `_REGISTRY` and is callable via `ensemble run ktbench.<name>` or via the Python invocation we used in the smoke test.

### Extending: add per-actor predicates

The default predicates aggregate across all submissions. To separate author from reviewer:

```python
# In integrations/ensemble.py's build_predicates()

def author_submit_passed(trace_json: str, args_json: str) -> bool:
    args = json.loads(args_json) if args_json else {}
    actor_id = args.get("actor_id") or args.get("user_id")
    if not actor_id:
        return False
    return any(
        r.get("actor_id") == actor_id and (r.get("final_score") or 0) > 0
        for r in state.submissions
    )
```

This requires the wrapped `submit_kernel` to also record the actor id on each submission. Add that by reading the call's actor from the trace at wrap time (the tool's wrapped function does not directly receive the actor; you'd need ensemble to thread it in, or you can stamp the most recent actor from the trace on each call). The latter is brittle; the cleaner fix is an ensemble-side extension that exposes the calling actor's id to the tool's wrapped function.

For now, scenarios that need actor disambiguation should consume `state.submissions` directly via a custom Python scenario rather than the grader DSL.

### Extending: add a new problem

Problem authoring does not change between the Inspect AI and ensemble paths because the eval harness is the same. The full schema for `meta.toml`, `source.py`, `oracle.py`, `reference_tgt.py`, `test_suite.toml`, and `generator.py` is in `plan.md` under "Repo Structure" and "Problem Format". A new problem is one directory under `problems/`:

```
problems/<problem_id>/
├── meta.toml         # axis, tags, difficulty, tolerances
├── source.py         # source-DSL implementation, class ModelNew
├── oracle.py         # numerically-correct PyTorch eager reference
├── reference_tgt.py  # hand-tuned target-DSL impl, performance baseline
├── test_suite.toml   # structured cases + stress shape ranges
├── generator.py      # make_inputs(shapes, dtype, rng, device) -> [Tensor]
└── notes.md          # (optional) human description and gotchas
```

After authoring, run `scripts/build_oracle_tensors.py --problem problems/<id>` to pre-generate oracle tensors. Then verify with `scripts/run_eval.py --problem problems/<id> --candidate problems/<id>/reference_tgt.py`. If the reference target passes its own eval, the problem is well-formed.

For the ensemble path specifically, the problem is loaded from `KTBENCH_PROBLEM_PATH`. The path is resolved relative to the KTBench repo root; pass an absolute path or one relative to the repo root.

A useful sanity check before adding a new translation axis is to confirm `tgt_dsl` exists in `src/ktbench/registry/dsl.py:_REGISTRY` and `tgt_hw` exists in `src/ktbench/registry/hardware.py:_REGISTRY`. Both registries are explicit dicts; an unknown DSL or HW key raises at problem load time.

### Extending: a new grader cell or score axis

A new grader cell is two edits.

1. Add the predicate to `build_predicates(state)` in `integrations/ensemble.py`. The predicate has access to the trace and to `state.submissions`. Return a bool.

2. Add the cell to the scenario's returned dict. The grader cell name is the dict key; the value is `1.0 if world.evaluate_predicate("<name>") else 0.0`.

A new score axis (something more than a yes/no question) requires a small extension. The KTBench eval pipeline returns structured metadata per submission; the integration captures everything via the `capture_submit=True` branch of `_wrap_tool`. To surface a continuous score (say, mean stress_pass_rate across submissions) rather than a bool, add a helper to the scenario that reads `state.submissions` directly and returns a float in the dict:

```python
# Inside the scenario
mean_stress = (
    sum(s.get("stress_pass_rate") or 0 for s in state.submissions) / len(state.submissions)
    if state.submissions else 0.0
)
yield {
    "submitted": 1.0 if world.evaluate_predicate("submit_called") else 0.0,
    "mean_stress_pass_rate": mean_stress,
    ...
}
```

For this to work, the scenario needs access to the state. Ensemble does not expose it directly; you can either import `_state_from_env` and load a parallel copy (cheap, but creates a second ToolContext that is not shared with the world) or read `state.submissions` indirectly by walking the trace for `ktbench_submissions` state-diff events. Walking the trace is the robust choice; it does not require the scenario to know about the integration's state object.

### Extending: a different problem source (not KTBench)

The integration assumes `Problem` objects from `ktbench.problem.load_problem`. If you want to drive the same agent loop against, say, KernelBench problems that have been wrapped to look like KTBench problems, write a `Problem`-shaped adapter and point `KTBENCH_PROBLEM_PATH` at the adapter's path. The five tools all take a `ToolContext` whose only requirement is a `problem` attribute with the right shape; nothing in the wrapper layer assumes the problem came from a TOML file.

This is the migration path for the popcornbench domain kernels (graph, probabilistic, bio) that currently live as PyTorch references under `kernels/kernelbench/level{1..4}/popcorn/`. Wrap each as a KTBench `Problem` and they drop into the existing harness.

## Common gotchas

A few things that have already bitten and are worth flagging.

**The `ensemble run` CLI shells to `uv run`.** That means the CLI tries to resolve `KTBench/pyproject.toml` dependencies before doing anything. If a dep is yanked or unresolvable, the CLI fails with a confusing error from uv. The fix is to make sure `pyproject.toml` resolves cleanly; if it does not and you're debugging, the workaround is to call `ensemble.cli_run` directly via Python or to invoke the scenario from the registry as we did in the smoke test.

**`world._native.log_note` is private.** The integration calls into ensemble's native bindings to emit the spawn-time system prompt and the problem prompt on the trace. This is a workaround for ensemble not exposing a public spawn event with the system prompt. The right long-term fix is on the ensemble side; until then, the `_native` calls are isolated to two helper functions per scenario, so the day they break is localised.

**Tool restrictions are an allow-list, not a deny-list.** When you pass `tools=[...]` to `spawn_agent`, the agent only sees those tools. If you pass `tools=None`, the agent sees every tool registered with the world (which today includes all five). Default to explicit allow-lists in any new scenario; the default-everything behaviour makes it too easy to give the reviewer `submit_kernel` by accident.

**The mock backend returns no tool calls by default.** Running `ensemble run ktbench.translate_problem --backend mock` produces a trace where the agent never calls any tool because the mock script is empty. That is useful for verifying the integration shape, not for end-to-end testing. For a real end-to-end run, use a real LLM (`--backend anthropic` plus the API key) or write a mock script that pushes scripted tool calls in sequence (see `examples/plank/bake_trace.py` in the ensemble repo for how the plank world does this).

**State is per-`World('ktbench')`, not per-scenario.** Each construction reads env vars. If you run two scenarios in the same Python process and want them to use different problems, set the env var between them (or pass different scratch dirs). The state is captured at world construction; mutating env vars mid-run does nothing.

**Persona load is lazy and tolerant.** A missing persona TOML returns `None` from `load_persona`; the scenario then falls back to the model's default behaviour with no persona prompt. This is by design (so a typo in a persona name does not crash a sweep), but it means you should verify the persona loaded correctly by inspecting the trace's `agent_spawn` note: if the system prompt is empty there, the persona did not resolve.

## Running

The integration supports two invocation paths.

**Through the ensemble CLI**, once `uv` can resolve the project:

```bash
KTBENCH_PROBLEM_PATH=problems/softmax_h200_to_triton \
KTBENCH_MODEL=claude-sonnet-4-5 \
KTBENCH_PERSONA=normal_translation \
ensemble run ktbench.translate_problem --world ktbench --package-dir scenarios
```

**Directly through Python**, useful for debugging or for cases where uv is misbehaving:

```python
import asyncio, sys
sys.path.insert(0, '/path/to/ensemble/python')
sys.path.insert(0, '/path/to/KTBench/src')
sys.path.insert(0, '/path/to/KTBench')
import scenarios.translate_problem  # registers the world + scenario
from ensemble.scenario import _REGISTRY

result = asyncio.run(
    _REGISTRY['ktbench.translate_problem'](trace_path='traces/run.jsonl')
)
print(result.scores)
```

Both paths produce the same trace shape and the same `RunResult`.

## Publishing

After one or more runs, publish the leaderboard:

```bash
python scripts/publish_traces.py --ensemble-root ~/Documents/ensemble
```

The script walks `traces/`, parses each trace for submission metadata, builds `runs.json` at the gh-pages root, copies `site/` (the leaderboard + run index) and the ensemble per-run viewer assets. For a continuous publish during a long sweep, pass `--watch 300` to republish every five minutes; pass `--dry-run` to inspect the worktree without committing or pushing.

The leaderboard ranks by `final_score` (KTBench's `correctness × stress × sol_score`) and breaks out a per-(src_dsl, tgt_dsl) view. SOL is the headline performance number because it is physically bounded by hardware peak; speedup vs the reference target is captured in the per-run record but is not the ranking column.

## File map

```
KTBench/
├── integrations/
│   └── ensemble.py             # KTBenchState, tool wrappers, predicates, register_world
├── scenarios/
│   ├── __init__.py             # (empty; lets scenarios/ be a package)
│   ├── translate_problem.py    # single-agent baseline
│   └── judge_translate.py      # author + reviewer multi-actor
├── personas/
│   ├── normal.toml             # baseline for PyTorch -> CUDA write task
│   ├── normal_translation.toml # baseline for translation task (A100 -> H100)
│   ├── methodical_engineer.toml # intervention: lint-first, careful iteration
│   ├── speed_obsessed.toml     # intervention: aggressive optimisation
│   └── code_reviewer.toml      # role redirect: audit, not authoring
├── site/
│   ├── style.css               # design tokens + base styles
│   ├── app.js                  # client-side leaderboard + runs index logic
│   ├── index.html              # leaderboard home (final_score, SOL, by axis)
│   └── runs.html               # full run index with filters and sort
├── scripts/
│   └── publish_traces.py       # walks traces/, builds runs.json, publishes gh-pages
└── docs/
    └── ensemble_setup.md       # this document
```
