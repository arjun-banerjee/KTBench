# Ensemble integration: in-depth guide

This document is the contract documentation for the ensemble integration. It covers what each piece does, how the pieces compose into a working agent run, and what you change to extend each axis (new persona, new actor, new scenario, new problem, new metric).

The intended reader is someone who has read the top-level `README.md` and `plan.md` and now wants to write code against the integration. Examples come from the existing files; cross-links go to specific section headers rather than to whole files.

The ensemble integration lives entirely under `integrations/ensemble/`. The eval harness, the per-problem layout, the SOL machinery, and the anti-hack patterns are shared with the Inspect AI integration and are documented in `plan.md`.

## Why ensemble at all

KTBench's eval harness is harness-agnostic. The five tools the model calls (`static_check`, `compile_kernel`, `run_correctness`, `get_gpu_specs`, `submit_kernel`) have a `Tool.execute(ctx, **kwargs) -> ToolResult` interface; anything that can drive a tool-calling loop against that signature can be a KTBench harness.

The ensemble path adds three affordances on top of Inspect AI:

1. **Multi-actor scenarios.** A scenario can spawn two or more agents (or a mix of agents and simulated users) that share the same world and tool registry but each carry their own context window, persona, and tool restriction. The `code_reviewer` persona reads the trace as the author builds the kernel and flags suspicious submissions. Comparing single-agent against author+reviewer on the same problem cell measures whether the multi-actor pattern improves correctness or SOL. The worked example is `integrations/ensemble/scenarios/judge_softmax_a100_to_h100.py`.

2. **Sandboxed CUDA dispatch.** A CUDA kernel that does an illegal memory access, hits a watchdog timeout, or otherwise corrupts the CUDA context poisons every subsequent CUDA call in the same process. Without isolation, one bad candidate fails every later eval in the sweep. The ensemble integration marks the three CUDA-touching tools (`compile_kernel`, `run_correctness`, `submit_kernel`) with `sandbox=True`; each call dispatches to a fresh `python -m ensemble.tool_worker` subprocess, so a fatal kernel kills only its worker and the next call gets a clean Python interpreter and a clean CUDA context. `static_check` and `get_gpu_specs` do not touch the GPU and run in-process.

3. **A trace recorder + viewer.** Every event (tool call, tool result, state diff, cost annotation, grader output, system note) lands in a JSONL trace that ensemble's static viewer can render. The viewer polls the trace file every two seconds, so traces are readable as they grow. `scripts/publish_traces.py` walks `traces/`, parses each into a summary record, and writes a `runs.json` plus a per-run viewer page to gh-pages.

If you only need single-shot or multi-turn Inspect AI runs across providers, stay on the Inspect AI integration. Reach for the ensemble integration when you want sandbox isolation, a reviewer actor, or the leaderboard.

## How the pieces compose

The agent loop on the ensemble path looks like this:

1. The CLI invocation `ensemble run ktbench.<problem_id> --world ktbench --manifest integrations/ensemble` (or the equivalent Python invocation in the smoke section below) hands control to ensemble's scenario runner.
2. Ensemble constructs `World('ktbench')`, which fires `_setup()` in `integrations/ensemble/ktbench_world/__init__.py`. The factory loads `configs/eval_defaults.toml` and builds five `PluginTool` wrappers plus six predicates. The three CUDA-touching tools are marked `sandbox=True`.
3. The scenario function (one of `integrations/ensemble/scenarios/*.py`) runs. It sets `KTBENCH_PROBLEM_PATH` in the environment, loads the resolved persona from `personas/`, calls `prompt_for_path(PROBLEM_PATH)` to render the user-facing prompt, concatenates the persona's system prompt with the problem prompt, and calls `world.spawn_agent(...)` with the combined system prompt and the five tool names.
4. The agent loop turns. Each turn, the LLM produces text plus zero or more tool calls. Each tool call dispatches to the matching `PluginTool.fn`:
   - For `static_check` and `get_gpu_specs`, the wrapper runs in-process. It reads `KTBENCH_PROBLEM_PATH` from the env, builds a fresh `ToolContext` from the env + the config, calls `Tool.execute(ctx, **args)`, and returns the JSON envelope.
   - For `compile_kernel`, `run_correctness`, and `submit_kernel`, the wrapper is dispatched to a subprocess via `python -m ensemble.tool_worker --world ktbench --tool <name>`. The subprocess re-imports the `ktbench_world` package (which re-runs `_setup`), re-builds the `ToolContext` from the inherited env, executes the tool, and writes the JSON envelope on its stdout. A CUDA crash in the subprocess does not affect the parent.
5. The agent's `submit_kernel` call runs the full KTBench eval pipeline (static check, compile, structured correctness, stress, performance, score). The result lands on the trace as a `state_diff` event with `field=ktbench_submissions` carrying the per-stage metadata.
6. The scenario's `until` predicate fires when the turn budget is exhausted. Grader predicates evaluate by walking the trace for `ktbench_submissions` events and return six numeric cells. The scenario returns the cells as a dict; ensemble emits a grader event on the trace.
7. The publisher (when run after the fact) walks `traces/`, parses each trace, builds `runs.json`, and publishes to gh-pages.

The seam that keeps this clean is the env-var pattern. The problem path lives in `KTBENCH_PROBLEM_PATH`; the eval knobs live in `configs/eval_defaults.toml`. Both cross the parent/subprocess boundary automatically (env vars are inherited; the config file is read from disk in both processes), so sandboxed dispatches reconstruct the same `ToolContext` the parent would have built.

## The world's setup factory

`integrations/ensemble/ktbench_world/__init__.py` contains the integration's core. The setup factory runs once per `World('ktbench')`:

```python
def _setup():
    config = _load_eval_config()         # reads configs/eval_defaults.toml
    tools = build_all_tools(config)      # five wrapped PluginTools
    predicates = build_predicates()      # six predicates
    return tools, predicates

register_world("ktbench", setup=_setup, personas_dir=PERSONAS_DIR)
```

`_load_eval_config` reads the TOML, defaults to `configs/eval_defaults.toml` at the repo root, and can be overridden via `KTBENCH_EVAL_CONFIG`. The config feeds into the `ToolContext`'s `n_timing_trials` field; other fields (subprocess_timeout, utilization_floor_pct, etc.) are read by the underlying KTBench eval modules directly and do not need to round-trip through the integration.

`build_all_tools(config)` wraps the five KTBench `Tool` instances as ensemble `PluginTool`s via `_wrap_tool`. Each wrapper reconstructs a fresh `ToolContext` from `KTBENCH_PROBLEM_PATH` + the config at every call. This is deliberate: sandboxed dispatches cannot share Python objects with the parent, so the wrapper has to be able to rebuild its context from inheritable state (env vars) alone.

`build_predicates()` returns six `PluginPredicate` instances that walk the trace for `ktbench_submissions` state-diff events. Predicates read from the trace, not from an in-memory ledger, for the same reason: a submission emitted by a sandboxed subprocess never lands in the parent's Python state.

## The tool wrappers

`_wrap_tool` is six steps per call:

1. Parse `args_json` to a Python dict.
2. Build a fresh `ToolContext` from env + config via `_ctx_from_env_and_config`. If `KTBENCH_PROBLEM_PATH` is unset, the wrapper returns a clear error envelope; the agent gets a useful message instead of a Python traceback.
3. Call `tool_obj.execute(ctx, **args)`. Any exception turns into a structured error envelope.
4. Run `fmt_result(result)` to produce the LLM-facing summary string (the tool's `.output` plus a `[meta] {...}` JSON tail).
5. If `capture_submit=True` (only `submit_kernel`), build a `diff` dict with the per-stage metadata and field=`ktbench_submissions`. The trace viewer renders this as a state-changes panel and predicates read it back.
6. Return a JSON envelope ensemble understands.

The wrapper marks `compile_kernel`, `run_correctness`, and `submit_kernel` as `sandbox=True` and `sandbox_world="ktbench"`. Ensemble's scenario runner sees `sandbox=True` at world construction and replaces the tool's normal dispatcher with `_make_sandbox_dispatcher("ktbench", <tool_name>)`, which spawns a subprocess on each call.

## The predicates

Six grader predicates, all in `build_predicates()`:

- `submit_called` — any `ktbench_submissions` state-diff event on the trace.
- `submit_passed` — any submission with `final_score > 0`.
- `correctness_passed` — any submission with `correctness_rate >= 1.0`.
- `stress_passed` — any submission with `stress_pass_rate >= 0.9`.
- `sol_above_threshold` — any submission with `sol_score >= 0.5`.
- `static_check_failed` — any `static_check` tool result with `effect.ok == false`.

The thresholds (1.0 for correctness, 0.9 for stress, 0.5 for SOL) match KTBench's eval defaults from `plan.md`. Adjust them in `build_predicates` or replace the thresholds with config-driven values if you want them tunable per run.

## The scenarios

Two scenario shapes ship with the integration; both live under `integrations/ensemble/scenarios/`.

`softmax_a100_to_h100.py` is the single-agent template. One actor (`kernel_engineer`), the five tools, the persona's framing layered on top of the problem prompt. The scenario reads `KTBENCH_MODEL`, `KTBENCH_PERSONA`, `KTBENCH_MAX_TURNS` from the environment. Six grader cells are returned.

`judge_softmax_a100_to_h100.py` is the multi-actor template. Two actors: an `author` with the full tool kit and an author persona (default `normal_translation`), and a `reviewer` with the `code_reviewer` persona and a read-only tool subset (`static_check`, `run_correctness` only — not `compile_kernel`, not `submit_kernel`, not `get_gpu_specs`). The reviewer is seeded with `.say()` to the author so the conversation opens with both roles announced. Both actors share the same world. Same six grader cells, so comparison against the single-agent template is direct.

Both scenarios:
- Set `KTBENCH_PROBLEM_PATH` at the top of the function so the world's tool wrappers (including sandboxed dispatches) can re-derive the problem.
- Call `prompt_for_path(PROBLEM_PATH)` to build the user-facing prompt.
- Concatenate the persona's system prompt with the problem prompt as the agent's combined system context.
- Call `_log_agent_prompt(...)` after each `spawn_agent` so the trace carries the resolved system prompt for each actor (workaround for ensemble not yet emitting a spawn event with the system prompt).

`gen_scenarios.py` generates these templates per problem dir under `problems/`. Run `python integrations/ensemble/gen_scenarios.py --multi-actor` to emit both shapes per problem.

## Extending the integration

The integration is intentionally small so extensions land in obvious places. Six common extension paths follow.

### Add a new persona

Personas live as TOML files under `integrations/ensemble/personas/`. Each file declares a name, mode, optional style, optional hidden-state schema, and a `system_prompt.template` body.

```toml
# integrations/ensemble/personas/aggressive_translator.toml
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

Rules of thumb:

1. **Compose over a baseline.** Either `normal` (write-task framing) or `normal_translation` (translation framing) is the baseline. Copy the verbatim baseline section into the new persona's `system_prompt.template`, then add a `## How you work` section with the intervention. The `methodical_engineer`, `speed_obsessed`, and `code_reviewer` files are working templates.

2. **Decide the role.** Most personas extend the existing role (translator, reviewer). If you're creating a fundamentally new role — say, an `optimiser` that only modifies an existing submission rather than authoring from scratch — that's a role change. Document it at the top of the file and budget for a corresponding scenario change to match.

3. **Hidden state is for things the grader can read later.** A `verdict` field on `code_reviewer` is consumed by the grader at end of run. A `target_speedup` on `speed_obsessed` is a knob the agent reads via the persona resolver. If a field has neither role, it's noise.

4. **Test the persona load.** Once the file is in place, the simplest verification is:

   ```bash
   python -c "from ensemble.persona import load_persona; print(load_persona('integrations/ensemble/personas/<name>.toml').system_prompt[:200])"
   ```

   If that returns the right text, the persona is wired.

Personas are auto-discovered when `PERSONAS_DIR` (in `ktbench_world/__init__.py`) is set. No registry edit required for a new file.

### Add a multi-actor scenario

The pattern is in `scenarios/judge_softmax_a100_to_h100.py`. Three moving parts:

1. **Spawn each actor with its own system prompt.** Each `spawn_agent(id=..., model=..., system_prompt=..., tools=[...])` call adds an actor to the world. `tools` is the per-actor allow-list; passing `["static_check", "run_correctness"]` restricts the reviewer to the read-only subset.

2. **Seed the conversation if the actors need to be aware of each other.** Use `actor.say(target_id, text)` to deliver a seeded message. The judge scenario opens with the reviewer announcing itself to the author so the author's first turn happens with the reviewer's presence acknowledged.

3. **Decide whose state matters.** All actors share the same `KTBenchState`, so submissions from any actor land on the same ledger. If you want predicates to disambiguate "did the author pass" vs "did the reviewer pass", add per-actor predicates that read `args["user_id"]` and filter the ledger by submitter.

Worked example of a different multi-actor shape — two parallel authors with different personas, no reviewer:

```python
# integrations/ensemble/scenarios/two_author_softmax.py
@scenario("ktbench.two_author_softmax", world="ktbench")
async def two_author_softmax(world):
    os.environ["KTBENCH_PROBLEM_PATH"] = "problems/softmax_a100_to_h100"
    problem_prompt = prompt_for_path("problems/softmax_a100_to_h100")
    a_persona = _persona_system_prompt("normal_translation")
    b_persona = _persona_system_prompt("speed_obsessed")

    world.spawn_agent(
        id="author_a",
        model=os.environ.get("KTBENCH_MODEL_A", "claude-opus-4-7"),
        system_prompt=(a_persona + "\n\n---\n\n" + problem_prompt).lstrip(),
        tools=["static_check", "compile_kernel", "run_correctness",
               "get_gpu_specs", "submit_kernel"],
    )
    world.spawn_agent(
        id="author_b",
        model=os.environ.get("KTBENCH_MODEL_B", "claude-opus-4-7"),
        system_prompt=(b_persona + "\n\n---\n\n" + problem_prompt).lstrip(),
        tools=["static_check", "compile_kernel", "run_correctness",
               "get_gpu_specs", "submit_kernel"],
    )

    yield world.until(world.turn_count > 40)

    yield {
        "any_submit_passed": 1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "any_correctness_passed": 1.0 if world.evaluate_predicate("correctness_passed") else 0.0,
    }
```

The grader cells aggregate across the two; to separate them you'd record the actor id on each submission's diff payload and add per-actor predicates that filter accordingly.

Each new scenario needs the `@scenario("ktbench.<name>", world="ktbench")` decorator. Once decorated and imported, it registers in ensemble's `_REGISTRY` and is callable via `ensemble run ktbench.<name> --manifest integrations/ensemble`.

### Add a new problem

Problem authoring does not change between the Inspect AI and ensemble paths because the eval harness is the same. The full schema for `meta.toml`, `source.py`, `oracle.py`, `reference_tgt.py`, `test_suite.toml`, and `generator.py` is in `docs/adding_problems.md` and `plan.md`. A new problem is one directory under `problems/`:

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

After authoring, run `scripts/build_oracle_tensors.py --problem problems/<id>` to pre-generate oracle tensors. Then `python integrations/ensemble/gen_scenarios.py --multi-actor` emits a per-problem single-actor scenario plus a `judge_<problem_id>.py` multi-actor variant.

For the ensemble path specifically, the problem is bound via `KTBENCH_PROBLEM_PATH`. Per-problem scenario templates set this env var as their first line so a CLI invocation does not have to. Multi-problem sweeps that drive several problems through the same Python process should clear the env between runs (or rely on the scenario's `os.environ[...] = PROBLEM_PATH` to overwrite).

A useful sanity check before adding a new translation axis is to confirm `tgt_dsl` exists in `src/ktbench/registry/dsl.py:_REGISTRY` and `tgt_hw` exists in `src/ktbench/registry/hardware.py:_REGISTRY`. Both registries are explicit dicts; an unknown DSL or HW key raises at problem load time.

### Add a new grader cell or score axis

A new grader cell is two edits.

1. Add the predicate to `build_predicates()` in `ktbench_world/__init__.py`. The predicate has access to the trace; if the cell depends on submission metadata, walk the trace for `ktbench_submissions` diffs and read the relevant field. Return a bool.

2. Add the cell to the scenario's returned dict. The grader cell name is the dict key; the value is `1.0 if world.evaluate_predicate("<name>") else 0.0`.

A new continuous score axis (something more than a yes/no question) requires the scenario to read submission metadata directly. The trace-walk pattern is:

```python
# In a scenario, after the agent loop
import json
trace = world.trace()
submissions = []
for ev in trace:
    payload = ev.get("payload") or {}
    if payload.get("kind") != "state_diff":
        continue
    diffs = payload.get("diff") or []
    items = diffs if isinstance(diffs, list) else [diffs]
    for item in items:
        if isinstance(item, dict) and item.get("field") == "ktbench_submissions":
            submissions.append(item.get("new") or {})

mean_stress = (
    sum(s.get("stress_pass_rate") or 0 for s in submissions) / len(submissions)
    if submissions else 0.0
)
yield {
    "submitted": 1.0 if world.evaluate_predicate("submit_called") else 0.0,
    "mean_stress_pass_rate": mean_stress,
}
```

### Add a config knob

Eval-side knobs live in `configs/eval_defaults.toml`. To add a new knob the integration reads:

1. Add the key to the right section in `eval_defaults.toml`.
2. Read it in `_ctx_from_env_and_config` (or wherever it should land) by inspecting `config.get("<section>", {}).get("<key>", <default>)`.
3. Pass it into `ToolContext` (or wherever it needs to flow) at construction time.
4. Override path: `KTBENCH_EVAL_CONFIG=/path/to/alt.toml` for the whole config, or per-knob env vars if you want finer-grained overrides.

For knobs the underlying `Tool.execute()` reads directly (e.g., `subprocess_timeout`, `utilization_floor_pct`), no integration edit is needed — the tool will pick the value up from its own loader.

### Add a new tool

The integration auto-wraps everything `build_all_tools` returns. To add a sixth tool:

1. Define a new `Tool` subclass in `tools/tools.py` following the existing pattern (`name`, `description`, `input_schema`, `execute(ctx, **kwargs)`).
2. Add it to `get_tools()` in `tools/tools.py` so all harnesses see it.
3. Add a `_wrap_tool(NewTool(), config)` line in `build_all_tools` in `ktbench_world/__init__.py`. Pass `capture_submit=True` if the tool's metadata should land on the trace as a `ktbench_submissions` diff.
4. If the tool touches the GPU, add its name to the `_SANDBOXED` set so the wrapper marks it `sandbox=True`.
5. Add the tool name to the `tools=[...]` list in each scenario that should expose it to the agent.

The Inspect AI integration picks up the new tool automatically because `_make_tools(ctx)` in `integrations/inspect_ai.py` iterates over the same registry.

### Drive sweeps from configs/

For an ensemble-driven sweep, the existing pattern is to drive the per-problem scenarios from a shell script that varies env vars and calls `ensemble run` per cell, then call `scripts/publish_traces.py` once. Example for two models × three problems:

```bash
for problem in softmax_a100_to_h100 fused_rmsnorm_a100_to_h100 swiglu_a100_to_h100; do
    for model in claude-opus-4-7 anthropic/claude-sonnet-4-5; do
        KTBENCH_PROBLEM_PATH=problems/$problem \
        KTBENCH_MODEL=$model \
        KTBENCH_EVAL_CONFIG=configs/eval_defaults.toml \
        ensemble run ktbench.$problem \
            --world ktbench \
            --manifest integrations/ensemble
    done
done

python scripts/publish_traces.py --ensemble-root ~/Documents/ensemble
```

Override `KTBENCH_EVAL_CONFIG` per cell when you want different timing / stress / antihack settings for a specific sweep (e.g., `configs/eval_fast.toml` for smoke runs, `configs/eval_full.toml` for the published sweep).

## Common gotchas

A few things worth flagging.

**The `ensemble run` CLI shells to `uv run`.** The CLI tries to resolve `pyproject.toml` dependencies before doing anything. A yanked or unresolvable dep makes the CLI fail with a confusing error from uv. The fix is to make sure `pyproject.toml` resolves cleanly; if it doesn't and you're debugging, fall back to the direct-Python invocation under "Running" below.

**Sandbox is implicit — the agent does not see the boundary.** When `sandbox=True`, the tool dispatch goes to a subprocess but the agent's tool-result envelope looks identical. The subprocess re-imports `ktbench_world` (which re-runs `_setup`) and re-builds the `ToolContext` from env + config. This means the env vars the parent set (`KTBENCH_PROBLEM_PATH`, `KTBENCH_TIMING_TRIALS`, etc.) cross the boundary automatically; objects in the parent's Python state do not. Build the integration on env + config + the trace; in-memory state is invisible to sandbox workers.

**`world._native.log_note` is private.** The integration calls into ensemble's native bindings to emit the spawn-time system prompt and the problem prompt on the trace. This is a workaround for ensemble not exposing a public spawn event with the system prompt. The right long-term fix is on the ensemble side; until then, the `_native` calls are isolated to two helper functions per scenario.

**Tool restrictions are an allow-list, not a deny-list.** When you pass `tools=[...]` to `spawn_agent`, the agent only sees those tools. `tools=None` means the agent sees every tool the world registered. Default to explicit allow-lists in any new scenario.

**The mock backend returns no tool calls by default.** `ensemble run --backend mock` produces a trace where the agent never calls any tool because the mock script is empty. That is useful for verifying the integration shape, not for end-to-end testing. For a real end-to-end run, use a real LLM (`--backend anthropic` plus the API key) or write a mock script that pushes scripted tool calls in sequence.

**Persona load is lazy and tolerant.** A missing persona TOML returns `None`; the scenario falls back to the model's default behaviour with no persona prompt. Verify the persona loaded correctly by inspecting the trace's `agent_spawn` note: if the system prompt is empty there, the persona did not resolve.

**Sandboxed dispatches inherit env, not state.** A predicate that reads from an in-memory Python ledger sees only the parent's state — submissions made by a sandboxed subprocess never write to it. KTBench's predicates walk the trace instead, which is why each `submit_kernel` call emits a `ktbench_submissions` state-diff event. New tools that produce predicate-relevant state should follow the same pattern.

## Running

The integration supports two invocation paths.

**Through the ensemble CLI** (production path):

```bash
KTBENCH_PROBLEM_PATH=problems/softmax_a100_to_h100 \
KTBENCH_MODEL=claude-opus-4-7 \
KTBENCH_PERSONA=normal_translation \
ensemble run ktbench.softmax_a100_to_h100 \
    --world ktbench --manifest integrations/ensemble
```

**Directly through Python** (debugging escape hatch when `uv` is misbehaving):

```python
import asyncio, sys, os
sys.path.insert(0, '/path/to/ensemble/python')
sys.path.insert(0, '/path/to/KTBench/src')
sys.path.insert(0, '/path/to/KTBench')
sys.path.insert(0, '/path/to/KTBench/integrations/ensemble')
sys.path.insert(0, '/path/to/KTBench/integrations/ensemble/scenarios')

os.environ['KTBENCH_PROBLEM_PATH'] = 'problems/softmax_a100_to_h100'
import softmax_a100_to_h100  # imports ktbench_world too
from ensemble.scenario import _REGISTRY

result = asyncio.run(
    _REGISTRY['ktbench.softmax_a100_to_h100'](trace_path='traces/run.jsonl')
)
print(result.scores)
```

Both paths produce the same trace shape and the same `RunResult`.

## Publishing

After one or more runs, publish the leaderboard:

```bash
python scripts/publish_traces.py --ensemble-root ~/Documents/ensemble
```

The script walks `traces/`, parses each trace for submission metadata (the `ktbench_submissions` state-diff events), builds `runs.json` at the gh-pages root, copies `site/` (the leaderboard + run index) and the ensemble per-run viewer assets. For a continuous publish during a long sweep, pass `--watch 300`; for offline verification, `--dry-run`.

The leaderboard ranks by `final_score` (KTBench's `correctness × stress × sol_score`) and breaks out a per-(src_dsl, tgt_dsl) view. SOL is the headline performance number because it is physically bounded by hardware peak; speedup vs the reference target is captured in the per-run record but is not the ranking column.

## File map

```
KTBench/
├── integrations/ensemble/
│   ├── world.toml                        # ensemble manifest
│   ├── ktbench_world/
│   │   └── __init__.py                   # _setup factory, tool wrappers, predicates
│   ├── scenarios/
│   │   ├── softmax_a100_to_h100.py       # single-agent template
│   │   └── judge_softmax_a100_to_h100.py # multi-actor (author + reviewer) template
│   ├── personas/
│   │   ├── normal.toml                   # baseline for PyTorch -> CUDA write task
│   │   ├── normal_translation.toml       # baseline for translation task
│   │   ├── methodical_engineer.toml      # lint-first intervention
│   │   ├── speed_obsessed.toml           # aggressive intervention
│   │   ├── code_reviewer.toml            # reviewer role redirect
│   │   └── ktbench_harness.toml          # scripted (non-LLM) sender
│   └── gen_scenarios.py                  # auto-generate scenarios per problem
├── configs/
│   └── eval_defaults.toml                # eval knobs read by ktbench_world
├── site/
│   ├── style.css
│   ├── app.js
│   ├── index.html                        # leaderboard home
│   └── runs.html                         # full run index
├── scripts/
│   └── publish_traces.py                 # walks traces/, builds runs.json, publishes gh-pages
└── docs/
    ├── adding_problems.md                # problem authoring guide (Arjun)
    ├── problems.md
    └── ensemble_setup.md                 # this document
```
