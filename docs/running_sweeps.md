# Running KTBench sweeps

This guide is the comprehensive recipe for taking a new translation
problem from zero to a published leaderboard cell. Four parts:

1. Adding a new problem to `problems/`.
2. Writing a sweep config that tells the runner which (model x problem
   x persona x scenario) cells to evaluate.
3. Running the config and the supporting commands you will use to
   watch it.
4. What actually happens under the hood when the config runs, so you
   can debug a stuck cell or extend the runner.

For a deeper dive into the ensemble plumbing this guide stands on top
of, see `docs/ensemble_setup.md`. For the bare problem-authoring
schema (what each file in `problems/<id>/` must contain), see
`docs/adding_problems.md`.

## 1. Add a new problem

Every problem lives in `problems/<problem_id>/` and ships seven
files. The schema is the authoritative source; this section is a
recipe.

```
problems/<problem_id>/
├── meta.toml         # translation axis, tolerances, difficulty
├── source.py         # the source-DSL kernel the model translates from
├── reference_tgt.py  # a hand-written target-DSL kernel (perf baseline)
├── oracle.py         # numerically correct PyTorch eager reference
├── test_suite.toml   # structured cases + stress shape ranges
├── generator.py      # make_inputs(shapes, dtype, rng, device) -> [Tensor]
└── notes.md          # optional human description and gotchas
```

The invariant tying them together: every `ModelNew.forward` accepts
the same input tensors that `generator.make_inputs` produces. If your
op takes `(q, k, v)`, every file accepts `(q, k, v)` in that order.

### Step by step

1. **Scaffold.** `python scripts/add_problem.py --id <problem_id>
   --src-dsl cuda --src-hw nvidia_a100_sxm --tgt-dsl triton
   --tgt-hw nvidia_h100_sxm --name "..."` writes template files
   you fill in.

2. **Pick an axis.** Set `src_dsl`, `src_hw`, `tgt_dsl`, `tgt_hw` in
   `meta.toml`. The hardware keys live in
   `src/ktbench/registry/hardware.py:_REGISTRY`; DSLs live in
   `src/ktbench/registry/dsl.py:_REGISTRY`. Unknown keys raise at
   problem load.

3. **Write `oracle.py`.** PyTorch eager, no custom kernels. This is
   the numerical contract. The grader compares the candidate's output
   to this within the tolerances declared in `meta.toml`.

4. **Write `source.py`.** The source kernel the model sees. Use
   `torch.utils.cpp_extension.load_inline` for CUDA, or a JIT-loaded
   `.py` for Triton/Tilelang/Helion. The candidate must consume the
   same input contract.

5. **Write `reference_tgt.py`.** Same op, expressed natively in the
   target DSL. This is the leaderboard's performance baseline; the
   candidate's `speedup_vs_ref` is denominated against it.

6. **Write `test_suite.toml`.** Structured cases (fixed shapes,
   random values), plus a `[stress]` block that samples both shapes
   and values randomly. Both pull dtype from the case.

7. **Write `generator.py`.** `make_inputs(shapes, dtype, rng, device)`
   returns a `list[Tensor]`. All randomness goes through `rng` so
   values cannot be hardcoded by the candidate.

8. **Build oracle tensors** (optional, only when you want
   pre-computed reference outputs cached on disk):
   `python scripts/build_oracle_tensors.py --problem
   problems/<problem_id> --device 0`.

9. **Verify the source kernel.** Treat the source as a candidate and
   run the full pipeline against the new problem to confirm it
   compiles and passes correctness:

   ```
   python scripts/run_eval.py \
       --problem problems/<problem_id> \
       --candidate problems/<problem_id>/source.py \
       --device 0 --n-timing 5 --verbose
   ```

   If the source kernel fails its own oracle, your tolerances are too
   tight or the generator and the source disagree on the input shape.
   Fix before continuing; the model cannot do better than the
   source on this problem.

10. **Generate scenarios.** `python
    integrations/ensemble/gen_scenarios.py --multi-actor` emits a
    single-agent `<problem_id>.py` and a multi-actor
    `judge_<problem_id>.py` scenario per problem directory. Existing
    files are kept unless `--overwrite` is passed.

11. **Smoke-test with the mock backend.** Confirms the scenario,
    world, and tool wrappers load:

    ```
    KTBENCH_PROBLEM_PATH=problems/<problem_id> \
    ensemble run ktbench.<problem_id> \
        --world ktbench --backend mock --no-sync
    ```

    Expect a short trace with `agent_spawned`, `user_spawned`, and
    `scheduler quiescent` events (mock returns no tool calls).

The problem is now ready to appear in a sweep.

## 2. Create a sweep config

A "sweep" is a set of cells the runner walks. Each cell is one (model,
problem, persona, scenario shape) tuple. Today the launcher
(`scripts/launch_waves.sh`) hard-codes the problem list and reads the
model + persona from environment variables. The next section shows the
exact env vars; the rest of this section explains the knobs.

### The cell axes

| Axis | Env var (or flag) | Default | Notes |
|---|---|---|---|
| Model | `KTBENCH_MODEL` | `claude-opus-4-7` | One model per process. Passed verbatim to the backend. |
| Problem | `KTBENCH_PROBLEM_PATH` | (none) | One problem per process. Must point at `problems/<id>`. |
| Persona | `KTBENCH_PERSONA` | `normal_translation` | Persona TOML file in `integrations/ensemble/personas/`. |
| Scenario shape | scenario name | `ktbench.<problem_id>` | `ktbench.<problem_id>` is single-agent; `ktbench.judge_<problem_id>` is author + reviewer. |
| Reasoning effort | `KTBENCH_REASONING_EFFORT` | (unset) | Forwarded into the agent's `params` as `reasoning_effort`; the openai backend translates it into `reasoning.effort` on the Responses API. |
| Turn budget | `KTBENCH_MAX_TURNS` | `80` in templates, `300` in `launch_waves.sh` | Per-scenario stop condition: halt when `submit_called` fires or `turn_count > MAX_TURNS`. |
| Tool-call cap | `ENSEMBLE_MAX_TOOL_TURNS` | `8` upstream, `300` in `launch_waves.sh` | Per-inbound-message cap on how many model-driven tool turns the AgentActor runs before returning. |
| Quiescence window | `ENSEMBLE_QUIESCENCE_MS` | `60000` upstream, `600000` in `launch_waves.sh` | How long the scheduler waits for any event before halting. Bump for long blocking tool dispatches like nvcc compiles. |

### Hardware pinning

The launcher pins one ensemble process per GPU with
`CUDA_VISIBLE_DEVICES=N`. The python sandbox worker inherits this env
so each scenario's compile/run/submit dispatch lands on the same GPU
the parent saw. Eight problems on eight H100s is the saturating
unit; if you have fewer GPUs, drop entries from the `PROBLEMS` array
in `launch_waves.sh`.

### Wave A vs Wave B

The launcher runs two passes:

- **Wave A** runs `ktbench.<problem_id>` per problem (single-agent
  baseline).
- **Wave B** runs `ktbench.judge_<problem_id>` per problem (author +
  `code_reviewer` reviewer).

Comparing the two cell-for-cell measures whether the multi-actor
pattern moves the score. Each wave is fully serial; the launcher waits
for every cell in Wave A to finish before starting Wave B so the GPU
budget is not split.

### A two-model sweep

Today the launcher takes one `KTBENCH_MODEL` at a time. Run it twice
with different models to get a head-to-head:

```
KTBENCH_MODEL=gpt-5.5 bash scripts/launch_waves.sh
KTBENCH_MODEL=claude-opus-4-7 bash scripts/launch_waves.sh
```

The publisher (next section) walks `traces/` and aggregates every
run into `runs.json`, so the leaderboard reflects both models without
extra wiring.

## 3. Run the sweep

This section is the literal command sequence for a fresh sweep.

### Prerequisites

```
# Python venv with torch+cu124 + ensemble + KTBench installed.
source .venv/bin/activate

# Backend credentials. Azure OpenAI v1 accepts the OPENAI_* env vars
# directly; for OpenAI proper, set OPENAI_API_KEY and skip the URL.
export OPENAI_API_KEY="$(grep TEJAS_AZURE_KEY .env | cut -d= -f2)"
export OPENAI_BASE_URL="https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1"

# Confirm torch sees the GPUs.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Confirm ensemble's world loads.
ensemble worlds list
# expect: ktbench  /scratch/tejas/KTBench/integrations/ensemble
# if missing: `ensemble worlds add ktbench integrations/ensemble`
```

### Launch the sweep in tmux

`scripts/launch_waves.sh` runs Wave A then Wave B. Detach with
Ctrl+B then D; reattach with `tmux attach -t ktbench`.

```
tmux new-session -d -s ktbench "bash scripts/launch_waves.sh \
    2>&1 | tee /tmp/ktbench_waves/orchestrator.log"
```

Per-cell logs land at `/tmp/ktbench_waves/A/<problem>.log` and
`/tmp/ktbench_waves/B_judge/<problem>.log`. Each cell writes a JSONL
trace to `traces/ktbench_<scenario>.jsonl`.

### Watch the publisher

A second process keeps gh-pages in sync while the sweep is running.
Run it in a separate tmux pane (or as a background process):

```
nohup python scripts/publish_traces.py \
    --ensemble-root /scratch/tejas/ensemble \
    --remote tejas \
    --watch 120 \
    > /tmp/ktbench_publish.log 2>&1 &
```

`--watch 120` rebuilds the worktree, copies `site/` and the ensemble
viewer assets next to each trace, writes `runs.json`, commits, and
pushes every two minutes. `--remote tejas` selects a personal fork
when you do not have push rights on `origin`. `--dry-run` builds the
worktree but skips the commit and push.

### Live URLs

The leaderboard root is `https://<github-user>.github.io/<repo>/`. Per-run
viewers live at `<root>/ktbench_<scenario>/viewer.html`. The
publisher logs the chosen URL on the first successful push.

### Verifying a single cell

For a quick end-to-end sanity check without launching the full
sweep, run one cell directly:

```
KTBENCH_PROBLEM_PATH=problems/swiglu_activation_a100_to_h100 \
KTBENCH_MODEL=gpt-5.5 \
KTBENCH_REASONING_EFFORT=low \
ensemble run ktbench.swiglu_activation_a100_to_h100 \
    --world ktbench --backend openai --no-sync
```

A passing run ends with `submit_passed: 1.0` in the printed grader
JSON. A "submitted but failed" cell ends with `submitted: 1.0,
submit_passed: 0.0`, and the reason is in the trace under the
`submit_kernel` tool result (see "Failure modes" below).

### Failure modes

The grader cells are not enough to debug a cell on their own. The
trace under `traces/ktbench_<scenario>.jsonl` is the authoritative
record. Common patterns:

- **`submitted: 0.0`.** The model never called `submit_kernel`. Cause
  is one of: (a) it called the tool cap, (b) it returned no
  tool_calls and went quiescent, (c) the scheduler quiesced during a
  slow tool dispatch. Check the last `tool_call` in the trace; if it
  has no matching `tool_result`, the dispatch did not finish in
  `ENSEMBLE_QUIESCENCE_MS`. Bump that env, or bump
  `ENSEMBLE_MAX_TOOL_TURNS` if the model is mid-iteration.

- **`submitted: 1.0`, `submit_passed: 0.0`, `correctness_passed: 0.0`.**
  The kernel ran but the output disagreed with the oracle. The
  `submit_kernel` tool result summary names the failing case;
  `correctness_detail.per_case` in the trace's state_diff has
  per-case errors.

- **`submitted: 1.0`, `submit_passed: 0.0`, `correctness_passed:
  1.0`, `stress_passed: 1.0`, `sol_above_threshold: 0.0`.** The
  kernel passed correctness and stress but did not clear the SOL
  utilization gate (`compute < 2%` AND `memory < 2%`). KTBench's
  antihack treats this as a suspected no-op and zeros the final
  score. The cause is usually: a kernel that finishes in
  microseconds on the timing input (the first structured case),
  driving its measured DRAM and compute utilization below the floor.
  Either pick a larger first structured case in `test_suite.toml`,
  or lower `[antihack].utilization_floor_pct` in
  `configs/eval_defaults.toml` for that sweep.

- **`static_check_failed: 1.0`.** The static checker found a
  try/except fallback, a `os.kill`/`subprocess` shell, a frame walk,
  or another anti-eval pattern. The kernel is rejected before
  compilation.

- **`backend error (gpt-5.5)`** in a system note. The backend
  rejected the request. The error text is verbatim from the
  provider; common ones include `temperature` rejected by reasoning
  models (set `ENSEMBLE_OPENAI_API=responses` and pass effort via
  `KTBENCH_REASONING_EFFORT` instead of raw temperature),
  `DeploymentNotFound` (the model name is not a valid Azure
  deployment), and 429 rate limits.

- **`backend error (user-model)`** in a system note. The harness
  user actor is calling the openai backend with model name
  `user-model`, which is not a real deployment. This is benign noise
  from a scripted persona that ensemble still routes through the
  backend; it does not affect the agent's run.

The published leaderboard surfaces every cell, including the
non-passing ones, with the outcome marker (`passed`, `failed`, or
`incomplete`). A sweep with many `incomplete` cells means the turn
budget or the tool cap is the bottleneck; a sweep with many
`failed` cells with `sol_above_threshold = 0` means the utilization
gate is the bottleneck.

## 4. What happens when you run the sweep

This section traces a single cell from `bash launch_waves.sh` to the
published leaderboard row, so you know which file to edit when
something is wrong.

1. **The launcher exports env vars.** `launch_waves.sh` sets
   `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `KTBENCH_MODEL`,
   `KTBENCH_MAX_TURNS`, `ENSEMBLE_QUIESCENCE_MS`,
   `ENSEMBLE_MAX_TOOL_TURNS`. These are inherited by every process
   the launcher forks.

2. **The launcher fans 8 cells.** For each `(gpu, problem)` pair, it
   spawns `CUDA_VISIBLE_DEVICES=$gpu KTBENCH_PROBLEM_PATH=... ensemble
   run ktbench.<problem> --world ktbench --no-sync &` in the
   background, then `wait`s for all 8 before launching the next
   wave.

3. **The ensemble CLI shells to python.** `ensemble run` is a thin
   Rust wrapper that invokes
   `python -m ensemble.cli_run --scenario ... --world ktbench
   --backend openai --no-sync`. `--no-sync` skips `uv run` and uses
   the active venv's interpreter directly.

4. **cli_run imports the world's package.** It reads
   `~/.ensemble/worlds.toml` to find `ktbench`'s path, adds it to
   `sys.path`, and runs `import ktbench_world`. That import fires
   `register_world("ktbench", ...)` which captures the setup factory,
   the personas directory, and the world's python package name and
   directory.

5. **cli_run imports the scenarios package.** It tries `import
   scenarios`; if there is no `scenarios/__init__.py`, it walks the
   directory and imports each `*.py` by file path. Each scenario
   module's `@scenario("ktbench.<problem_id>", world="ktbench")`
   decorator registers the coroutine in the global `_REGISTRY`.

6. **The runner constructs a `World`.** `World("ktbench",
   backend="openai", trace_path="traces/ktbench_<scenario>.jsonl")`
   runs the setup factory: `_load_eval_config()` reads
   `configs/eval_defaults.toml`, `build_all_tools(config)` wraps the
   five KTBench tools as `PluginTool`s (three of them marked
   `sandbox=True`), `build_predicates()` returns six predicates. The
   trace sink is opened, the world is registered with the native
   tool and predicate registries.

7. **The scenario coroutine runs.** It sets
   `KTBENCH_PROBLEM_PATH=problems/<id>` in the environment so the
   tool wrappers know which problem to load. It calls
   `prompt_for_path(PROBLEM_PATH)` to build the user-facing prompt,
   logs that as a `problem_prompt` event, resolves the persona
   system prompt, and spawns the agent. The spawn emits an
   `agent_spawned` event with the resolved model, tools, system
   prompt, and `params`.

8. **A harness user seeds the inbox.** Without a queued inbound
   message at startup the scheduler would quiesce on the first
   tick. The harness user posts one `.say()` to the kernel
   engineer with a bare "submit_kernel is the only event that
   counts" kickoff. The kickoff intentionally does not prescribe a
   workflow.

9. **The scheduler starts.** It runs on a tokio task pool: the
   AgentActor steps when a message lands in its inbox; the
   UserActor steps the same way. Tool dispatches go through
   `dispatch_async` to a blocking pool so they do not stall the
   scheduler thread.

10. **The agent's first model call.** AgentActor::step() builds a
    `CompletionRequest` (model, system prompt, history with the
    harness user's message, tool schemas, extra_params with
    `reasoning_effort` if set), and awaits `backend.complete(req)`.

11. **The openai backend translates the request into Responses API
    shape.** ChatMessages become input items: user messages stay
    as roles, assistant tool_calls split into function_call items,
    tool replies become function_call_output items keyed by
    tool_call_id. `reasoning_effort` migrates to `reasoning.effort`
    with `summary=auto`. The body posts to `<base_url>/responses`.

12. **The response comes back.** Output items walk into text,
    reasoning summary, and proposed tool_calls. The actor emits the
    reasoning text as an `agent_message` first, then the regular
    text, then issues each `tool_call` to the bus. Each tool call
    dispatches the matching PluginTool's `fn`.

13. **Tool dispatch goes to the sandbox worker.** For the three
    CUDA-touching tools, the dispatcher spawns `python -m
    ensemble.tool_worker --world ktbench --tool <name>`. The
    worker re-imports `ktbench_world` via
    `ENSEMBLE_SANDBOX_PACKAGE` and `ENSEMBLE_SANDBOX_PACKAGE_DIR`,
    rebuilds the same ToolContext from
    `KTBENCH_PROBLEM_PATH` and `configs/eval_defaults.toml`, runs
    the tool, and writes the JSON envelope on stdout. The parent
    waits and emits the result as a `tool_result` event. A CUDA
    crash in the subprocess kills only the worker.

14. **`submit_kernel` runs the full eval pipeline.** Static check,
    compile, structured correctness (every case in test_suite),
    stress (30 random shape + value trials), performance timing
    (CUDA events on the timing input), antihack utilization gate,
    final score. The result lands on the trace as a `tool_result`
    plus a `state_diff` event with `field=ktbench_submissions`
    carrying per-stage metadata.

15. **The `until` predicate fires.** The scenario's
    `world.until_predicate("submit_called") | (turn_count >
    MAX_TURNS)` halts the scheduler as soon as a submission lands
    (or the turn budget is exhausted). The scenario coroutine
    yields its second value (the grader cells dict).

16. **The grader cells are written to the trace.** Ensemble emits a
    `grader` system event with the final scores. The scheduler
    halts; the trace file is closed; the python entry point prints
    `{scenario, scores, trace_path}` on stdout.

17. **The launcher's `wait` returns.** All 8 cells in Wave A are
    done; the launcher proceeds to Wave B with the same fan-out,
    but pointing at `ktbench.judge_<problem>` scenarios instead.

18. **The publisher rebuilds gh-pages.** Every 120 seconds while the
    sweep is running, `publish_traces.py` walks `traces/`, parses
    each trace for `ktbench_submissions` events, builds a summary
    record per run, writes `runs.json` at the gh-pages worktree
    root, copies the local `site/` (the leaderboard) and the
    ensemble viewer assets next to each trace, commits, and pushes
    to the remote selected by `--remote`.

19. **The leaderboard updates.** Within seconds of the push the
    GitHub Pages build picks up the new commit and republishes the
    static site. A user watching the URL sees the per-cell
    `final_score`, `sol_score`, `correctness_rate`,
    `stress_pass_rate`, model, and persona for every cell completed
    so far.

That's the whole loop. The seam that keeps it clean is the env-var
pattern: the parent's env is the only thing that crosses the sandbox
boundary, so the worker rebuilds the same context the parent had
without sharing Python objects.
