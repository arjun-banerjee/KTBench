#!/usr/bin/env bash
# Run Wave A (single-agent) then Wave B (judge) across all 8 KTBench
# problems on 8 H100s.
#
# Each wave fans 8 ensemble processes, one per GPU (pinned via
# CUDA_VISIBLE_DEVICES). Logs land under /tmp/ktbench_waves/<wave>/.
#
# Designed to run unattended inside a tmux pane. The publisher
# (scripts/publish_traces.py --watch ...) keeps gh-pages up to date
# while this runs.

set -uo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source .venv/bin/activate

# Endpoint + key for the kernel-engineer agent's backend. Defaults to
# tejas-eastus2 (gpt-5.5); set KTBENCH_AZURE=thava for gpt-5.4 on
# thava-openai, or KTBENCH_AZURE=popcorn for the priority pool.
case "${KTBENCH_AZURE:-tejas}" in
  tejas)
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://tejas-mohrgcfh-eastus2.services.ai.azure.com/openai/v1}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-$(grep TEJAS_AZURE_KEY .env | cut -d= -f2)}"
    export KTBENCH_MODEL="${KTBENCH_MODEL:-gpt-5.5}"
    ;;
  thava)
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://thava-openai.services.ai.azure.com/openai/v1}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-$(grep THAVA_AZURE_KEY .env | cut -d= -f2)}"
    export KTBENCH_MODEL="${KTBENCH_MODEL:-gpt-5.4}"
    ;;
  popcorn)
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://popcorn-centralus-resource.services.ai.azure.com/openai/v1}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-$(grep POPCORN_CENTRAL_AZURE_KEY .env | cut -d= -f2)}"
    export KTBENCH_MODEL="${KTBENCH_MODEL:-gpt-5.5}"
    ;;
  *)
    echo "unknown KTBENCH_AZURE='${KTBENCH_AZURE}'; expected tejas|thava|popcorn" >&2
    exit 1
    ;;
esac
export KTBENCH_MAX_TURNS="${KTBENCH_MAX_TURNS:-300}"
# Sandboxed compile_kernel can run nvcc for 30-90s on a fresh LLM-emitted
# kernel; default 60s quiescence kills the run mid-compile. Bump.
export ENSEMBLE_QUIESCENCE_MS="${ENSEMBLE_QUIESCENCE_MS:-600000}"
# Ensemble's per-turn tool-call cap defaults to 8, which is too tight
# for iterative kernel work (the model commonly burns 4-6 on compile +
# correctness before submitting). 50 gives plenty of headroom and the
# until_predicate halts the moment a submission lands anyway.
export ENSEMBLE_MAX_TOOL_TURNS="${ENSEMBLE_MAX_TOOL_TURNS:-300}"

PROBLEMS=(
  causal_conv1d_silu_a100_to_h100
  chunk_decay_scan_a100_to_h100
  fused_rms_norm_residual_a100_to_h100
  hadamard_transform_a100_to_h100
  softmax_a100_to_h100
  swiglu_activation_a100_to_h100
  wkv_recurrence_a100_to_h100
  softmax_h200_to_triton
)

run_wave() {
  local wave="$1"           # e.g. "A" or "B_judge"
  local scenario_kind="$2"  # "" for single-agent, "judge_" for multi-actor
  local logdir="/tmp/ktbench_waves/${wave}"
  mkdir -p "$logdir"

  local pids=()
  for i in "${!PROBLEMS[@]}"; do
    local gpu=$i
    local problem="${PROBLEMS[$i]}"
    local scenario="ktbench.${scenario_kind}${problem}"
    local log="${logdir}/${problem}.log"
    echo "[wave_${wave}] gpu=${gpu} ${scenario} -> ${log}"
    CUDA_VISIBLE_DEVICES=$gpu \
    KTBENCH_PROBLEM_PATH="problems/${problem}" \
    nohup ensemble run "${scenario}" \
        --world ktbench \
        --package-dir integrations/ensemble \
        --backend openai \
        --no-sync \
        > "$log" 2>&1 &
    pids+=("$!")
  done
  echo "[wave_${wave}] launched ${#pids[@]} cells. pids=${pids[*]}"
  printf '%s\n' "${pids[@]}" > "${logdir}/pids.txt"

  echo "[wave_${wave}] waiting for cells to finish..."
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  echo "[wave_${wave}] all cells done."
}

echo "==== Wave A: single-agent baselines ===="
run_wave "A" ""

echo "==== Wave B: judge (author + reviewer) ===="
run_wave "B_judge" "judge_"

echo "==== Both waves complete ===="
