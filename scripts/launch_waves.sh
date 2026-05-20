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

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(grep TEJAS_AZURE_KEY .env | cut -d= -f2)}"
export KTBENCH_MODEL="${KTBENCH_MODEL:-gpt-5.5}"
export KTBENCH_MAX_TURNS="${KTBENCH_MAX_TURNS:-80}"
# Sandboxed compile_kernel can run nvcc for 30-90s on a fresh LLM-emitted
# kernel; default 60s quiescence kills the run mid-compile. Bump.
export ENSEMBLE_QUIESCENCE_MS="${ENSEMBLE_QUIESCENCE_MS:-600000}"

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
  local scenario_prefix="$2" # e.g. "ktbench" or "ktbench.judge"
  local logdir="/tmp/ktbench_waves/${wave}"
  mkdir -p "$logdir"

  local pids=()
  for i in "${!PROBLEMS[@]}"; do
    local gpu=$i
    local problem="${PROBLEMS[$i]}"
    local scenario="${scenario_prefix}.${problem}"
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
run_wave "A" "ktbench"

echo "==== Wave B: judge (author + reviewer) ===="
run_wave "B_judge" "ktbench.judge"

echo "==== Both waves complete ===="
