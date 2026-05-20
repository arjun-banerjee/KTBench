#!/usr/bin/env bash
# Wave A: GPT-5.5 baseline across all 8 KTBench problems, one per GPU.
#
# Each cell runs as a separate `ensemble run` process pinned to one
# GPU via CUDA_VISIBLE_DEVICES. Logs land under /tmp/ktbench_wave_a/.
#
# Assumes the Azure proxy is already running on :8088 (see
# scripts/azure_openai_proxy.py).

set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source .venv/bin/activate

export OPENAI_BASE_URL="http://localhost:8088"
export OPENAI_API_KEY="$(grep TEJAS_AZURE_KEY .env | cut -d= -f2)"
export KTBENCH_MODEL="${KTBENCH_MODEL:-gpt-5.5}"
export KTBENCH_MAX_TURNS="${KTBENCH_MAX_TURNS:-20}"

mkdir -p /tmp/ktbench_wave_a traces

# (gpu, problem) cells. One per GPU; problem→scenario name is identical.
declare -a CELLS=(
  "0 causal_conv1d_silu_a100_to_h100"
  "1 chunk_decay_scan_a100_to_h100"
  "2 fused_rms_norm_residual_a100_to_h100"
  "3 hadamard_transform_a100_to_h100"
  "4 softmax_a100_to_h100"
  "5 swiglu_activation_a100_to_h100"
  "6 wkv_recurrence_a100_to_h100"
  "7 softmax_h200_to_triton"
)

declare -a PIDS=()
for cell in "${CELLS[@]}"; do
  gpu=${cell%% *}
  problem=${cell#* }
  log=/tmp/ktbench_wave_a/${problem}.log
  echo "[wave_a] gpu=${gpu} problem=${problem} -> ${log}"
  CUDA_VISIBLE_DEVICES=$gpu \
  KTBENCH_PROBLEM_PATH=problems/${problem} \
  nohup python -m ensemble.cli_run \
    --scenario ktbench.${problem} \
    --world ktbench \
    --package-dir integrations/ensemble \
    --backend openai \
    > "$log" 2>&1 &
  PIDS+=("$!")
done

echo "[wave_a] launched ${#PIDS[@]} cells. pids=${PIDS[*]}"
echo "[wave_a] tail logs: tail -f /tmp/ktbench_wave_a/*.log"
echo "[wave_a] traces: traces/ktbench_*.jsonl"
echo "${PIDS[@]}" > /tmp/ktbench_wave_a/pids.txt
