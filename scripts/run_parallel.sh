#!/usr/bin/env bash
# run_parallel.sh — run KTBench across all 8 GPUs in parallel.
#
# For Azure OpenAI (key-based auth), use the openai/ prefix and point
# OPENAI_BASE_URL at your Azure endpoint:
#
#   OPENAI_API_KEY=<your-azure-key> \
#   OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/ \
#   bash scripts/run_parallel.sh --model openai/gpt55
#
# Each GPU gets its own subset of problems. Per-GPU results are written to
# results/gpu{N}.json and merged into --out at the end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

# Pin python to the conda py312 env so the script works regardless of how
# the shell was invoked (plain bash doesn't source conda init scripts).
PYTHON="${PYTHON:-/opt/conda/envs/py312/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
echo "Python: $PYTHON ($(${PYTHON} --version 2>&1))"

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="openai/gpt-5.5-priority"
OUT="results/sweep.json"
N_GPUS=8
MAX_TURNS=30
N_TIMING=20

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2";   shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    --gpus)    N_GPUS="$2";  shift 2 ;;
    --turns)   MAX_TURNS="$2"; shift 2 ;;
    --timing)  N_TIMING="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── collect all problem paths ─────────────────────────────────────────────────
mapfile -t ALL_PROBLEMS < <($PYTHON -c "
import sys; sys.path.insert(0, 'src')
from ktbench.dataset import load_all_problems
for p in load_all_problems():
    print(p.problem_dir)
")

N_PROBLEMS=${#ALL_PROBLEMS[@]}
if [[ $N_PROBLEMS -eq 0 ]]; then
  echo "No problems found. Is configs/problems.toml populated?" >&2
  exit 1
fi

echo "Problems : $N_PROBLEMS"
echo "GPUs     : $N_GPUS"
echo "Model    : $MODEL"
echo "Output   : $OUT"
echo ""

mkdir -p results results/gpu_logs

# ── distribute problems round-robin across GPUs ───────────────────────────────
declare -A GPU_PROBLEMS
for i in $(seq 0 $((N_GPUS - 1))); do
  GPU_PROBLEMS[$i]=""
done

for i in "${!ALL_PROBLEMS[@]}"; do
  gpu=$(( i % N_GPUS ))
  GPU_PROBLEMS[$gpu]+="${ALL_PROBLEMS[$i]} "
done

# ── launch one worker per GPU ─────────────────────────────────────────────────
PIDS=()
GPU_OUTS=()

for gpu in $(seq 0 $((N_GPUS - 1))); do
  problems_str="${GPU_PROBLEMS[$gpu]:-}"
  if [[ -z "${problems_str// /}" ]]; then
    continue  # no problems assigned to this GPU
  fi

  gpu_out="results/gpu${gpu}.json"
  log_file="results/gpu_logs/gpu${gpu}.log"
  GPU_OUTS+=("$gpu_out")

  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/run_sweep.py \
    --problems $problems_str \
    --models "$MODEL" \
    --device 0 \
    --max-turns "$MAX_TURNS" \
    --n-timing "$N_TIMING" \
    --log-dir "results/inspect_logs/gpu${gpu}" \
    --out "$gpu_out" \
    > "$log_file" 2>&1 &

  pid=$!
  PIDS+=($pid)
  echo "GPU $gpu  PID $pid  $(echo $problems_str | wc -w | tr -d ' ') problem(s)  → $gpu_out"
  echo "         log: $log_file"
done

echo ""
echo "All workers launched. Waiting…"
echo ""

# ── wait and collect exit codes ───────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  log_file="results/gpu_logs/gpu${i}.log"
  if wait "$pid"; then
    echo "GPU $i  PID $pid  ✓"
  else
    echo "GPU $i  PID $pid  ✗ (exit $?)"
    echo "--- GPU $i log (${log_file}) ---"
    cat "$log_file"
    echo "--- end GPU $i log ---"
    FAILED=1
  fi
done

echo ""

# ── merge per-GPU JSON results into one file ──────────────────────────────────
$PYTHON - "$OUT" <<'PYEOF'
import json, sys, glob, pathlib

out_path = pathlib.Path(sys.argv[1])
gpu_files = sorted(glob.glob("results/gpu*.json"))
if not gpu_files:
    print("No per-GPU result files found.")
    sys.exit(0)

merged = []
for f in gpu_files:
    try:
        data = json.loads(pathlib.Path(f).read_text())
        if isinstance(data, list):
            merged.extend(data)
        else:
            print(f"Warning: {f} is not a list, skipping")
    except Exception as e:
        print(f"Warning: could not read {f}: {e}")

merged.sort(key=lambda r: (-r.get("final_score", 0), r.get("model", ""), r.get("problem", "")))

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(merged, indent=2))
print(f"Merged {len(merged)} results → {out_path}")

print()
header = f"{'MODEL':<35}  {'PROBLEM':<40}  {'SCORE':>6}"
print(header)
print("-" * len(header))
for r in merged:
    print(f"{r.get('model',''):<35}  {r.get('problem',''):<40}  {r.get('final_score',0):>6.4f}")
PYEOF

[[ $FAILED -eq 0 ]] && echo "" && echo "Done → $OUT"
exit $FAILED
