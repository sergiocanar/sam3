#!/usr/bin/env bash
# Run the text-prompt tool-segmentation demo + GT comparison across all 16
# MultiBypass140 videos, in parallel across a set of GPUs.
#
# Usage:
#   scripts/run_tool_prompt_all_videos.sh ["<text prompt>"] [num_frames] [gpu_list]
#
# Examples:
#   scripts/run_tool_prompt_all_videos.sh
#   scripts/run_tool_prompt_all_videos.sh "gray tool" 300 "1 2 3 4 5 6 7"

set -euo pipefail

TEXT="${1:-gray tool}"
NUM_FRAMES="${2:-300}"
read -ra GPUS <<< "${3:-1 2 3 4 5 6 7}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VIDEO_BASE="data/multibypasst40_challenge_trainval/videos_cutmargin512"
VIDEOS=(C1V1 C1V3 C1V4 C1V5 C1V6 C1V7 C2V1 C2V2 C2V3 C2V4 C2V5 C2V6 C2V10 C2V11 C2V12 C2V14)

SLUG="${TEXT// /_}"
LOG_DIR="outputs/surgical_tool_demo/logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=${#GPUS[@]}
echo "Running text=\"${TEXT}\" num_frames=${NUM_FRAMES} across ${#VIDEOS[@]} videos on GPUs: ${GPUS[*]}"

i=0
for video in "${VIDEOS[@]}"; do
  gpu="${GPUS[$((i % NUM_GPUS))]}"
  log_file="${LOG_DIR}/${video}_${SLUG}.log"
  echo "[$((i + 1))/${#VIDEOS[@]}] dispatching ${video} on GPU ${gpu} -> ${log_file}"
  scripts/run_tool_prompt_experiment.sh "$TEXT" "${VIDEO_BASE}/${video}" "$NUM_FRAMES" "$gpu" \
    > "$log_file" 2>&1 &
  i=$((i + 1))

  # throttle: once we've dispatched one job per GPU, wait for that batch to finish
  if (( i % NUM_GPUS == 0 )); then
    wait
  fi
done

wait
echo "All ${#VIDEOS[@]} videos done. Logs in ${LOG_DIR}/"
