#!/usr/bin/env bash
# Run the text-prompt tool-segmentation demo + GT comparison on the FULL length
# of the remaining 15 MultiBypass140 videos (no frame truncation), in parallel
# across a set of GPUs. C2V5 is excluded -- it's already been validated
# end-to-end as a standalone test run of the chunked pipeline.
#
# Job durations vary a lot across videos (~1.8h for the shortest to ~6.5h for
# the longest at ~0.4-0.5 fps), so this uses a dynamic GPU token pool instead of
# fixed round-robin batching: each GPU picks up the next queued video as soon as
# it frees up, rather than waiting for an entire batch to finish.
#
# Runs in chunked mode (fresh session every chunk_size frames, overlap-window
# mask-IoU id re-linking across chunks) by default: a single session on these
# videos (thousands of frames) reliably CUDA-OOMs from unbounded per-frame
# tracker state (`cached_frame_outputs`) growth partway through, regardless of
# total video length -- see surgical_tool_text_prompt_demo.py's
# propagate_chunked docstring.
#
# Usage:
#   scripts/run_tool_prompt_all_videos_full.sh ["<text prompt>"] [gpu_list] [chunk_size]
#
# Examples:
#   scripts/run_tool_prompt_all_videos_full.sh
#   scripts/run_tool_prompt_all_videos_full.sh "gray tool" "3 4 5 6 7"
#   scripts/run_tool_prompt_all_videos_full.sh "gray tool" "3 4 5 6 7" 300

set -euo pipefail

TEXT="${1:-gray tool}"
# Default excludes GPUs 0,1 (occupied by another job) and 2 (running the C2V5
# validation test) -- pass an explicit gpu_list to override once those free up.
read -ra GPUS <<< "${2:-3 4 5 6 7}"
CHUNK_SIZE="${3:-300}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VIDEO_BASE="data/multibypasst40_challenge_trainval/videos_cutmargin512"
# Longest videos first (by frame count) so the slowest jobs start immediately
# rather than being picked up last by a dynamic pool. C2V5 excluded (see above).
VIDEOS=(C1V7 C1V4 C1V5 C2V6 C1V6 C1V1 C2V14 C1V3 C2V1 C2V3 C2V11 C2V12 C2V2 C2V10 C2V4)

SLUG="${TEXT// /_}"
LOG_DIR="outputs/surgical_tool_demo/logs"
mkdir -p "$LOG_DIR"

echo "Running text=\"${TEXT}\" FULL-LENGTH (chunk_size=${CHUNK_SIZE}) across ${#VIDEOS[@]} videos on GPUs: ${GPUS[*]}"
echo "Logs in ${LOG_DIR}/"

# GPU token pool: a FIFO pre-loaded with one token per GPU. Each worker blocks
# reading a token, runs its job, then returns the token so another job can use
# that GPU.
POOL_FIFO="$(mktemp -u)"
mkfifo "$POOL_FIFO"
exec 3<>"$POOL_FIFO"
rm -f "$POOL_FIFO"
for gpu in "${GPUS[@]}"; do echo "$gpu" >&3; done

run_one() {
  local video="$1" gpu="$2"
  local log_file="${LOG_DIR}/${video}_${SLUG}_full.log"
  echo "[dispatch] ${video} on GPU ${gpu} -> ${log_file}"
  if scripts/run_tool_prompt_experiment.sh "$TEXT" "${VIDEO_BASE}/${video}" full "$gpu" "$CHUNK_SIZE" \
      > "$log_file" 2>&1; then
    echo "[done]     ${video} on GPU ${gpu}"
  else
    echo "[FAILED]   ${video} on GPU ${gpu} -- see ${log_file}"
  fi
}

for video in "${VIDEOS[@]}"; do
  read -u 3 gpu
  (
    run_one "$video" "$gpu"
    echo "$gpu" >&3
  ) &
done

wait
exec 3>&-
echo "All ${#VIDEOS[@]} videos done (full length). Logs in ${LOG_DIR}/"
