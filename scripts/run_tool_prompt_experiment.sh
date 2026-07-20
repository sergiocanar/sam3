#!/usr/bin/env bash
# Run the surgical-tool text-prompt demo for a given prompt, then compare its
# per-frame predicted counts against ground truth.
#
# Usage:
#   scripts/run_tool_prompt_experiment.sh "<text prompt>" [video_dir] [num_frames] [cuda_device] [chunk_size]
#
# num_frames may be "full" (or empty) to propagate through the entire video instead
# of truncating. Ignored if chunk_size is set (chunked mode always covers the whole
# video -- see surgical_tool_text_prompt_demo.py's propagate_chunked docstring for why:
# a single session on a long video reliably CUDA-OOMs from unbounded tracker memory-bank
# growth, so chunk_size resets that state every N frames via a fresh session per chunk).
#
# Examples:
#   scripts/run_tool_prompt_experiment.sh "tool"
#   scripts/run_tool_prompt_experiment.sh "surgical metal instrument"
#   scripts/run_tool_prompt_experiment.sh "surgical metal instrument" \
#       data/multibypasst40_challenge_trainval/videos_cutmargin512/C1V1 300 0
#   scripts/run_tool_prompt_experiment.sh "gray tool" \
#       data/multibypasst40_challenge_trainval/videos_cutmargin512/C1V1 full 0
#   scripts/run_tool_prompt_experiment.sh "gray tool" \
#       data/multibypasst40_challenge_trainval/videos_cutmargin512/C1V1 full 0 300

set -euo pipefail

TEXT="${1:?usage: $0 \"<text prompt>\" [video_dir] [num_frames] [cuda_device] [chunk_size]}"
VIDEO_DIR="${2:-data/multibypasst40_challenge_trainval/videos_cutmargin512/C2V5}"
NUM_FRAMES="${3:-300}"
CUDA_DEVICE="${4:-1}"
CHUNK_SIZE="${5:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="outputs/surgical_tool_demo"
VIDEO_NAME="$(basename "$VIDEO_DIR")"
SLUG="${TEXT// /_}"
GT_JSON="data/multibypasst40_challenge_trainval/label_files_challenge/${VIDEO_NAME}.json"

DEMO_ARGS=()
if [[ -n "$CHUNK_SIZE" ]]; then
  DEMO_ARGS=(--chunk-size "$CHUNK_SIZE")
elif [[ -n "$NUM_FRAMES" && "$NUM_FRAMES" != "full" ]]; then
  DEMO_ARGS=(--num-frames "$NUM_FRAMES")
fi

echo "== Running demo: video=${VIDEO_NAME} text=\"${TEXT}\" num_frames=${NUM_FRAMES} chunk_size=${CHUNK_SIZE:-none} gpu=${CUDA_DEVICE} =="
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/surgical_tool_text_prompt_demo.py \
  --video-dir "$VIDEO_DIR" \
  --text "$TEXT" \
  "${DEMO_ARGS[@]}"

VIDEO_OUT_DIR="${OUT_DIR}/${VIDEO_NAME}"
COUNTS_JSON="${VIDEO_OUT_DIR}/${SLUG}_counts.json"
COMPARE_CSV="${VIDEO_OUT_DIR}/${SLUG}_vs_gt.csv"

if [[ ! -f "$GT_JSON" ]]; then
  echo "No ground-truth label file at ${GT_JSON} -- skipping comparison."
  exit 0
fi

echo "== Comparing against ground truth: ${GT_JSON} =="
python3 scripts/compare_tool_counts_with_gt.py \
  --counts-json "$COUNTS_JSON" \
  --gt-json "$GT_JSON" \
  --out "$COMPARE_CSV"
