"""
Compare per-frame predicted tool counts (from surgical_tool_text_prompt_demo.py)
against ground-truth instrument counts from label_files_challenge/<Video>.json.

GT count is the number of *distinct* triplet annotation rows per frame (each row is
one instrument-verb-target action), not deduplicated by instrument category -- e.g. a
frame with two triplet rows both tagged instrument_id=grasper (one instrument
retracting two different targets, or two separate graspers each tagged once)
counts as 2.

Note: the raw label files contain a substantial number of exact-duplicate rows (same
image_id, raw_id, instrument_id, verb_id, target_id repeated verbatim -- 15-25% of rows
across the 16 MultiBypass videos), which is a data-export artifact rather than a real
second instance of the same action. These are deduplicated before counting.

Usage:
  python scripts/compare_tool_counts_with_gt.py \
      --counts-json outputs/surgical_tool_demo/C2V5_tool_counts.json \
      --gt-json data/multibypasst40_challenge_trainval/label_files_challenge/C2V5.json \
      --out outputs/surgical_tool_demo/C2V5_tool_vs_gt.csv
"""

import argparse
import csv
import json
from collections import defaultdict


def load_gt_counts(gt_json_path):
    with open(gt_json_path) as f:
        gt = json.load(f)
    counts_by_image_id = defaultdict(int)
    seen_rows = set()
    for ann in gt["annotations"]:
        row_key = (
            ann["image_id"],
            ann["raw_id"],
            ann["instrument_id"],
            ann["verb_id"],
            ann["target_id"],
        )
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        counts_by_image_id[ann["image_id"]] += 1
    return dict(counts_by_image_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-json", type=str, required=True)
    parser.add_argument("--gt-json", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.counts_json) as f:
        pred_counts = json.load(f)
    gt_counts = load_gt_counts(args.gt_json)

    frame_indices = sorted(int(k) for k in pred_counts.keys())

    rows = []
    for frame_idx in frame_indices:
        pred = pred_counts[str(frame_idx)]["num_objects"]
        gt = gt_counts.get(frame_idx, 0)
        rows.append(
            {
                "frame_idx": frame_idx,
                "predicted_count": pred,
                "gt_count": gt,
                "diff": pred - gt,
            }
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "predicted_count", "gt_count", "diff"])
        writer.writeheader()
        writer.writerows(rows)

    mean_pred = sum(r["predicted_count"] for r in rows) / len(rows)
    mean_gt = sum(r["gt_count"] for r in rows) / len(rows)
    exact_match = sum(1 for r in rows if r["diff"] == 0) / len(rows)

    print(f"Compared {len(rows)} frames")
    print(f"Mean predicted count: {mean_pred:.2f}")
    print(f"Mean GT count:        {mean_gt:.2f}")
    print(f"Frames with exact count match: {exact_match * 100:.1f}%")
    print(f"Saved per-frame CSV to {args.out}")


if __name__ == "__main__":
    main()
