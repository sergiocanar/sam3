"""
Convert a surgical_tool_text_prompt_demo.py `*_masks.json` output (per-frame
RLE masks/boxes/scores, keyed by original on-disk frame number) into a
standard COCO instance-segmentation JSON with `images`, `annotations`, and
`categories` top-level keys.

Each video frame becomes one COCO "image" entry, and each detected/tracked
object in that frame becomes one COCO "annotation" entry. The text prompt
used for detection becomes the single category. Track identity (needed for
downstream per-instrument triplet prediction) isn't part of standard COCO, so
it's kept as an extra `track_id` field on each annotation rather than
dropped -- COCO consumers that ignore unknown keys are unaffected.

Usage:
  python scripts/convert_masks_to_coco.py <masks_json> [<out_json>]
  python scripts/convert_masks_to_coco.py --all outputs/surgical_tool_demo
"""

import argparse
import glob
import json
import os

import pycocotools.mask as mask_utils
from PIL import Image


def convert(masks_json_path, out_path):
    with open(masks_json_path) as f:
        data = json.load(f)

    video_dir = data["video_dir"]
    text = data["text"]
    frames = data["frames"]

    # All frames were the same fixed size on disk; read one image to get it
    # rather than relying on a detection being present in some frame.
    sample_frame = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))[0]
    width, height = Image.open(sample_frame).size

    video_name = os.path.basename(os.path.normpath(video_dir))
    categories = [{"id": 1, "name": text, "supercategory": "surgical_tool"}]

    images = []
    annotations = []
    ann_id = 1
    for frame_num in sorted(int(k) for k in frames.keys()):
        file_name = f"{frame_num:06d}.jpg"
        images.append(
            {
                "id": frame_num,
                "file_name": file_name,
                "width": width,
                "height": height,
                "video_name": video_name,
            }
        )
        for obj in frames[str(frame_num)]:
            rle = obj["segmentation"]
            x_norm, y_norm, w_norm, h_norm = obj["bbox_xywh_norm"]
            bbox = [x_norm * width, y_norm * height, w_norm * width, h_norm * height]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": frame_num,
                    "category_id": 1,
                    "segmentation": {"size": rle["size"], "counts": rle["counts"]},
                    "bbox": bbox,
                    "area": float(mask_utils.area(rle)),
                    "iscrowd": 0,
                    "score": obj["prob"],
                    "track_id": obj["obj_id"],
                }
            )
            ann_id += 1

    coco = {
        "info": {"video_dir": video_dir, "text_prompt": text},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(
        f"{masks_json_path} -> {out_path} "
        f"({len(images)} images, {len(annotations)} annotations)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("masks_json", nargs="?", help="Path to a single *_masks.json file")
    parser.add_argument("out_json", nargs="?", help="Output path (default: alongside input, _coco.json suffix)")
    parser.add_argument(
        "--all",
        metavar="OUT_DIR",
        help="Convert every <OUT_DIR>/*/*_masks.json in one go instead of a single file",
    )
    args = parser.parse_args()

    if args.all:
        masks_files = sorted(glob.glob(os.path.join(args.all, "*", "*_masks.json")))
        if not masks_files:
            raise SystemExit(f"No *_masks.json files found under {args.all}")
        for masks_path in masks_files:
            out_path = masks_path.replace("_masks.json", "_coco.json")
            convert(masks_path, out_path)
    else:
        if not args.masks_json:
            raise SystemExit("Provide a masks_json path or use --all <OUT_DIR>")
        out_path = args.out_json or args.masks_json.replace("_masks.json", "_coco.json")
        convert(args.masks_json, out_path)


if __name__ == "__main__":
    main()
