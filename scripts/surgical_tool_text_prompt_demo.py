"""
Zero-shot text-prompt tool segmentation/tracking on a surgical video with SAM 3.1
(Object Multiplex).

Runs a single text prompt (default "tool") on a directory of pre-extracted JPEG
frames, propagates through the whole video, and saves a mask-overlay MP4 for
visual review. Since the prompt is re-applied by the detector on every frame
during propagation, tools that enter/exit the scene mid-video are picked up and
dropped automatically, with no need to re-prompt.

Usage:
  python scripts/surgical_tool_text_prompt_demo.py
  python scripts/surgical_tool_text_prompt_demo.py --video-dir data/.../C1V1 --text "surgical instrument"
"""

import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from sam3.model_builder import build_sam3_multiplex_video_predictor
from sam3.visualization_utils import save_masklet_video_side_by_side
from scipy.optimize import linear_sum_assignment


def _to_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def encode_frame_objects(outputs):
    """RLE-encode masks and collect boxes/scores for one frame's present objects."""
    obj_ids = outputs["out_obj_ids"]
    boxes = outputs["out_boxes_xywh"]
    probs = outputs["out_probs"]
    masks = outputs["out_binary_masks"]
    objects = []
    for oid, box, prob, mask in zip(obj_ids, boxes, probs, masks):
        mask_np = _to_numpy(mask).astype(np.uint8)
        if not mask_np.any():
            continue
        rle = mask_utils.encode(np.asfortranarray(mask_np))
        objects.append(
            {
                "obj_id": int(oid),
                "prob": float(_to_numpy(prob)),
                # normalized [x, y, w, h] in [0, 1], relative to the video's frame size
                "bbox_xywh_norm": [float(v) for v in _to_numpy(box)],
                "segmentation": {
                    "size": rle["size"],
                    "counts": rle["counts"].decode("ascii"),
                },
            }
        )
    return objects

DEFAULT_VIDEO_DIR = (
    "data/multibypasst40_challenge_trainval/videos_cutmargin512/C2V5"
)


def load_sorted_frame_paths(video_dir):
    frame_paths = glob.glob(os.path.join(video_dir, "*.jpg"))
    frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return frame_paths


def load_blank_frame_names(blank_frames_json, video_name):
    """Return the set of blank/blown-out frame filenames flagged for this video."""
    if not blank_frames_json or not os.path.isfile(blank_frames_json):
        return set()
    with open(blank_frames_json) as f:
        data = json.load(f)
    return set(data.get("per_video", {}).get(video_name, []))


def propagate_in_video(predictor, session_id, max_frame_num_to_track=None):
    outputs_per_frame = {}
    try:
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                max_frame_num_to_track=max_frame_num_to_track,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
    except Exception:
        # Don't lose an otherwise-successful long propagation run to a late-stage
        # error (e.g. a boundary edge case in the batched grounding chunking) --
        # save whatever frames we did get and still produce a reviewable video.
        import traceback

        traceback.print_exc()
        print(
            f"\nWARNING: propagation raised an exception after "
            f"{len(outputs_per_frame)} frames; continuing with partial results."
        )
    return outputs_per_frame


def _mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union)


def _collect_obj_masks_over_window(outputs_by_offset):
    """outputs_by_offset: {offset: outputs_dict} -> {obj_id: {offset: mask}}."""
    per_obj = defaultdict(dict)
    for offset, outputs in outputs_by_offset.items():
        for oid, mask in zip(outputs["out_obj_ids"], outputs["out_binary_masks"]):
            if mask.any():
                per_obj[int(oid)][offset] = mask
    return per_obj


def _match_ids_by_overlap_iou(prev_window, new_window, iou_thresh=0.3):
    """Match this chunk's raw object ids to already-assigned global ids using
    mean mask IoU over the frames both chunks share (the overlap window).

    prev_window: {offset: outputs_dict} keyed by global id (already remapped)
    new_window: {offset: outputs_dict} keyed by this chunk's raw id
    Returns {raw_id: global_id} for pairs matched above `iou_thresh`.
    """
    prev_masks = _collect_obj_masks_over_window(prev_window)
    new_masks = _collect_obj_masks_over_window(new_window)
    prev_ids = list(prev_masks.keys())
    new_ids = list(new_masks.keys())
    if not prev_ids or not new_ids:
        return {}

    iou = np.zeros((len(new_ids), len(prev_ids)))
    for i, nid in enumerate(new_ids):
        for j, gid in enumerate(prev_ids):
            shared_offsets = new_masks[nid].keys() & prev_masks[gid].keys()
            if not shared_offsets:
                continue
            ious = [_mask_iou(new_masks[nid][o], prev_masks[gid][o]) for o in shared_offsets]
            iou[i, j] = float(np.mean(ious))

    row_idx, col_idx = linear_sum_assignment(-iou)
    return {
        new_ids[r]: prev_ids[c]
        for r, c in zip(row_idx, col_idx)
        if iou[r, c] >= iou_thresh
    }


def _empty_outputs():
    """A zero-object outputs dict, for frames where no detection is available
    (either genuinely nothing present, or a gap we can't recover)."""
    return {
        "out_obj_ids": np.zeros(0, dtype=np.int64),
        "out_binary_masks": np.zeros((0, 0, 0), dtype=bool),
        "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
        "out_probs": np.zeros(0, dtype=np.float32),
    }


def _add_prompt_with_retry(predictor, session_id, text, num_chunk_frames):
    """Call add_prompt starting at frame 0, advancing to later frames until one
    yields at least one detected object.

    The SAM2-style tracker requires a conditioning frame with >=1 detected
    object before propagate_in_video can run at all (it raises immediately
    otherwise) -- so if the text prompt happens to match nothing on a fresh
    chunk's very first frame (tool briefly occluded/out of view), prompting
    only frame 0 would crash the whole chunk. Scanning forward for the first
    frame with a hit, then propagating "both" directions from there (the
    default), recovers the earlier frames in the same chunk via backward
    propagation instead of losing them.

    Returns the local frame index used as the conditioning frame, or None if
    no frame in the whole chunk had any detection.
    """
    for frame_idx in range(num_chunk_frames):
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                text=text,
            )
        )
        if len(response["outputs"]["out_obj_ids"]) > 0:
            if frame_idx > 0:
                print(
                    f"    No '{text}' detection at chunk-local frame 0; "
                    f"first hit at frame {frame_idx}."
                )
            return frame_idx
    return None


def _remap_obj_ids(outputs, remap, next_global_id):
    """Rewrite out_obj_ids through `remap`, assigning fresh global ids (and
    growing `remap` in place) for any raw id seen for the first time.
    Returns (new_outputs, next_global_id).
    """
    new_ids = []
    for raw_id in outputs["out_obj_ids"]:
        raw_id = int(raw_id)
        if raw_id not in remap:
            remap[raw_id] = next_global_id
            next_global_id += 1
        new_ids.append(remap[raw_id])
    new_outputs = dict(outputs)
    new_outputs["out_obj_ids"] = np.array(new_ids, dtype=np.int64)
    return new_outputs, next_global_id


def propagate_chunked(predictor, frame_paths, text, chunk_size, overlap=50):
    """Propagate through the video in fixed-size, overlapping chunks, each in
    its own fresh session, instead of one session for the whole video.

    On long videos (thousands of frames) a single session both loads every
    frame onto the GPU up front and lets per-frame tracker state
    (`cached_frame_outputs` in sam3_multiplex_tracking.py) grow for the entire
    propagation with no eviction -- in practice this reliably CUDA-OOMs after
    roughly the same number of *processed* frames regardless of total video
    length. Starting a new session every `chunk_size` frames bounds both the
    resident frame batch and that unbounded per-frame state.

    A fresh session also means each chunk's detector/tracker reassigns object
    ids from scratch, which would otherwise break identity continuity across
    chunk boundaries. To preserve continuity (needed for downstream per-object
    triplet prediction), consecutive chunks overlap by `overlap` frames; object
    ids in the overlap window are matched across chunks by mean mask IoU
    (Hungarian assignment) and remapped onto a single running global id space.
    An unmatched id (IoU below threshold in the overlap window, or an object
    that first appears later in the chunk) gets a new global id.

    Each chunk needs a conditioning frame with >=1 detected object before its
    tracker can propagate at all; if the text prompt matches nothing on a
    fresh chunk's frame 0 (tool briefly occluded/out of view), we scan forward
    for the first frame that does hit and propagate both directions from
    there (see `_add_prompt_with_retry`) instead of losing the whole chunk.
    Any local frame still missing afterwards (no detection anywhere in the
    chunk) is backfilled as an explicit zero-object frame so no frame is ever
    silently absent from the output.
    """
    outputs_per_frame = {}
    num_frames = len(frame_paths)
    next_global_id = 0

    chunk_bounds = []
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        chunk_bounds.append((start, end))
        if end >= num_frames:
            break
        start = end - overlap

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_bounds):
        chunk_paths = frame_paths[chunk_start:chunk_end]
        print(f"  Chunk frames [{chunk_start}, {chunk_end}) of {num_frames}...")
        images = [Image.open(p).convert("RGB") for p in chunk_paths]
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=images,
                offload_video_to_cpu=True,
            )
        )
        session_id = response["session_id"]
        cond_frame = _add_prompt_with_retry(
            predictor, session_id, text, len(chunk_paths)
        )
        if cond_frame is None:
            print(f"    No '{text}' detections anywhere in this chunk; recording as empty.")
            chunk_outputs = {}
        else:
            chunk_outputs = propagate_in_video(predictor, session_id)
        try:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        except Exception:
            print("WARNING: close_session failed for this chunk; continuing anyway.")

        if chunk_idx == 0:
            remap = {}
            write_from = 0
        else:
            prev_window = {
                off: outputs_per_frame[chunk_start + off]
                for off in range(overlap)
                if (chunk_start + off) in outputs_per_frame
            }
            new_window = {
                off: chunk_outputs[off] for off in range(overlap) if off in chunk_outputs
            }
            remap = _match_ids_by_overlap_iou(prev_window, new_window)
            write_from = overlap  # frames < overlap are already covered by the previous chunk

        for local_idx in sorted(chunk_outputs):
            if local_idx < write_from:
                continue
            remapped_outputs, next_global_id = _remap_obj_ids(
                chunk_outputs[local_idx], remap, next_global_id
            )
            outputs_per_frame[chunk_start + local_idx] = remapped_outputs

        # A chunk with no detections anywhere (cond_frame is None above), or a
        # boundary case where forward+backward propagation still doesn't reach
        # every local index, would otherwise leave that frame silently absent
        # from the output entirely. Backfill any such gap as an explicit
        # zero-object frame instead.
        missing = [
            local_idx
            for local_idx in range(write_from, len(chunk_paths))
            if (chunk_start + local_idx) not in outputs_per_frame
        ]
        if missing:
            print(
                f"    {len(missing)} frame(s) in this chunk had no propagation "
                f"output; recording as empty (e.g. local indices {missing[:5]}...)."
            )
            for local_idx in missing:
                outputs_per_frame[chunk_start + local_idx] = _empty_outputs()
        del images
    return outputs_per_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--text", type=str, default="tool")
    parser.add_argument("--out-dir", type=str, default="outputs/surgical_tool_demo")
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Only propagate through the first N frames (default: whole video). "
        "Ignored if --chunk-size is set.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Process the whole video in fixed-size chunks, each in a fresh "
        "session, to bound GPU memory growth on long videos (see "
        "propagate_chunked docstring). Default: single session for the "
        "whole video.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Frames of overlap between consecutive chunks, used to re-link "
        "object ids across the chunk boundary by mask IoU. Only used if "
        "--chunk-size is set.",
    )
    parser.add_argument(
        "--blank-frames-json",
        type=str,
        default=(
            "/home/scanar/scanar2/endovis/models/MultiSAT_Challenge_Team_Uniandes/"
            "data_preprosses/outputs/blank_frames.json"
        ),
        help="Path to a {'per_video': {video_name: [frame_filename, ...]}} JSON "
        "of blank/blown-out frames to skip entirely. Pass '' to disable.",
    )
    parser.add_argument(
        "--batched-grounding-batch-size",
        type=int,
        default=4,
        help=(
            "Frames grounded together per detector forward pass. The detector runs "
            "at 1008x1008 regardless of source resolution, and without FlashAttention "
            "(pre-Ampere GPUs) attention memory scales ~linearly with this value; 16 "
            "(the library default) OOMs even on a 48GB GPU on pre-Ampere hardware."
        ),
    )
    args = parser.parse_args()

    video_name = os.path.basename(os.path.normpath(args.video_dir))
    slug = args.text.replace(" ", "_")
    # Each video gets its own subfolder (video + masks + counts together) rather
    # than flat files sharing one directory across all videos.
    out_dir = os.path.join(args.out_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{slug}.mp4")

    frame_paths = load_sorted_frame_paths(args.video_dir)
    blank_names = load_blank_frame_names(args.blank_frames_json, video_name)
    if blank_names:
        before = len(frame_paths)
        frame_paths = [p for p in frame_paths if os.path.basename(p) not in blank_names]
        print(
            f"Skipping {before - len(frame_paths)} blank frames (of {before}) "
            f"per {args.blank_frames_json}"
        )
    print(f"Video: {args.video_dir} ({len(frame_paths)} frames)")

    # Propagation/video-saving index frames by position in `frame_paths`. GT
    # annotations and the on-disk filenames instead key by the original frame
    # number -- these only diverge once blank frames are filtered out above, so
    # remap positions back to the original number before writing counts/masks.
    frame_index_map = [
        int(os.path.splitext(os.path.basename(p))[0]) for p in frame_paths
    ]

    print("Building SAM 3.1 multiplex predictor...")
    # FlashAttention-3 requires Hopper (compute capability 9.0); disable it on
    # older GPUs (e.g. Turing/Ampere) to avoid an unsupported-kernel crash.
    use_fa3 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (9, 0)
    predictor = build_sam3_multiplex_video_predictor(
        multiplex_count=args.multiplex_count,
        max_num_objects=args.max_num_objects,
        use_fa3=use_fa3,
        batched_grounding_batch_size=args.batched_grounding_batch_size,
    )

    start_time = time.time()
    if args.chunk_size:
        print(f"Propagating in chunks of {args.chunk_size} frames (fresh session per chunk)...")
        outputs_per_frame = propagate_chunked(
            predictor, frame_paths, args.text, args.chunk_size, overlap=args.chunk_overlap
        )
    else:
        # Note: offload_video_to_cpu is accepted here but has no effect on peak GPU
        # memory for the multiplex path today -- _construct_initial_input_batch()
        # unconditionally moves the whole video onto GPU regardless of this flag.
        # For long videos this (and unbounded tracker memory-bank growth across a
        # single session) reliably CUDA-OOMs -- use --chunk-size instead.
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=args.video_dir,
                offload_video_to_cpu=True,
            )
        )
        session_id = response["session_id"]

        print(f"Adding text prompt: '{args.text}'")
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=args.text,
            )
        )

        print("Propagating through video...")
        outputs_per_frame = propagate_in_video(
            predictor, session_id, max_frame_num_to_track=args.num_frames
        )

        try:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        except Exception:
            print("WARNING: close_session failed; continuing to save results anyway.")
    elapsed = time.time() - start_time

    unique_obj_ids = set()
    for outputs in outputs_per_frame.values():
        unique_obj_ids.update(int(oid) for oid in outputs["out_obj_ids"])

    print(
        f"\nProcessed {len(outputs_per_frame)} frames in {elapsed:.1f}s "
        f"({len(outputs_per_frame) / elapsed:.1f} fps)"
    )
    print(f"Unique object IDs seen across the video: {len(unique_obj_ids)}")

    print(f"\nSaving side-by-side (raw | overlay) video to {out_path} ...")
    save_masklet_video_side_by_side(frame_paths, outputs_per_frame, out_path, fps=args.fps)

    # Per-frame predicted object counts (only objects with a non-empty mask count
    # as "present" -- a matched-but-occluded track can have all-zero masks), for
    # later comparison against ground-truth instrument annotations.
    counts_path = os.path.join(out_dir, f"{slug}_counts.json")
    per_frame_counts = {}
    for frame_idx, outputs in outputs_per_frame.items():
        present_obj_ids = [
            int(oid)
            for oid, mask in zip(outputs["out_obj_ids"], outputs["out_binary_masks"])
            if mask.any()
        ]
        per_frame_counts[frame_index_map[int(frame_idx)]] = {
            "num_objects": len(present_obj_ids),
            "obj_ids": present_obj_ids,
        }
    with open(counts_path, "w") as f:
        json.dump(per_frame_counts, f, indent=2, sort_keys=True)
    print(f"Saved per-frame predicted counts to {counts_path}")

    # Full per-frame masks (RLE-encoded, pycocotools format) + boxes + scores, for
    # downstream use beyond just counting (e.g. as pseudo-labels or further analysis).
    masks_path = os.path.join(out_dir, f"{slug}_masks.json")
    frames_payload = {
        frame_index_map[int(frame_idx)]: encode_frame_objects(outputs)
        for frame_idx, outputs in outputs_per_frame.items()
    }
    with open(masks_path, "w") as f:
        json.dump(
            {"video_dir": args.video_dir, "text": args.text, "frames": frames_payload},
            f,
        )
    print(f"Saved per-frame masks+boxes (RLE) to {masks_path}")


if __name__ == "__main__":
    main()
