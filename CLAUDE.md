# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SAM 3 ("Segment Anything with Concepts"): Meta's foundation model for promptable image/video
segmentation. Given a text phrase or visual exemplar (points/boxes/masks), it detects, segments,
and tracks all matching object instances. Python package `sam3`, published as
`facebookresearch/sam3`. Requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+ GPU. Checkpoints are
gated on Hugging Face (`facebook/sam3`, `facebook/sam3.1`) and must be pulled via `hf auth login`.

## Setup & commands

```bash
# Base install (inference only)
pip install -e .

# Add notebook deps (for examples/*.ipynb) or dev/train deps
pip install -e ".[notebooks]"
pip install -e ".[dev,train]"

# Format (required to pass CI — see .github/workflows/format.yml)
ufmt format .

# Tests
pytest test/                       # unit tests (small, e.g. io_utils routing logic)
python sam3/perflib/tests/tests.py # perflib (triton/fused-op) correctness tests
```

Note: `pyproject.toml`'s `[tool.pytest.ini_options]` sets `testpaths = ["tests"]`, but the actual
directory is `test/` (singular) — invoke pytest with an explicit path (`pytest test/`) rather than
bare `pytest`.

Training / evaluation both go through the same Hydra entrypoint:

```bash
# Finetune on a custom dataset (local)
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Finetune on a SLURM cluster
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1 --partition <p> --account <a> --qos <q> --num-gpus 8 --num-nodes 2

# Evaluation is the same script with trainer.mode=val set in the config
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_eval.yaml
```

Configs live in `sam3/train/configs/` (Hydra YAML): `gold_image_evals/`, `silver_image_evals/`,
`saco_video_evals/`, `roboflow_v100/`, `odinw13/`, plus shared `eval_base.yaml` (edit dataset/
checkpoint paths here before running an eval). See `README_TRAIN.md` for job-array sweeps and
ODinW13 few-shot reproduction details, and `scripts/eval/{gold,silver,veval}/README.md` for
per-benchmark data prep/download instructions before running eval configs.

## Architecture

**Two coupled subsystems sharing a vision encoder:** a DETR-style *detector* conditioned on text,
geometry, and image exemplars, and a SAM-2-style *tracker* (transformer encoder-decoder) that
handles video segmentation and interactive refinement. The decoupled detector/tracker design plus
a "presence token" (for discriminating similar prompts, e.g. "player in white" vs "player in red")
are the key architectural departures from SAM 2. All model assembly happens in
`sam3/model_builder.py` — it wires together the pieces from `sam3/model/` (vit backbone/neck,
vision-language combiner, transformer encoder/decoder, segmentation head, geometry encoder,
tracker mask-memory backbone) and `sam3/sam/` (SAM1/2-style prompt encoder, mask decoder,
transformer — the interactive point/box/mask prompting machinery). Entry points to use, not
reimplement: `build_sam3_image_model`, `build_sam3_video_predictor`,
`build_sam3_multiplex_video_predictor` (and `build_sam3_predictor` as a general dispatcher).

**Object Multiplex (SAM 3.1):** the original video pipeline tracks each object independently
(cost scales linearly with object count). `sam3_multiplex_*.py` files implement a shared-memory
scheme that buckets objects and processes them jointly for large speedups at high object counts —
see `RELEASE_SAM3p1.md` and Appendix H of the paper for the design rationale before touching this
code path.

**Predictor / processor layer** (what user code actually calls):
`Sam3Processor` (`sam3/model/sam3_image_processor.py`) for images;
`Sam3VideoPredictor` / `Sam3VideoPredictorMultiGPU` (`sam3/model/sam3_video_predictor.py`) for
video, driven by a `handle_request(request=dict(type=..., ...))` session API
(`start_session`, `add_prompt`, etc.) rather than direct method calls.

**Agent (`sam3/agent/`):** an LLM-orchestrated wrapper (`agent_core.py`, `client_llm.py` for the
LLM, `client_sam3.py` for the SAM3 model) that decomposes complex/compositional text prompts into
iterative SAM3 calls; prompts live in `system_prompts/`, drawing/geometry helpers in `helpers/`.

**Eval (`sam3/eval/`):** cgF1 (`cgf1_eval.py`) is the official metric for the SA-Co benchmarks
(Gold = images, multi-annotator oracle eval; Silver = images, single-annotator; VEval = video).
COCO-style eval (`coco_eval*.py`) and video tracking eval (`hota_eval_toolkit/`,
`teta_eval_toolkit/`, `ytvis_eval.py`) support the public-benchmark comparisons in the README
tables. `scripts/eval/{gold,silver,veval}/README.md` document per-domain data download/prep
scripts that must run before the corresponding `sam3/train/configs/*_evals/*.yaml` configs.

**Training (`sam3/train/`):** `train.py` is the CLI entrypoint (local `single_node_runner` or
`submitit`-based SLURM `SubmititRunner`); `trainer.py`'s `Trainer` class owns the actual loop
(optim/checkpoint/logging config dataclasses at the top of the file). `data/`, `loss/`, `optim/`,
`transforms/`, `utils/` are the supporting pieces referenced from the Hydra configs.

**perflib (`sam3/perflib/`):** custom fused/Triton kernels (NMS, connected components, IoU,
detector-tracker association) used to keep inference fast; `triton/` holds the raw kernels,
`fa3.py` wraps FlashAttention-3 (optional dep, see README "faster inference" install step).
