<a id="readme-top"></a>

<div align="center">

# VLMEvalKit for VideoChat3 & TimeLens2

**Reproducible evaluation for image, video, and temporal-grounding models.**

[![License](https://img.shields.io/github/license/zyuhan1999/VLMEvalKit?style=flat-square)](LICENSE)
[![Models](https://img.shields.io/badge/🤗%20Models-MCG--NJU-ffd21e?style=flat-square)](https://huggingface.co/MCG-NJU)
[![Upstream](https://img.shields.io/badge/Based%20on-VLMEvalKit-5b67d6?style=flat-square)](https://github.com/open-compass/VLMEvalKit)

[Models](#models) · [Quick start](#quick-start) · [TimeLens data](#timelens-data) ·
[Evaluation](#evaluation) · [Results](#results) · [Citation](#citation)

</div>

This repository is a focused extension of
[OpenCompass VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for evaluating
[VideoChat3-4B](https://huggingface.co/MCG-NJU/VideoChat3-4B) and the
[TimeLens2](https://huggingface.co/collections/MCG-NJU/timelens2) model family.
It keeps the familiar one-command VLMEvalKit workflow while adding temporal-grounding
datasets, metrics, frame caching, and public evaluation recipes.

## Highlights

- Four Hugging Face checkpoints registered under stable CLI names.
- Image, video QA, and video temporal-grounding evaluation in one runner.
- Native TimeLens metrics: **mIoU** and **Recall@1** at IoU 0.3, 0.5, and 0.7.
- Configurable frame sampling and reusable pre-extracted frame caches.
- Single-GPU, automatic multi-GPU placement, and `torchrun` data parallelism.
- No private cluster setup is required by the documented workflows.

## Models

| CLI name | Hugging Face checkpoint | Implementation |
| --- | --- | --- |
| `VideoChat3-4B` | [`MCG-NJU/VideoChat3-4B`](https://huggingface.co/MCG-NJU/VideoChat3-4B) | VideoChat3 |
| `TimeLens2-2B` | [`MCG-NJU/TimeLens2-2B`](https://huggingface.co/MCG-NJU/TimeLens2-2B) | Qwen3-VL |
| `TimeLens2-4B` | [`MCG-NJU/TimeLens2-4B`](https://huggingface.co/MCG-NJU/TimeLens2-4B) | Qwen3-VL |
| `TimeLens2-8B` | [`MCG-NJU/TimeLens2-8B`](https://huggingface.co/MCG-NJU/TimeLens2-8B) | Qwen3-VL |

Model weights are downloaded from Hugging Face on first use and cached by
`huggingface_hub`.

## Quick start

### 1. Install

Recommended environment:

- Linux
- Python 3.10+
- A CUDA-capable GPU with a compatible PyTorch installation
- FFmpeg/`ffprobe` for video metadata and decoding

```bash
git clone https://github.com/zyuhan1999/VLMEvalKit.git
cd VLMEvalKit

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

VideoChat3 uses FlashAttention 2:

```bash
python -m pip install flash-attn --no-build-isolation
```

Use a recent `transformers` release with Qwen3-VL support. Match PyTorch,
FlashAttention, and CUDA versions to the driver installed on your machine.

### 2. Configure data

`LMUData` is the shared data and cache directory used by VLMEvalKit:

```bash
export LMUData=/absolute/path/to/LMUData
```

You may place environment variables in a local `.env` file instead. `.env` files
are ignored by Git; never commit API keys or storage credentials.

### 3. Run an evaluation

Evaluate VideoChat3-4B on Video-MME:

```bash
python run.py \
  --model VideoChat3-4B \
  --data Video-MME_2fps \
  --mode all \
  --work-dir outputs
```

Evaluate TimeLens2-4B on all three TimeLens subsets:

```bash
python run.py \
  --model TimeLens2-4B \
  --data TimeLens_Charades_2fps TimeLens_ActivityNet_2fps TimeLens_QVHighlights_2fps \
  --mode all \
  --work-dir outputs
```

Replace the model name with `TimeLens2-2B` or `TimeLens2-8B` to evaluate another
checkpoint.

## TimeLens data

Set `TIMELENS_BENCH_ROOT` to the extracted benchmark directory:

```bash
export TIMELENS_BENCH_ROOT=/absolute/path/to/TimeLens-Bench
```

Expected layout:

```text
TimeLens-Bench/
├── activitynet-timelens.json
├── charades-timelens.json
├── qvhighlights-timelens.json
└── videos/
    ├── activitynet/
    │   └── *.mp4
    ├── charades/
    │   └── *.mp4
    └── qvhighlights/
        └── *.mp4
```

The first run creates VLMEvalKit metadata and frame-cache files beside the
annotations and under `LMUData`. Ensure both locations are writable.

To pre-extract frames before evaluation:

```bash
bash scripts/pre_extract_video_frames/extract_video_frames.sh \
  --dataset TimeLens_Charades_2fps
```

Frame availability is checked by default. If a complete cache already exists,
you can skip the check with `--check-extracted-frames false`.

## Evaluation

### Public helper scripts

```bash
# TimeLens2; MODEL_NAME defaults to TimeLens2-4B
MODEL_NAME=TimeLens2-4B \
  bash scripts/eval_qwen3vl/eval_timelens2.sh TimeLens_Charades_2fps

# VideoChat3; MODEL_NAME defaults to VideoChat3-4B
bash scripts/eval_videochat3/eval_4b.sh Video-MME_2fps
```

Both scripts require `LMUData` and accept one or more registered dataset names.
Set `MODE=infer`, `MODE=eval`, or `MODE=all` to control the stage.

### Useful runner options

| Option | Meaning |
| --- | --- |
| `--mode all` | Run inference followed by evaluation. |
| `--mode infer` | Generate predictions only. |
| `--mode eval` | Evaluate existing predictions only. |
| `--reuse` | Reuse the latest compatible prediction/intermediate files. |
| `--work-dir PATH` | Write outputs under a custom directory. |
| `--retry N` | Set API-model and judge retry count; the default is `5`. |
| `--api-nproc N` | Set API request concurrency. |
| `--check-extracted-frames false` | Trust an existing frame cache without checking it. |

For long model responses, avoid Excel cell-length limits:

```bash
export PRED_FORMAT=tsv
```

For benchmarks requiring an API judge, configure the provider credentials through
environment variables or `.env`, then pass the appropriate `--judge` value. See the
[upstream quick-start guide](docs/en/Quickstart.md) for provider-specific settings.

### Multiple GPUs

A single process can use all visible GPUs when the model adapter supports automatic
device placement:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --model TimeLens2-8B \
  --data TimeLens_Charades_2fps \
  --mode all
```

For data-parallel evaluation, launch multiple processes:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py \
  --model TimeLens2-4B \
  --data TimeLens_Charades_2fps \
  --mode all
```

## Benchmarks

This fork adds first-class support for the three TimeLens temporal-grounding
subsets and retains VLMEvalKit's broader image/video benchmark registry.

Common video entries include:

- `TimeLens_Charades_*`, `TimeLens_ActivityNet_*`, `TimeLens_QVHighlights_*`
- `Video-MME_*`, `LongVideoBench_*`, `LVBench_*`, `VideoMMMU_*`
- `TOMATO_*`, `MMVU_*`, `MotionBench_*`, `TVBench_*`

Sampling rate, frame limit, resolution, and reasoning variants are encoded in the
registered dataset name. See
[`vlmeval/dataset/video_dataset_config.py`](vlmeval/dataset/video_dataset_config.py)
for the complete list and [`vlmeval/config.py`](vlmeval/config.py) for all registered
models.

## Results

Results are written below:

```text
<work-dir>/<model>/<run-id>/
```

Prediction files contain per-sample model outputs; score/rating files contain the
benchmark metrics. To aggregate temporal-grounding results:

```bash
python print_grounding_result.py outputs/TimeLens2-4B
```

To build a broader benchmark summary:

```bash
python print_result.py outputs/TimeLens2-4B
```

Both commands write `summary.csv` inside the supplied model output directory.

## Repository layout

```text
run.py                                  Main evaluation entry point
vlmeval/config.py                       Model registry
vlmeval/vlm/videochat3/                 VideoChat3 adapter and prompts
vlmeval/dataset/timelens.py             TimeLens loading and metrics
vlmeval/dataset/video_dataset_config.py Video benchmark registry
scripts/                                Portable evaluation and analysis helpers
```

For custom models and datasets, refer to the
[VLMEvalKit development guide](docs/en/Development.md).

## Troubleshooting

- **CUDA out of memory:** use a smaller model, reduce the frame limit/resolution
  through a registered dataset variant, or expose more GPUs.
- **FlashAttention import error:** install a FlashAttention build compatible with
  the active PyTorch and CUDA versions.
- **Missing TimeLens annotation/video:** verify `TIMELENS_BENCH_ROOT` and the layout
  above.
- **Truncated long answers:** set `PRED_FORMAT=tsv` before inference.
- **Interrupted run:** repeat the command with `--reuse`.

## Citation

Please cite the relevant model and benchmark publications listed on their Hugging
Face model cards. If you use VLMEvalKit, cite:

```bibtex
@inproceedings{duan2024vlmevalkit,
  title     = {VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models},
  author    = {Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and
               Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and
               Zhang, Pan and Wang, Jiaqi and others},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  pages     = {11198--11201},
  year      = {2024}
}
```

## Acknowledgements

This project builds on
[OpenCompass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank its
authors, benchmark maintainers, and the open-source model community.

## License

The code in this repository is released under the [Apache License 2.0](LICENSE).
Model weights and datasets may use separate licenses; review their respective terms
before use.

<div align="right"><a href="#readme-top">Back to top ↑</a></div>
