#!/bin/bash
set -ex
# export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh
# export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"
cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

# Use local HF-format checkpoints on GPFS (overrides hardcoded /root/s3 paths in vlmeval/config.py)
# export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf"


# Video-MME_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps 

# torchrun --nproc-per-node=8 run.py --data Video-MME_2fps LongVideoBench_0.5fps \
#                 VideoMMMU_2fps \
#     --model Qwen3-VL-2B-Instruct --verbose --reuse --judge qwen3-30b-a3b-instruct-2507

# torchrun --nproc-per-node=8 run.py --data TOMATO_2fps --model Qwen3-VL-2B-Instruct --verbose --reuse

torchrun --nproc-per-node=8 --master_port=16666 run.py \
  --data MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL OCRBench ChartQA_TEST DocVQA_VAL InfoVQA_VAL MathVista_MINI MathVision_MINI Video-MME_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps \
  --model Qwen3-VL-4B-Instruct \
  --verbose \
  --reuse \
  --mode infer

# python run.py \
#     --data TOMATO_2fps \
#     --model Qwen3-VL-2B-Instruct \
#     --verbose --reuse --mode eval \
#     --judge qwen3-235b-a22b-instruct-2507



# =========================
# Newly added benchmarks
# =========================
export MINERVA_ROOT="/root/s3/videogpu/zhuyuhan/benchmarks/Minerva"
export TIMELENS_BENCH_ROOT="/root/s3/videogpu/zhuyuhan/benchmarks/TimeLens/TimeLens-Bench"
export MOTIONBENCH_ROOT="/root/s3/videogpu/zhuyuhan/benchmarks/MotionBench"
export VUE_TR_ROOT="/root/s3/videogpu/zhuyuhan/benchmarks/vidi/VUE_TR"

# # --- Minerva (Video-MCQ) ---
# torchrun --nproc-per-node=8 run.py \
#     --data Minerva_2fps \
#     --model Qwen3-VL-2B-Instruct \
#     --verbose --reuse
#
# --- Minerva (Video-MCQ) + LLM Judge (qwen3-235b-a22b-instruct-2507) ---
# NOTE: `--judge` is used in evaluation; if you haven't run inference yet, run the block above first (or use --mode all).
# python run.py \
#     --data Minerva_2fps \
#     --model Qwen3-VL-2B-Instruct \
#     --verbose --reuse --mode eval \
#     --judge qwen3-235b-a22b-instruct-2507

# # --- TimeLens (Video Temporal Grounding) ---
# torchrun --nproc-per-node=8 run.py \
#     --data TimeLens_2fps \
#     --model Qwen3-VL-2B-Instruct \
#     --verbose --reuse

# python run.py \
#     --data TimeLens_2fps \
#     --model Qwen3-VL-2B-Instruct \
#     --verbose --reuse --mode eval \
#     --judge qwen3-235b-a22b-instruct-2507

# # # --- MotionBench (Video-MCQ) ---
# # torchrun --nproc-per-node=8 run.py \
# #     --data MotionBench_2fps \
# #     --model Qwen3-VL-2B-Instruct \
# #     --verbose --reuse

# python run.py \
#     --data MotionBench_2fps \
#     --model Qwen3-VL-2B-Instruct \
#     --verbose --reuse --mode eval \
#     --judge qwen3-235b-a22b-instruct-2507


# Stop vLLM judge server before running other benchmarks.
# cleanup_vllm
# trap - EXIT

# --- VUE_TR (Video Temporal Retrieval) ---
torchrun --nproc-per-node=8 run.py \
    --data VUE_TR_2fps \
    --model Qwen3-VL-2B-Instruct \
    --verbose --reuse

python run.py \
    --data VUE_TR_2fps \
    --model Qwen3-VL-2B-Instruct \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-instruct-2507

# torchrun --nproc-per-node=8 run.py \
#     --data MotionBench_2fps \
#     --model Qwen3-VL-2B-Instruct \
#     --verbose --reuse
