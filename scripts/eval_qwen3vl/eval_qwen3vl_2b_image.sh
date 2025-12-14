#!/bin/bash
set -x
export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh
# export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"

# torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar \
#                 MMMU_DEV_VAL OCRBench ChartQA_TEST DocVQA_TEST InfoVQA_TEST \
#                 MathVista_MINI MathVerse_MINI MathVision_MINI \
#     --model Qwen3-VL-2B-Instruct --verbose --reuse --judge qwen3-30b-a3b-instruct-2507

# torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar \
#                 MMMU_DEV_VAL OCRBench ChartQA_TEST DocVQA_VAL InfoVQA_VAL \
#     --model Qwen3-VL-2B-Instruct --verbose --reuse --judge qwen3-30b-a3b-instruct-2507

torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model Qwen3-VL-2B-Instruct --verbose --reuse --mode infer

python run.py --data MathVision_MINI --model Qwen3-VL-2B-Instruct --verbose --reuse --mode eval --judge qwen3-235b-a22b-instruct-2507