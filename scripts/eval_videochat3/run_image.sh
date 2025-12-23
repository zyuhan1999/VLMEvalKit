#!/bin/bash
set -ex

# export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh
export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

torchrun --nproc-per-node=8 run.py \
  --data MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL OCRBench ChartQA_TEST DocVQA_VAL InfoVQA_VAL MathVista_MINI MathVision_MINI \
  --model VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_t1 \
  --verbose \
  --reuse \
  --mode infer

python run.py \
  --data MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL MathVista_MINI MathVision_MINI \
  --model VideoChat3_4B_train_stage2_llava_video_academic \
  --verbose \
  --reuse \
  --mode eval \
  --judge qwen3-235b-a22b-instruct-2507