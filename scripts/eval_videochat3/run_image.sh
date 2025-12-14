#!/bin/bash
set -ex

# export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

torchrun --nproc-per-node=8 run.py \
  --data RealWorldQA OCRBench ChartQA_TEST DocVQA_VAL InfoVQA_VAL \
  --model VideoChat3_4B_train_stage2_llava_video_academic_new_caprl2mrecap \
  --verbose \
  --reuse \
  --mode infer

python run.py \
  --data AI2D_TEST MMStar MMMU_DEV_VAL MathVista_MINI MathVision_MINI \
  --model VideoChat3_4B_train_stage2_llava_video_academic_new \
  --verbose \
  --reuse \
  --mode eval \
  --judge qwen3-235b-a22b-instruct-2507