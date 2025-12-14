#!/bin/bash
set -ex

# export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

torchrun --nproc-per-node=8 run.py \
  --data Video-MME_medium_2fps \
  --model VideoChat3_4B_train_stage2_llava_video_academic_new_caprl2mrecap \
  --verbose \
  --reuse