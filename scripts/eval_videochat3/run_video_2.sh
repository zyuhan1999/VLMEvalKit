#!/bin/bash
set -ex

# export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
# export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"
# bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

# Video-MME_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps

torchrun --nproc-per-node=8 --master_port=16666 run.py \
  --data VideoMMMU_2fps MMVU_2fps \
  --model VideoChat3_4B_train_stage2_llava_video_academic_t1 \
  --verbose \
  --reuse

torchrun --nproc-per-node=8 --master_port=18888 run.py \
  --data VideoMMMU_2fps MMVU_2fps \
  --model VideoChat3_4B_train_stage2_llava_video_academic_t1_fps1 \
  --verbose \
  --reuse  

torchrun --nproc-per-node=8 --master_port=18888 run.py \
  --data VideoMMMU_2fps MMVU_2fps \
  --model VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_t1 \
  --verbose \
  --reuse