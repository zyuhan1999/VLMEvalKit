#!/bin/bash
set -ex

export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
# export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
# export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

# Video-MME_2fps_limit_768 LongVideoBench_2fps_limit_768 LVBench_2fps_limit_768 VideoMMMU_2fps_limit_768 MMVU_2fps_limit_768

# torchrun --nproc-per-node=8 --master_port=29601 run.py \
#   --data Qwen2.5-VL-7B-Instruct-ForVideo \
#   --model VideoChat_o3_7B_sft_600 \
#   --verbose \
#   --reuse

torchrun --nproc-per-node=8 --master_port=16666 run.py \
  --data VideoMMMU_2fps_limit_768 MMVU_2fps_limit_768 \
  --model VideoChat_o3_7B_sft_600 \
  --verbose \
  --reuse

python run.py \
    --data Video-MME_short_2fps \
    --model VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216 \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-instruct-2507