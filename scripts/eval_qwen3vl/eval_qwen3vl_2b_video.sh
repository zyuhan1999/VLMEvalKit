#!/bin/bash
set -ex
export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
# export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
# bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh
# export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"
cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit


# Video-MME_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps 

# torchrun --nproc-per-node=8 run.py --data Video-MME_2fps LongVideoBench_0.5fps \
#                 VideoMMMU_2fps \
#     --model Qwen3-VL-2B-Instruct --verbose --reuse --judge qwen3-30b-a3b-instruct-2507

torchrun --nproc-per-node=8 run.py --data TOMATO_2fps --model Qwen3-VL-2B-Instruct --verbose --reuse

python run.py \
    --data TOMATO_2fps \
    --model Qwen3-VL-2B-Instruct \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-instruct-2507