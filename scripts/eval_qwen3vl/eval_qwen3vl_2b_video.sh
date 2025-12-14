#!/bin/bash
set -ex
# export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

# torchrun --nproc-per-node=8 run.py --data Video-MME_2fps LongVideoBench_0.5fps \
#                 VideoMMMU_2fps \
#     --model Qwen3-VL-2B-Instruct --verbose --reuse --judge qwen3-30b-a3b-instruct-2507

torchrun --nproc-per-node=8 run.py --data LongVideoBench_2fps --model Qwen3-VL-2B-Instruct --verbose --reuse

# torchrun --nproc-per-node=8 run.py --data Video-MME_2fps --model Qwen3-VL-2B-Instruct --verbose --reuse --judge qwen3-30b-a3b-instruct-2507

python run.py --data VideoMMMU_2fps --model Qwen3-VL-2B-Instruct --verbose --reuse --mode eval