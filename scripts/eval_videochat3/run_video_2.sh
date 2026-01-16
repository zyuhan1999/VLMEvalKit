#!/bin/bash
set -ex

# export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
export LMUData="/root/s3/videogpu/zhuyuhan/LMUData"
# export LOCAL_LLM="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/Qwen3-235B-A22B-Instruct-2507"
# bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

# Video-MME_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps

python run.py \
    --data Video-MME_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps \
    --model VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl2 \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-instruct-2507

python run.py \
    --data OCRBench ChartQA_TEST DocVQA_VAL InfoVQA_VAL \
    --model VideoChat3_4B_train_stage2_llava_video \
    --verbose --reuse --mode eval