#!/bin/bash
set -ex

export LMUData="/nvme/zhuyuhan/videogpu/zhuyuhan/LMUData"

# Video-MME_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps
export LOCAL_LLM="qwen3-235b"
srun -p videoop --gres=gpu:0 --quotatype=spot \
  python run.py \
    --data Video-MME_short_2fps MMVU_2fps \
    --model VideoChat3_4B_train_stage2_image_video_minisft_v1_lr8e-5_vtlr2e-5 \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-thinking-2507

srun -p videoop --gres=gpu:0 --quotatype=spot \
  python run.py \
    --data Video-MME_short_2fps MMVU_2fps \
    --model VideoChat3_4B_train_stage2_beedata_minisft_lr8e-5_vtlr2e-5 \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-thinking-2507

srun -p videoop --gres=gpu:0 --quotatype=spot \
  python run.py \
    --data Video-MME_short_2fps MMVU_2fps \
    --model VideoChat3_4B_train_stage2_bee_image_minisft_lr5e-5_vtlr2e-5 \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-thinking-2507
