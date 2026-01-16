#!/bin/bash
set -ex

cd /mnt/petrelfs/zhuyuhan/workspace/VLMEvalKit

# MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL

export LOCAL_LLM="qwen3-235b"
srun -p videoop --gres=gpu:0 --quotatype=spot \
  python run.py \
    --data MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL Video-MME_short_2fps MMVU_2fps \
    --model VideoChat3_4B_train_stage2_bee_image_minisft_lr5e-5_vtlr2e-5 \
    --verbose --reuse --mode eval \
    --judge qwen3-235b-a22b-thinking-2507

python run.py \
  --data OCRBench ChartQA_TEST DocVQA_VAL InfoVQA_VAL \
  --model VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl2 \
  --verbose \
  --reuse \
  --mode eval