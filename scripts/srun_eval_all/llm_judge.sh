#!/bin/bash
set -x

cd /mnt/petrelfs/zhuyuhan/workspace/VLMEvalKit

########################################################################################

MODEL="VideoChat3_4B_train_stage2_soha_v1_image_video_minisft_v1_lr8e-5_vtlr2e-5"

########################################################################################

# DATASETS=(MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL Video-MME_short_2fps MMVU_2fps)
# DATASETS=(MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL MathVista_MINI MathVision_MINI Video-MME_short_2fps Video-MME_medium_2fps Video-MME_long_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps VUE_TR_1fps)
# DATASETS=(MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL MathVista_MINI MathVision_MINI Video-MME_short_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps VUE_TR_1fps)
DATASETS=(MMBench_DEV_EN_V11)

for DATASET in "${DATASETS[@]}"; do
  echo "Running inference on dataset: ${DATASET}"
  srun -p videoop --gres=gpu:0 --quotatype=spot --job-name=${DATASET}_llm_judge bash scripts/s3mount_then_run3.sh ${DATASET} ${MODEL}
done