#!/bin/bash
set -x

cd /mnt/petrelfs/zhuyuhan/workspace/VLMEvalKit

########################################################################################

MODEL="VideoChat3_4B_train_stage2_debugbase4x2_beedata_minisft_lr8e-5_vtlr2e-5"

########################################################################################

DATASETS=(OCRBench ChartQA_TEST DocVQA_VAL InfoVQA_VAL)

for DATASET in "${DATASETS[@]}"; do
  echo "Running inference on dataset: ${DATASET}"
  srun --async -p videop1 --gres=gpu:8 --quotatype=spot --job-name=${DATASET} bash scripts/s3mount_then_run2.sh ${DATASET} ${MODEL}
done

########################################################################################

# DATASETS=(MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL Video-MME_short_2fps MMVU_2fps)
# DATASETS=(MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL MathVista_MINI MathVision_MINI Video-MME_short_2fps Video-MME_medium_2fps Video-MME_long_2fps LongVideoBench_2fps LVBench_2fps TOMATO_2fps VideoMMMU_2fps MMVU_2fps VUE_TR_1fps)
DATASETS=(MMBench_DEV_EN_V11 RealWorldQA AI2D_TEST MMStar MMMU_DEV_VAL Video-MME_short_2fps VideoMMMU_2fps TOMATO_2fps MMVU_2fps)

for DATASET in "${DATASETS[@]}"; do
  echo "Running inference on dataset: ${DATASET}"
  srun --async -p videop1 --gres=gpu:8 --quotatype=reserved --job-name=${DATASET} --preempt bash scripts/s3mount_then_run.sh ${DATASET} ${MODEL}
done


rm batchscript-*