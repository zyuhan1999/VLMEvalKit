
# DATASETS="VideoMMMU_2fps"
# MODEL="VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_t1"

# rjob submit --name=${MODEL}-${DATASETS} \
#     --gpu=8 --memory=1900000 --cpu=170 \
#     --task-type=idle --private-machine=no --restart-policy=restartjobonfailure --backoff_limit=100 \
#     --mount=gpfs://gpfs1/zhuyuhan:/mnt/shared-storage-user/zhuyuhan \
#     --mount=gpfs://gpfs1/zengxiangyu:/mnt/shared-storage-user/zengxiangyu \
#     --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
#     --custom-resources brainpp.cn/fuse=1 \
#     --image=registry.h.pjlab.org.cn/ailab-intern9-intern9_gpu/zxy:videorl-20251118185838 \
#     -- sh /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit/scripts/eval_videochat3/eval_4b.sh $DATASETS $MODEL

DATASETS=(
  "Video-MME_2fps"
  "LongVideoBench_2fps"
  "LVBench_2fps"
  "TOMATO_2fps"
  "VideoMMMU_2fps"
  "MMVU_2fps"
)

MODEL="VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_t1"

for DATASET in "${DATASETS[@]}"; do
  echo "Submitting job for dataset: $DATASET"

  rjob submit --name=${MODEL}-${DATASET} \
      --gpu=8 --memory=1900000 --cpu=170 \
      --task-type=idle --private-machine=no \
      --restart-policy=restartjobonfailure --backoff_limit=100 \
      --mount=gpfs://gpfs1/zhuyuhan:/mnt/shared-storage-user/zhuyuhan \
      --mount=gpfs://gpfs1/zengxiangyu:/mnt/shared-storage-user/zengxiangyu \
      --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
      --custom-resources brainpp.cn/fuse=1 \
      --image=registry.h.pjlab.org.cn/ailab-intern9-intern9_gpu/zxy:videorl-20251118185838 \
      -- sh /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit/scripts/eval_videochat3/eval_4b.sh \
         "$DATASET" "$MODEL"
done