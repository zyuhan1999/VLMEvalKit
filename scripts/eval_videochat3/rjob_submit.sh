
DATASETS="Video-MME_2fps"
MODEL="VideoChat3_4B_train_stage2_llava_video_academic"

rjob submit --name=${MODEL}-$(date +"%Y%m%d-%H%M") \
    --gpu=8 --memory=1900000 --cpu=170 \
    --task-type=idle --private-machine=no --restart-policy=restartjobonfailure --backoff_limit=100 \
    --mount=gpfs://gpfs1/zhuyuhan:/mnt/shared-storage-user/zhuyuhan \
    --mount=gpfs://gpfs1/zengxiangyu:/mnt/shared-storage-user/zengxiangyu \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --custom-resources brainpp.cn/fuse=1 \
    --image=registry.h.pjlab.org.cn/ailab-intern9-intern9_gpu/zxy:videorl-20251118185838 \
    -- sh /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit/scripts/eval_videochat3/eval_4b.sh $DATASETS $MODEL