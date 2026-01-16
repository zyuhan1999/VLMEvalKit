mkdir -p /mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan
fusermount -u /mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan
export AWS_ACCESS_KEY_ID=o0asnixdrdvctn7pdpjb
export AWS_SECRET_ACCESS_KEY=ul34evliiq6929jbvby3x4w1xxb6fp8xh0gqo3pg
/mnt/petrelfs/zhuyuhan/s3mount zhuyuhan /mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan --endpoint-url http://hdd1.h.pjlab.org.cn:8060 --force-path-style --log-directory /mnt/petrelfs/zhuyuhan/workspace/s3mount_scripts

mkdir -p /mnt/petrelfs/zhuyuhan/s3/pnorm2/videochat3
fusermount -u /mnt/petrelfs/zhuyuhan/s3/pnorm2/videochat3
export AWS_ACCESS_KEY_ID=HES1ITGIVHJ36GHYMTV4
export AWS_SECRET_ACCESS_KEY=G2FKmfJxKCTkfyICOD4GsT08x8wUNCbsa17OmNQM
/mnt/petrelfs/zhuyuhan/s3mount videochat3 /mnt/petrelfs/zhuyuhan/s3/pnorm2/videochat3 --endpoint-url http://p-ceph-hdd2-inside.pjlab.org.cn --force-path-style --log-directory /mnt/petrelfs/zhuyuhan/workspace/s3mount_scripts

cd /mnt/petrelfs/zhuyuhan/workspace/VLMEvalKit

export LMUData="/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/LMUData"

DATASET=$1
MODEL=$2

torchrun --nproc-per-node=8 --master_port=10492 run.py \
  --data ${DATASET} \
  --model ${MODEL} \
  --verbose \
  --reuse \
  --judge qwen3-235b-a22b-thinking-2507