#!/bin/bash
set -ex

# >>> conda initialize >>>
export PATH="/mnt/shared-storage-user/zhuyuhan/miniconda3/bin:$PATH"
__conda_setup="$('/mnt/shared-storage-user/zhuyuhan/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/shared-storage-user/zhuyuhan/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/shared-storage-user/zhuyuhan/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/shared-storage-user/zhuyuhan/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda deactivate
conda activate videochat3
# <<< conda initialize <<<

export LMUData="/mnt/shared-storage-user/zhuyuhan/LMUData"
bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh

cd /mnt/shared-storage-user/zhuyuhan/videochat3/VLMEvalKit

DATASETS="$1"
MODEL="$2"

torchrun --nproc-per-node=8 run.py \
  --data $DATASETS \
  --model $MODEL \
  --verbose \
  --reuse \
  --judge qwen3-30b-a3b-instruct-2507