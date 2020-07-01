#!/usr/bin/env bash
set -e

PARTITION=1
NUM_GPUS=2
CONFIG_FILE="tools/cfgs/kitti_models/pointpillar.yaml"
BATCH_SIZE=32
CKPT="ckpt/pointpillar_7728.pth"

sh tools/scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \ 
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
