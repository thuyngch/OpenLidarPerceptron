#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-"python"}

GPUS=4
CPUS=16
EPOCHS=80
BATCH_SIZE=32
CONFIG_FILE="tools/cfgs/kitti_models/pv_rcnn.yaml"
CKPT="output/cfgs/kitti_models/pv_rcnn/pv_rcnn/ckpt/checkpoint_epoch_80.pth"

export CUDA_VISIBLE_DEVICES=0,1,2,3
${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPUS} tools/test.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
    --workers ${CPUS} --ckpt ${CKPT}
