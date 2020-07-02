#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-"python"}

GPUS=2
CPUS=8
EPOCHS=80
BATCH_SIZE=16
CONFIG_FILE="tools/cfgs/kitti_models/pv_rcnn.yaml"
CKPT="output/cfgs/kitti_models/pv_rcnn/default/ckpt/pv_rcnn_8369.pth"

${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPUS} tools/test.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
    --workers ${CPUS} --ckpt ${CKPT}
