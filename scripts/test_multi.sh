#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-"python"}

GPUS=2
CPUS=8
EPOCHS=80
BATCH_SIZE=32
CONFIG_FILE="tools/cfgs/kitti_models/pointpillar_gn.yaml"
CKPT="output/cfgs/kitti_models/pointpillar_gn/pointpillar_gn/ckpt/checkpoint_epoch_80.pth"

${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPUS} tools/test.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
    --workers ${CPUS} --ckpt ${CKPT}
