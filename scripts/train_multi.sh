#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-"python"}

GPUS=2
CPUS=8
EPOCHS=80
BATCH_SIZE=12
CONFIG_FILE="tools/cfgs/kitti_models/pointpillar_gn.yaml"

${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPUS} tools/train.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
    --workers ${CPUS} --fix_random_seed --extra_tag "pointpillar_gn"
