#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-"python"}

GPUS=2
CPUS=8
EPOCHS=80
BATCH_SIZE=4
CONFIG_FILE="tools/cfgs/kitti_models/pv_rcnn.yaml"
APEX=''

${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPUS} tools/train.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
    --workers ${CPUS} --fix_random_seed --extra_tag "pv_rcnn" #--apex_level ${APEX}
