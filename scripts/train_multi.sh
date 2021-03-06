#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-"python"}

CPUS=8
EPOCHS=80
BATCH_SIZE=8
CONFIG_FILE="tools/cfgs/kitti_models/pv_rcnn.yaml"
APEX=''

GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPUS} tools/train.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
    --workers ${CPUS} --fix_random_seed --extra_tag "pv_rcnn" #--apex_level ${APEX}
