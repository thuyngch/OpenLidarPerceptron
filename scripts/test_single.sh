#!/usr/bin/env bash
set -e

CONFIG_FILE="tools/cfgs/kitti_models/pointpillar.yaml"
BATCH_SIZE=32
CKPT="ckpt/pointpillar_7728.pth"

python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT} #--eval_all