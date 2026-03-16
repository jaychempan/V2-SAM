#!/bin/bash

# 1 : modify CHANNEL_AGNOSTIC_CELL_MODEL, CHAMMI_DATA_PATH and OUTPUT_DIR below
# 2 : call this script with the two arguments specified below

#Arguments: 
# $1 : dataset, e.g  CP
# $2 : task number, e.g TASK_TWO

CHAMMI_DATA_PATH=""
CHANNEL_AGNOSTIC_CELL_MODEL="path_to_model/model.pth"
OUTPUT_DIR=YOUR_OUTPUT_PATH_$1_$2

if [ "$2" == "TASK_FOUR" ]; then
    OTHER_ARG="--leave-one-out-dataset $CHAMMI_DATA_PATH/CP/enriched_meta.csv "
elif [ "$1" == "HPA" -a "$2" == "TASK_THREE" ]; then
    OTHER_ARG="--leave-one-out-dataset $CHAMMI_DATA_PATH/CHAMMI/HPA/enriched_meta.csv "
else
    OTHER_ARG=""
fi  
echo $OTHER_ARG

PYTHONPATH=..:../../dinov2/data python ../../dinov2/run/eval/cell_dino/knn.py \
--config-file ../../dinov2/configs/eval/cell_dino/vitl16_channel_adaptive_pretrain.yaml \
--pretrained-weights $CHANNEL_AGNOSTIC_CELL_MODEL \
--output-dir $OUTPUT_DIR \
--train-dataset CHAMMI_$1:split=TRAIN:root=$CHAMMI_DATA_PATH \
--val-dataset CHAMMI_$1:split=$2:root=$CHAMMI_DATA_PATH \
--metric-type mean_per_class_multiclass_f1 \
--crop-size 224 \
--batch-size 32 \
--resize-size 256 \
--bag-of-channels \
$OTHER_ARG \
