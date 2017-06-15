#!/bin/bash

# export MXNET_ENGINE_TYPE=NaiveEngine

NET=resnet
PRETRAINED=`pwd`/model/resnet-101-eltsum
OUTPUT_PREFIX=`pwd`/model/resnet-101-joint/

LOAD_EPOCH=1
BEGIN_EPOCH=0
END_EPOCH=12
GPUS=3
LR=0.001

mkdir -p $OUTPUT_PREFIX

# LOG="logs/ilsvrc2016-`echo $NET`-joint.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
# # LOG="logs/ilsvrc2016-`echo $NET`-joint.txt.test"
# exec &> >(tee -a "$LOG")
# echo Logging output to "$LOG"

set -x

CALCTIME="-m cProfile -s cumtime"
CALCTIME=

python -u $CALCTIME train_end2end.py \
	--pretrained=$PRETRAINED \
	--load_epoch=$LOAD_EPOCH \
	--output_prefix=$OUTPUT_PREFIX \
	--begin_epoch=$BEGIN_EPOCH \
	--end_epoch=$END_EPOCH \
	--gpus=$GPUS \
	--lr=$LR \
	--frequent=20