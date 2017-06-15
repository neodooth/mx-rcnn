#!/bin/bash

# export MXNET_ENGINE_TYPE=NaiveEngine

NET=resnet
PRETRAINED=`pwd`/model/resnet-101-eltsum
# PRETRAINED=`pwd`/model/resnet-101-fix123/
OUTPUT_PREFIX=`pwd`/model/resnet-101-fix123/

# NET=vgg
# PRETRAINED=`pwd`/model/vgg16
# OUTPUT_PREFIX=`pwd`/model/vgg-old/

# pretraiend imagenet model epoch
EPOCH=2
# rpn/rcnn model epoch
BEGIN_EPOCH=4
RPN_EPOCH=2
RCNN_EPOCH=10
GPUS=0,1

mkdir -p $OUTPUT_PREFIX

LOG="logs/ilsvrc2016-`echo $NET`-rcnn3-multiscaleroi.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
# LOG="logs/ilsvrc2016-`echo $NET`.txt.test"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set -x

CALCTIME="-m cProfile -s cumtime"
CALCTIME=

python -u $CALCTIME train_alternate.py \
	--net=$NET \
	--pretrained=$PRETRAINED \
	--epoch=$EPOCH \
	--output_prefix=$OUTPUT_PREFIX \
	--begin_epoch=$BEGIN_EPOCH \
	--rpn_epoch=$RPN_EPOCH \
	--rcnn_epoch=$RCNN_EPOCH \
	--gpus=$GPUS
	# --devkit_path=/home/zhangjiangqi/project/imagenet2016/det/faster-rcnn/mx-rcnn/data/VOCdevkit \
	# --year=2007