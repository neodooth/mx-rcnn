#!/bin/bash

# export MXNET_ENGINE_TYPE=NaiveEngine

NET=resnet
# PRETRAINED=`pwd`/model/resnet-101-eltsum
PRETRAINED=`pwd`/model/resnet-101-fix123/rcnn3-multiscaleroi-edit_params

# NET=vgg
# PRETRAINED=`pwd`/model/vgg16
# OUTPUT_PREFIX=`pwd`/model/vgg-old/

IMAGE_SET=test

EPOCH=5
GPUS=$1
START=$2
NUM=$3
# ID_LIST=$4

SCALE=600

SUFFIX=multiscaleroi-epoch5-scale`echo $SCALE`

LOG="logs/det-rcnn3_`echo $IMAGE_SET`/ilsvrc2017-`echo $NET`-test_rcnn-`echo $START`-`echo $NUM`.txt.`date +'%Y-%m-%d_%H-%M-%S'`-`echo $SUFFIX`"
# LOG="logs/ilsvrc2016-`echo $NET`.txt.test"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set -x

CALCTIME="-m cProfile -s cumtime"
CALCTIME=

python -u $CALCTIME -m tools/test_rcnn \
	--year=2017 \
	--image_set=$IMAGE_SET \
	--net=$NET \
	--prefix=$PRETRAINED \
	--epoch=$EPOCH \
	--gpu=$GPUS \
	--start=$START \
	--num=$NUM \
	--suffix=$SUFFIX \
	--scale=$SCALE
	# --id_list=$ID_LIST
	# --devkit_path=data/VOCdevkit \
	# --year=2007