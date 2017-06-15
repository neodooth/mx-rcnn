#!/bin/bash

# export MXNET_ENGINE_TYPE=NaiveEngine

NET=resnet
# PRETRAINED=`pwd`/model/resnet-101-eltsum
PRETRAINED=`pwd`/model/resnet-101-fix123/rpn3-lr0001

# NET=vgg
# PRETRAINED=`pwd`/model/vgg16
# OUTPUT_PREFIX=`pwd`/model/vgg-old/

IMAGE_SET=val

EPOCH=2
GPUS=$1
START=$2
NUM=$3
ID_LIST=$4

# LOG="logs/hukun-`echo $NET`-`echo $START`-`echo $NUM`.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="logs/ilsvrc2017-det-rpn-`echo $NET`-`echo $START`-`echo $NUM`.txt.`echo $IMAGE_SET`-800"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set -x

CALCTIME="-m cProfile -s cumtime"
CALCTIME=

python -u $CALCTIME -m tools/test_rpn \
	--year=2017 \
	--image_set=$IMAGE_SET \
	--net=$NET \
	--prefix=$PRETRAINED \
	--epoch=$EPOCH \
	--gpu=$GPUS \
	--start=$START \
	--num=$NUM
	# --vis
	# --id_list=$ID_LIST
	# --devkit_path=data/VOCdevkit \
	# --year=2007