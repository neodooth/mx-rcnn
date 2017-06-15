#!/bin/bash

set -x

RES1="data/ILSVRCdevkit/results/ILSVRC2016/VID/test/comp4_det_test.txt_multiscaleroi-epoch5-scale800"
RES2="data/ILSVRCdevkit/results/ILSVRC2016/VID/test/comp4_det_test.txt_multiscaleroi-epoch5-scale600"
RES3="data/ILSVRCdevkit/results/ILSVRC2016/DET/rcnn3/merged/DB"
OUTPUT="data/ILSVRCdevkit/results/ILSVRC2016/VID/test/test_merged.txt"

python merge.py $RES1 $RES2 $OUTPUT