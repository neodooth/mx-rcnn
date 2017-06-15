#!/bin/bash

NUM=2520

count=0
nprocess=2
start=0
for gpu in 3 2 1 0; do
	for i in `seq $nprocess`; do
		PARAMS="$gpu $start $NUM"
		# id_list=list/failed.0${count}
		# PARAMS=$PARAMS $id_list
		set -x 
		nohup bash test_rcnn.sh $PARAMS > /dev/null &
		set +x
		let count=count+1
		let start=${start}+${NUM}
		sleep 1
	done
done