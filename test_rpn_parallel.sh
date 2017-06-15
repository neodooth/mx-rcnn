#!/bin/bash

NUM=2020
nprocess=5

count=0
start=0
for gpu in 1 2; do
	for i in `seq $nprocess`; do
		PARAMS="$gpu $start $NUM"
		# id_list=list/failed.0${count}
		# PARAMS=$PARAMS $id_list
		nohup bash test_rpn.sh $PARAMS > /dev/null &
		let count=count+1
		let start=${start}+${NUM}
		sleep 1
	done
done