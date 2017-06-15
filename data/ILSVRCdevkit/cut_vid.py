#coding:utf8

# 把vid的是数据按每x帧取出来

import sys

base = 'data/ILSVRCdevkit/ILSVRC2016/ImageSets/VID/'
imageset = 'test'

fin = open(base + imageset + '.txt-original')
interval = 5

fout = open(base + imageset + '.txt-77', 'w')
fout_map = open(base + imageset + '.txt-77-map', 'w')

vidname = ''
count = 0
frame_index = 1
for l in fin:
    sp = l.split()
    vname = sp[0].split('/')[0]
    if vname != vidname:
        vidname = vname
        count = 0
    if count % interval != 0:
        # 76: %2==0; 77: %2!=0
        if count % 2 != 0:
            print >> fout, sp[0], frame_index
            print >> fout_map, frame_index, sp[1]
            frame_index += 1
    count += 1
