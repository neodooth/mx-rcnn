#coding:utf8
# 用来合并并行测试rpn/rcnn输出的代码

import os
import cPickle as cp
import sys

print 'args: base_dir prefix'

basedir = sys.argv[1]
prefix = sys.argv[2]

os.chdir(basedir)
files = [f for f in os.listdir('.') if f.startswith(prefix)]
files = sorted(files, key=lambda x: int(x.split('_')[-1].split('-')[0]))
for fn in files:
    print fn

output = []
for fn in files:
    with open(fn, 'rb') as f:
        dets = cp.load(f)
        print 'processing', fn, len(dets), 'boxes'
        output += dets

print len(output), 'boxes'
with open(prefix + '.pkl', 'wb') as fout:
    cp.dump(output, fout)
