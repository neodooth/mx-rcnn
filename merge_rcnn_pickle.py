#coding:utf8

import cPickle as cp
import os
import sys


print 'args: res_dir output_path'


res_dir = sys.argv[1]
output_path = sys.argv[2]

fns = os.listdir(res_dir)
fns = sorted(fns, key=lambda x: int(x.split('-')[1]))
starts = [int(_.split('-')[1]) for _ in fns]
ends = [int(_.split('-')[2].split('.')[0]) for _ in fns]

print fns
print starts
print ends
raw_input('enter to continue')

pks = []
for fn in fns:
    print 'loading', fn
    with open(os.path.join(res_dir, fn), 'rb') as f:
        pks.append(cp.load(f))

merged = [[] for _ in range(201)]

for p, s, e in zip(pks, starts, ends):
    e = min(e, len(p[1]))
    print 'range', s, e
    raw_input('enter to continue')
    for i in range(s, e):
        for j in range(1, 201):
            merged[j].append(p[j][i])

with open(output_path, 'wb') as f:
    cp.dump(merged, f)
