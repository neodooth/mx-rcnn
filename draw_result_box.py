#coding:utf8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random

base = 'data/ILSVRCdevkit/results/ILSVRC2016/VID/'
resfile = 'test-full.txt'
resfile = base + resfile
print resfile

results = [[] for _ in range(2000000)]
valid_indices = []
with open(resfile) as f:
    for l in f:
        sp = l.split()
        # classid, score, box
        results[int(sp[0])].append([int(sp[1]), float(sp[2]), float(sp[3]), float(sp[4]), float(sp[5]), float(sp[6])])
        valid_indices.append(int(sp[0]))

valid_indices = sorted(list(set(valid_indices)))

imageset_file = 'data/ILSVRCdevkit/ILSVRC2016/ImageSets/VID/test.txt-original'
with open(imageset_file) as f:
    images = [''] + [_.split()[0] for _ in f]
print len(images), 'images'

map_det_file = 'data/ILSVRCdevkit/ILSVRC2016/devkit/data/map_vid.txt'
with open(map_det_file) as f:
    map_det = ['background'] + [_.split()[-1] for _ in f]

lucky_inds = sorted(random.sample(valid_indices, 100))
# lucky_inds = valid_indices[0:100]
for i in lucky_inds:
    print i
    img = mpimg.imread('data/ILSVRCdevkit/ILSVRC2016/Data/VID/test/' + images[i] + '.JPEG')
    plt.imshow(img)
    for b in results[i]:
        bbox = b[2:]
        score = b[1]
        clsid = b[0]
        if score > 0.5:
            color = (random.random(), random.random(), random.random())  # generate a random color
            print b
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(map_det[clsid], score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.savefig('detections/{}'.format(i))
    plt.cla()
