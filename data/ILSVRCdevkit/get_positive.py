import xml.etree.ElementTree as ET
import os

with open('ILSVRC2016/ImageSets/DET/train.txt') as f:
    indices = [_.split()[0] for _ in f if 'extra' not in _]

pos = []
for i in indices:
    filename = os.path.join('ILSVRC2016', 'Annotations', 'DET', 'train', i + '.xml')

    # Images without any annotated objects may not have a corresponding xml file.
    if not os.path.exists(filename):
        print 'Did not found', filename
        continue

    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)
    if num_objs == 0:
        print 'No objects found from {}'.format(i)
        continue

    pos.append(i)

with open('positive.txt', 'w') as f:
    for p in pos:
        print >> f, p
