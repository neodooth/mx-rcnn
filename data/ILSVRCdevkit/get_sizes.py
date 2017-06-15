#!/usr/bin/env python
# coding:utf8

import os
from PIL import Image

year = 2017

def _load_image_set_index(image_set):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join('ILSVRC{}'.format(year), 'ImageSets', 'DET',
                                 image_set + '.txt')
    assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    return image_index

def read_size(set_name, image_set, filename):
    def do_read(img_path):
        return Image.open(img_path).size
    base = os.path.join('ILSVRC{}'.format(year), 'Data', 'DET', set_name)
    replace_base = os.path.join('data_to_replace', 'Data', 'DET', set_name)
    all_n = len(image_set)
    with open(filename, 'w') as f:
        for ind, i in enumerate(image_set):
            p = os.path.join(base, i) + '.JPEG'
            replace_path = os.path.join(replace_base, i) + '.JPEG'
            if os.path.exists(replace_path):
                p = replace_path
            size = do_read(p)
            # width, height, image
            print >> f, size[0], size[1], i
            if (ind+1) % 100 == 0:
                print ind+1, all_n

 # ILSVRC2013_train_extra0/ILSVRC2013_train_00000001
# image_set_train = _load_image_set_index('train')
# image_set_val = _load_image_set_index('val')
image_set_test = _load_image_set_index('test')
# read_size('train', image_set_train, 'image_sizes_train.txt')
# read_size('val', image_set_val, 'image_sizes_val.txt')
read_size('test', image_set_test, 'image_sizes_test.txt-det-2017')
