# coding:utf8

# 有的图片的宽高比太大，导致在生成anchor时所有anchor都不在图片里于是出错
# 这个程序是找到宽高比（高宽比）大于5的那些图片，给他们加上边框来让宽高比更正常
# 转换后的数据在data_to_replace里
# 列表是image_sizes_train_replace.txt，格式和image_sizes_train.txt相同

import cv2
import os
from os import path
import re

MAX_RATIO = 5.0
IDEAL_RATIO = 5
WHITE = [255, 255, 255]

def save(im_name, im, annotation):
    sp = im_name.split('/')
    _data_base = 'data_to_replace/Data/DET/train'
    _anno_base = 'data_to_replace/Annotations/DET/train'
    for directory in sp[:-1]:
        _data_base = path.join(_data_base, directory)
        _anno_base = path.join(_anno_base, directory)
        if not path.exists(_data_base):
            os.mkdir(_data_base)
        if not path.exists(_anno_base) and annotation is not None:
            os.mkdir(_anno_base)
    cv2.imwrite(path.join(_data_base, sp[-1] + '.JPEG'), im)
    if annotation is not None:
        with open(path.join(_anno_base, sp[-1] + '.xml'), 'w') as f:
            print >> f, annotation

with open('image_sizes_train.txt') as f:
    lines = f.read().splitlines()
line_num = len(lines)
fout = open('image_sizes_train_replace.txt', 'w')
for ind, l in enumerate(lines):
    sp = l.split(' ')
    w, h, im_name = int(sp[0]), int(sp[1]), sp[2]
    ratio = float(max(w, h)) / min(w, h)
    if ratio > MAX_RATIO:
        new_w, new_h = w, h
        if w > h:
            new_h = w / IDEAL_RATIO
        else:
            new_w = h / IDEAL_RATIO
        print >> fout, new_w, new_h, im_name
        padding_top = padding_bottom = (new_h - h) / 2
        padding_left = padding_right = (new_w - w) / 2
        print 'Padding {}, from {}x{} to {}x{}'.format(im_name, w, h, new_w, new_h)
        im = cv2.imread(path.join('ILSVRC2016/Data/DET/train', im_name + '.JPEG'))
        im_padded = cv2.copyMakeBorder(
            im, padding_top, padding_bottom, padding_left, padding_right,
            cv2.BORDER_REPLICATE, value=WHITE)

        xml_path = path.join('ILSVRC2016/Annotations/DET/train', im_name + '.xml')
        if not path.exists(xml_path):
            print 'no xml', im_name
            anno = None
        else:
            with open(xml_path) as f:
                anno = f.read()
            anno = re.sub('(<width>).*(</width>)', r'\g<1>{}\2'.format(new_w), anno)
            anno = re.sub('(<height>).*(</height>)', r'\g<1>{}\2'.format(new_h), anno)
            for xy, delta in [('xmax', padding_left), ('xmin', padding_left), ('ymax', padding_top), ('ymin', padding_top)]:
                begin = '<{}>'.format(xy)
                end = '</{}>'.format(xy)
                pos_st = anno.rfind(begin)
                pos_end = anno.rfind(end)
                while pos_st != -1:
                    ori_xy = anno[pos_st + len(begin): pos_end]
                    # print xy, ori_xy
                    # if ori_xy == '':
                    #     print 'xy', xy
                    #     print begin, end
                    #     print pos_st + len(begin), ':', pos_end
                    #     # print anno
                    anno = '{}{}{}'.format(
                        anno[:pos_st + len(begin)],
                        int(ori_xy) + delta,
                        anno[pos_end:]
                        )
                    pos_st = anno.rfind(begin, 0, pos_st-1)
                    pos_end = anno.rfind(end, 0, pos_end-1)
        save(im_name, im_padded, anno)
    if ind+1 % 1000 == 0:
        print ind, '/', line_num
fout.close()
