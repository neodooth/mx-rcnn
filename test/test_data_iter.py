import os
import cPickle

from helper.dataset import ilsvrc
from helper.processing import roidb
from rcnn import loader

import mxnet as mx

print 'using', mx.__path__

# test flip
# root_path = 'data'
# devkit_path = 'data/ILSVRCdevkit'
# voc = ilsvrc.ILSVRC('train', '2016', root_path, devkit_path)
# gt_roidb = voc.gt_roidb()
# ss_roidb = voc.selective_search_roidb(gt_roidb)
# ss_roidb = voc.append_flipped_images(ss_roidb)
# roidb.prepare_roidb(voc, ss_roidb)
# means, stds = roidb.add_bbox_regression_targets(ss_roidb)

cache_file_rpn_roidb = 'data/cache/ilsvrc_2016_train_rpn_roidb.pkl'
with open(cache_file_rpn_roidb, 'rb') as f:
    roidb_mean_std = cPickle.load(f)
print 'rpn roidb loaded from {}'.format(cache_file_rpn_roidb)
rpn_roidb = roidb_mean_std[0]

roi_iter = loader.ROIIter(rpn_roidb, shuffle=True)

# for j in range(0, 20):
#     print j
#     for databatch in roi_iter:
#         i = 0
#     roi_iter.reset()
