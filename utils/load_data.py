import cPickle
import os

from helper.dataset.pascal_voc import PascalVOC
from helper.processing.roidb import prepare_roidb, add_bbox_regression_targets
from helper.dataset.ilsvrc import ILSVRC

import time


def load_ss_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    if flip:
        ss_roidb = voc.append_flipped_images(ss_roidb)
    prepare_roidb(voc, ss_roidb)
    means, stds = add_bbox_regression_targets(ss_roidb)
    return voc, ss_roidb, means, stds


def load_gt_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    if flip:
        gt_roidb = voc.append_flipped_images(gt_roidb)
    prepare_roidb(voc, gt_roidb)
    return voc, gt_roidb


def load_rpn_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    rpn_roidb = voc.rpn_roidb(gt_roidb)
    if flip:
        rpn_roidb = voc.append_flipped_images(rpn_roidb)
    prepare_roidb(voc, rpn_roidb)
    means, stds = add_bbox_regression_targets(rpn_roidb)
    return voc, rpn_roidb, means, stds


def load_test_ss_roidb(image_set, year, root_path, devkit_path):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    prepare_roidb(voc, ss_roidb)
    return voc, ss_roidb


def load_test_rpn_roidb(image_set, year, root_path, devkit_path):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    rpn_roidb = voc.rpn_roidb(gt_roidb)
    prepare_roidb(voc, rpn_roidb)
    return voc, rpn_roidb

def load_ilsvrc_ss_roidb(image_set, year, root_path, devkit_path, flip=False):
    ilsvrc = ILSVRC(image_set, year, root_path, devkit_path)
    gt_roidb = ilsvrc.gt_roidb()
    ss_roidb = ilsvrc.selective_search_roidb(gt_roidb)
    if flip:
        ss_roidb = ilsvrc.append_flipped_images(ss_roidb)
    prepare_roidb(ilsvrc, ss_roidb)
    means, stds = add_bbox_regression_targets(ss_roidb)
    return ilsvrc, ss_roidb, means, stds


def load_ilsvrc_gt_roidb(image_set, year, root_path, devkit_path, flip=False):
    ilsvrc = ILSVRC(image_set, year, root_path, devkit_path)
    cache_file = os.path.join(ilsvrc.cache_path, ilsvrc.name + '_prepared_gt_roidb.pkl')
    if flip == False:
        cache_file = os.path.join(ilsvrc.cache_path, ilsvrc.name + '_prepared_gt_roidb_noflip.pkl')
    print 'gt_roidb', cache_file
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            gt_roidb = cPickle.load(fid)
        print '{} prepared gt roidb loaded from {}'.format(ilsvrc.name, cache_file)
    else:
        gt_roidb = ilsvrc.gt_roidb()
        if flip:
            gt_roidb = ilsvrc.append_flipped_images(gt_roidb)
        prepare_roidb(ilsvrc, gt_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote prepared gt roidb to {}'.format(cache_file)
    return ilsvrc, gt_roidb


def load_ilsvrc_rpn_roidb(image_set, year, root_path, devkit_path, flip=False):
    ilsvrc = ILSVRC(image_set, year, root_path, devkit_path)

    cache_file_rpn_roidb = os.path.join(ilsvrc.cache_path, ilsvrc.name + '_rpn_roidb.pkl')
    if not flip:
        cache_file_rpn_roidb = os.path.join(ilsvrc.cache_path, ilsvrc.name + '_rpn_roidb_noflip.pkl')

    if os.path.exists(cache_file_rpn_roidb):
        print 'loading', cache_file_rpn_roidb
        with open(cache_file_rpn_roidb, 'rb') as f:
            roidb_mean_std = cPickle.load(f)
        print '{} rpn roidb loaded from {}'.format(ilsvrc.name, cache_file_rpn_roidb)
        rpn_roidb = roidb_mean_std[0]
        means = roidb_mean_std[1]
        stds = roidb_mean_std[2]
    else:
        cache_file = os.path.join(ilsvrc.cache_path, ilsvrc.name + '_prepared_rpn_roidb.pkl')
        if flip == False:
            cache_file = os.path.join(ilsvrc.cache_path, ilsvrc.name + '_prepared_rpn_roidb_noflip.pkl')
        if os.path.exists(cache_file):
            print 'loading', cache_file
            with open(cache_file, 'rb') as fid:
                rpn_roidb = cPickle.load(fid)
            print '{} prepared rpn roidb loaded from {}'.format(ilsvrc.name, cache_file)
        else:
            gt_roidb = ilsvrc.gt_roidb()
            rpn_roidb = ilsvrc.rpn_roidb(gt_roidb)
            if flip:
                rpn_roidb = ilsvrc.append_flipped_images(rpn_roidb)
            prepare_roidb(ilsvrc, rpn_roidb)
            with open(cache_file, 'wb') as fid:
                cPickle.dump(rpn_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote prepared rpn roidb to {}'.format(cache_file)
            del gt_roidb

        means, stds = add_bbox_regression_targets(rpn_roidb)
        roidb_mean_std = (rpn_roidb, means, stds)
        with open(cache_file_rpn_roidb, 'wb') as f:
            cPickle.dump(roidb_mean_std, f, cPickle.HIGHEST_PROTOCOL)
        print 'wrote rpn roidb to {}'.format(cache_file_rpn_roidb)

    print 'load_ilsvrc_rpn_roidb means {} stds {}'.format(means, stds)

    return ilsvrc, rpn_roidb, means, stds


def load_ilsvrc_test_ss_roidb(image_set, year, root_path, devkit_path):
    ilsvrc = ILSVRC(image_set, year, root_path, devkit_path)
    gt_roidb = ilsvrc.gt_roidb()
    ss_roidb = ilsvrc.selective_search_roidb(gt_roidb)
    prepare_roidb(ilsvrc, ss_roidb)
    return ilsvrc, ss_roidb


def load_ilsvrc_test_rpn_roidb(image_set, year, root_path, devkit_path):
    ilsvrc = ILSVRC(image_set, year, root_path, devkit_path)
    cache_file_rpn_roidb = os.path.join(ilsvrc.cache_path, ilsvrc.name + '_rpn_roidb.pkl')
    if os.path.exists(cache_file_rpn_roidb):
        print 'loading', cache_file_rpn_roidb
        with open(cache_file_rpn_roidb, 'rb') as f:
            rpn_roidb = cPickle.load(f)
        print '{} rpn roidb loaded from {}'.format(ilsvrc.name, cache_file_rpn_roidb)
    else:
        print 'did not found {}'.format(cache_file_rpn_roidb)
        gt_roidb = ilsvrc.gt_roidb()
        rpn_roidb = ilsvrc.rpn_roidb(gt_roidb)
        prepare_roidb(ilsvrc, rpn_roidb)
        with open(cache_file_rpn_roidb, 'wb') as f:
            cPickle.dump(rpn_roidb, f, cPickle.HIGHEST_PROTOCOL)
        print 'wrote rpn roidb to {}'.format(cache_file_rpn_roidb)

    return ilsvrc, rpn_roidb
