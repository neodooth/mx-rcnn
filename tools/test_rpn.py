import cv2
import argparse
import os
import time
import cPickle as cp

import mxnet as mx

from rcnn.config import config
from rcnn.loader import ROIIter
from rcnn.rpn.generate import Detector, generate_detections, generate_detections_from_image
from rcnn.symbol import get_vgg_rpn_test
from rcnn.symbol_resnet101 import get_resnet_rpn_test
from utils.load_data import load_ilsvrc_gt_roidb, load_gt_roidb
from utils.load_model import load_param

# rpn generate proposal config
config.TEST.HAS_RPN = True
config.TEST.RPN_PRE_NMS_TOP_N = -1
config.TEST.RPN_POST_NMS_TOP_N = 2000


def test_rpn(image_set, year, root_path, devkit_path, prefix, epoch, ctx, vis=False, net='vgg', start=0, num=-1, id_list=''):
    if 'train' not in image_set:
        config.TEST.RPN_POST_NMS_TOP_N = 300
    # load symbol
    sym = eval('get_' + net + '_rpn_test')()

    # try:
    #     net_img = mx.visualization.plot_network(sym, shape={'data': (1, 3, 600, 901)}, node_attrs={"fixedsize":"fasle"})
    #     net_img.render('net_image/' + net + '/test_rpn')
    # except:
    #     pass

    # load testing data
    if 'ILSVRC' in devkit_path:
        voc, roidb = load_ilsvrc_gt_roidb(image_set, year, root_path, devkit_path)
    else:
        voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path)
    test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test')
    print 'test_data length', test_data.size

    # load model
    print 'loading', prefix, 'epoch', epoch
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)

    # start testing
    detector = Detector(sym, ctx, args, auxs)
    imdb_boxes = generate_detections(detector, test_data, voc, vis=vis, start=start, num=num, id_list=id_list)
    # voc.evaluate_recall(roidb, candidate_boxes=imdb_boxes)

def test_rpn_from_images(prefix, epoch, ctx, vis=False, start=0, num=-1):
    # load symbol
    sym = eval('get_resnet_rpn_test')()

    # try:
    #     net_img = mx.visualization.plot_network(sym, shape={'data': (1, 3, 600, 901)}, node_attrs={"fixedsize":"fasle"})
    #     net_img.render('net_image/' + net + '/test_rpn')
    # except:
    #     pass

    # load testing data
    images = open('share/mirflick25.list').read().splitlines()
    print 'test_data length', len(images)
    end = start + num if num != -1 else len(images)
    print 'from', start, 'to', end
    images = images[start:end]

    # load model
    print 'loading', prefix, 'epoch', epoch
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)

    # start testing
    detector = Detector(sym, ctx, args, auxs)
    boxes = []
    st = time.time()
    for i in images:
        print time.time() - st, i
        det = generate_detections_from_image(detector, i, vis)
        boxes.append(det)

    fname = 'data/rpn_data/mirflick25-{}-{}.pkl'.format(start, end)
    print 'saveing to', fname
    with open(fname, 'wb') as f:
        cp.dump(boxes, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Region Proposal Network')
    parser.add_argument('--image_set', dest='image_set', help='can be train, val or test',
                        default='train', type=str)
    parser.add_argument('--year', dest='year', help='can be 2016',
                        default='2016', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='ILSVRCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'ILSVRCdevkit'), type=str)
    parser.add_argument('--prefix', dest='prefix', help='model to test with', type=str)
    parser.add_argument('--epoch', dest='epoch', help='model to test with',
                        default=8, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--net', dest='net', help='net name, can be vgg, resnet',
                        default='vgg', type=str)
    parser.add_argument('--start', dest='start', help='start index',
                        default=0, type=int)
    parser.add_argument('--num', dest='num', help='number of entries to test',
                        default=-1, type=int)
    parser.add_argument('--id_list', dest='id_list', help='id list file',
                        default='', type=str)
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    test_rpn(args.image_set, args.year, args.root_path, args.devkit_path, args.prefix, args.epoch, ctx, args.vis, args.net, args.start, args.num, args.id_list)
    # test_rpn_from_images(args.prefix, args.epoch, ctx, args.vis, args.start, args.num)
