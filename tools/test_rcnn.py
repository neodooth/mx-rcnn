import cv2
import argparse
import os

import mxnet as mx

from rcnn.config import config
from rcnn.loader import ROIIter
from rcnn.detector import Detector
from rcnn.symbol import get_vgg_test, get_vgg_rcnn_test
from rcnn.symbol_resnet101 import get_resnet_test, get_resnet_rcnn_test
from rcnn.tester import pred_eval
from utils.load_data import load_ilsvrc_gt_roidb
from utils.load_data import load_ilsvrc_test_ss_roidb
from utils.load_data import load_ilsvrc_test_rpn_roidb
from utils.load_model import load_param


def test_rcnn(imageset, year, root_path, devkit_path, prefix, epoch, ctx, vis=False, has_rpn=True, proposal='rpn', net='resnet', start=0, num=-1, suffix=''):
    if len(suffix) > 0:
        suffix = '_' + suffix

    roi_scales = [2]

    # load symbol and testing data
    if has_rpn:
        print 'has rpn'
        sym = eval('get_' + net + '_test')()
        config.TEST.HAS_RPN = True
        config.TEST.RPN_PRE_NMS_TOP_N = 6000
        config.TEST.RPN_POST_NMS_TOP_N = 300
        voc, roidb = load_ilsvrc_gt_roidb(imageset, year, root_path, devkit_path)
    else:
        print 'does not have rpn'
        sym = eval('get_' + net + '_rcnn_test')(roi_scales=roi_scales)
        voc, roidb = eval('load_ilsvrc_test_' + proposal + '_roidb')(imageset, year, root_path, devkit_path)

    # net_img = mx.visualization.plot_network(sym, shape={'data': (1, 3, 600, 600)}, node_attrs={"fixedsize":"fasle"})
    # print net_img
    # net_img.render('net_image/' + net + '/test_rcnn')

    print 'image scale', config.SCALES, 'max_size', config.MAX_SIZE

    # get test data iter
    test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test', roi_scales=roi_scales)

    # load model
    print 'loading', prefix, epoch
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)

    # detect
    detector = Detector(sym, ctx, args, auxs, roi_scales=roi_scales)
    pred_eval(detector, test_data, voc, vis=vis, start=start, num=num, suffix=suffix)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--image_set', dest='image_set', help='can be val/test',
                        default='val', type=str)
    parser.add_argument('--year', dest='year', help='can be 2016',
                        default='2016', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='ILSVRCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'ILSVRCdevkit'), type=str)
    parser.add_argument('--prefix', dest='prefix', help='model to test with', type=str)
    parser.add_argument('--epoch', dest='epoch', help='model to test with',
                        default=8, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to test with',
                        default=0, type=int)
    parser.add_argument('--net', dest='net', help='net name, can be vgg, resnet',
                        default='resnet', type=str)
    parser.add_argument('--start', dest='start', help='start index',
                        default=0, type=int)
    parser.add_argument('--num', dest='num', help='number of entries to test',
                        default=-1, type=int)
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    parser.add_argument('--has_rpn', dest='has_rpn', help='generate proposals on the fly',
                        action='store_true')
    parser.add_argument('--proposal', dest='proposal', help='can be ss for selective search or rpn',
                        default='rpn', type=str)
    parser.add_argument('--suffix', dest='suffix', help='suffix of output files(detections.pkl, result.txt)',
                        default='', type=str)
    parser.add_argument('--scale', dest='scale', help='number of entries to test',
                        default=600, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    config.SCALES = (args.scale, )
    test_rcnn(args.image_set, args.year, args.root_path, args.devkit_path, args.prefix, args.epoch, ctx, args.vis,
              args.has_rpn, args.proposal, args.net, args.start, args.num, args.suffix)
