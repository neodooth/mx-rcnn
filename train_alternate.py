import argparse
import logging
import os

import mxnet as mx

from rcnn.config import config
from rcnn.loader import AnchorLoader, ROIIter
from tools.train_rpn import train_rpn
from tools.train_rcnn import train_rcnn
from tools.test_rpn import test_rpn
from utils.combine_model import combine_model


logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')


def alt_train_rcnn(image_set, test_image_set, year, root_path, devkit_path, net, pretrained, output_prefix, epoch,
                    ctx, begin_epoch, rpn_epoch, rcnn_epoch, frequent, kv_store, work_load_list=None):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config.TRAIN.BG_THRESH_LO = 0.0

    logging.info('########## Stage 1 TRAIN RCNN WITH IMAGENET INIT AND RPN DETECTION')
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    train_rcnn(image_set, year, root_path, devkit_path, pretrained, epoch,
               output_prefix + 'rcnn1', ctx, begin_epoch, rcnn_epoch, frequent, kv_store, work_load_list, lr=0.001, net=net)


def alternate_train(image_set, test_image_set, year, root_path, devkit_path, net, pretrained, output_prefix, epoch,
                    ctx, begin_epoch, rpn_epoch, rcnn_epoch, frequent, kv_store, work_load_list=None):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config.TRAIN.BG_THRESH_LO = 0.0

    stages = []
    # stages.append('rpn1')
    # stages.append('test rpn1')
    # stages.append('rcnn1')
    # stages.append('rpn2')
    # stages.append('test rpn2')
    # stages.append('combine rpn2 rcnn1')
    stages.append('rcnn2')
    # stages.append('combine rpn2 rcnn2')

    logging.info('########## Stage 1 TRAIN RPN WITH IMAGENET INIT')
    # pretrained -> rpn1
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1
    if 'rpn1' in stages:
        train_rpn(image_set, year, root_path, devkit_path, pretrained, epoch,
                  output_prefix + 'rpn1', ctx, begin_epoch, rpn_epoch, frequent, kv_store, work_load_list, lr=0.001, net=net)

    logging.info('########## Stage 1 GENERATE RPN DETECTION')
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    if 'test rpn1' in stages:
        test_rpn(image_set, year, root_path, devkit_path, output_prefix + 'rpn1', rpn_epoch, ctx[0], net=net)

    logging.info('########## Stage 1 TRAIN RCNN WITH IMAGENET INIT AND RPN DETECTION')
    # pretrained -> rcnn1
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    if 'rcnn1' in stages:
        train_rcnn(image_set, year, root_path, devkit_path, pretrained, epoch,
                   output_prefix + 'rcnn1', ctx, begin_epoch, rcnn_epoch, frequent, kv_store, work_load_list, lr=0.001, net=net)

    logging.info('########## Stage 2 TRAIN RPN WITH RCNN INIT')
    # rcnn1 -> rpn2
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1
    config.TRAIN.FINETUNE = True
    if 'rpn2' in stages:
        train_rpn(image_set, year, root_path, devkit_path, output_prefix + 'rpn3-lr0005', 2,
                  output_prefix + 'rpn3', ctx, begin_epoch, rpn_epoch, frequent, kv_store, work_load_list, lr=0.0001, net=net, resume=True)

    logging.info('########## Stage 2 GENERATE RPN DETECTION')
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    if 'test rpn2' in stages:
        test_rpn(image_set, year, root_path, devkit_path, output_prefix + 'rpn2', rpn_epoch, ctx[0], net=net)

    logger.info('########## Stage 2 COMBINE RPN2 WITH RCNN1')
    # rpn2 + rcnn1 -> rcnn2
    if 'combine rpn2 rcnn1' in stages:
        combine_model(output_prefix + 'rpn3-lr0001', rpn_epoch, output_prefix + 'rcnn2', rcnn_epoch, output_prefix + 'rcnn3', 0)

    logger.info('########## Stage 2 TRAIN RCNN WITH RPN INIT AND DETECTION')
    # rcnn2 -> rcnn2
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    if 'rcnn2' in stages:
        train_rcnn(image_set, year, root_path, devkit_path, output_prefix + 'rcnn3-multiscaleroi', 4,
               output_prefix + 'rcnn3-multiscaleroi', ctx, begin_epoch, rcnn_epoch, frequent, kv_store, work_load_list, lr=0.0001, net=net, resume=True)

    logger.info('########## Stage 2 COMBINE RPN2 WITH RCNN2')
    if 'combine rpn2 rcnn2' in stages:
        combine_model(output_prefix + 'rpn2', rpn_epoch, output_prefix + 'rcnn2', rcnn_epoch, output_prefix + 'final', 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network')
    parser.add_argument('--image_set', dest='image_set', help='can be train, val or test',
                        default='train', type=str)
    parser.add_argument('--test_image_set', dest='test_image_set', help='can be test or val',
                        default='val', type=str)
    parser.add_argument('--year', dest='year', help='can be 2016',
                        default='2016', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='ILSVRCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'ILSVRCdevkit'), type=str)
    parser.add_argument('--net', dest='net', help='net name, can be vgg, resnet',
                        default='vgg', type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--output_prefix', dest='output_prefix', help='output model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--rpn_epoch', dest='rpn_epoch', help='end epoch of rpn training',
                        default=8, type=int)
    parser.add_argument('--rcnn_epoch', dest='rcnn_epoch', help='end epoch of rcnn training',
                        default=8, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    # ctx = [mx.cpu()]

    func = alternate_train
    # func = alt_train_rcnn

    func(args.image_set, args.test_image_set, args.year, args.root_path, args.devkit_path,
                    args.net, args.pretrained, args.output_prefix, args.epoch, ctx, args.begin_epoch, args.rpn_epoch, args.rcnn_epoch,
                    args.frequent, args.kv_store, args.work_load_list)
