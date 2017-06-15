import argparse
import logging
import os
import gc
# import ipdb as pdb

import mxnet as mx

from rcnn.callback import Speedometer
from rcnn.config import config
from rcnn.loader import ROIIter
from rcnn.metric import AccuracyMetric, LogLossMetric, SmoothL1LossMetric
from rcnn.module import MutableModule
from rcnn.symbol import get_vgg_rcnn
from rcnn.symbol_resnet101 import get_resnet_rcnn
from utils.load_data import load_ilsvrc_ss_roidb
from utils.load_data import load_ilsvrc_rpn_roidb
from utils.load_model import load_checkpoint, load_param
from utils.save_model import save_checkpoint


def copy_unshared_resnet_params(args, auxs, postfixes):
    for a in [args, auxs]:
        for k in a.keys():
            for pf in postfixes:
                if 'res5' in k or 'bn5' in k and 'branch' in k and 'relu' not in k and 'pool' not in k and pf not in k:
                    sp = k.split('_')
                    if 'moving' in sp: # bn
                        newk = '_'.join(sp[:-2]) + '_' + pf + '_' + '_'.join(sp[-2:])
                    else:
                        newk = '_'.join(sp[:-1]) + '_' + pf + '_' + sp[-1]
                    print 'copy', k, newk
                    a[newk] = mx.nd.array(a[k].asnumpy())
    # pdb.set_trace()


def train_rcnn(image_set, year, root_path, devkit_path, pretrained, epoch,
               prefix, ctx, begin_epoch, end_epoch, frequent, kv_store,
               work_load_list=None, resume=False, proposal='rpn', lr=0.001, net='vgg'):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    roi_scales = [2]

    # load symbol
    sym = eval('get_' + net + '_rcnn')(roi_scales=roi_scales)

    # bs = config.TRAIN.BATCH_IMAGES * len(ctx)
    # s = {'data': (bs, 3, 1000, 1000),'rois': (bs, 64, 5), 'label': (bs, 64), 'bbox_target': (bs, 64, 4 * 201), 'bbox_inside_weight': (bs, 64, 4 * 201), 'bbox_outside_weight': (bs, 64, 4 * 201)}
    # for sc in roi_scales:
    #     s['roi_{}x'.format(sc)] = (config.TRAIN.BATCH_SIZE * len(ctx), 5)
    # net_img = mx.visualization.plot_network(sym, shape=s, node_attrs={"fixedsize":"fasle"})
    # print net_img
    # net_img.render('net_image/' + net + '/train_rcnn-multiscaleroi')

    # setup multi-gpu
    config.TRAIN.BATCH_IMAGES *= len(ctx)
    config.TRAIN.BATCH_SIZE *= len(ctx)

    # load pretrained
    print 'loading pretrained', pretrained, 'epoch', epoch
    args, auxs = load_param(pretrained, epoch, convert=True)

    print 'learning rate', lr
    print 'resume?', resume

    # load training data
    voc, roidb, means, stds = eval('load_ilsvrc_' + proposal + '_roidb')(image_set, year, root_path, devkit_path) #, flip=True)
    gc.collect()
    train_data = ROIIter(roidb, batch_size=config.TRAIN.BATCH_IMAGES, shuffle=True, mode='train',
                         ctx=ctx, work_load_list=work_load_list, roi_scales=roi_scales)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, 1000, 1000))]

    # initialize params
    if not resume:
        input_shapes = {k: v for k, v in train_data.provide_data + train_data.provide_label}
        arg_shape, _, _ = sym.infer_shape(**input_shapes)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
        args['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        args['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
        args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])
        # args['shared_conv_reduce_weight'] = mx.random.normal(0, 0.01, shape=(1024, 1536, 1, 1))
        # for sc in roi_scales:
        #     args['shared_conv_reduce_roi_{}x_weight'.format(sc)] = mx.random.normal(0, 0.01, shape=(1024, 1536, 1, 1))
        postfixes = ['roi_{}x'.format(sc) for sc in roi_scales]
        copy_unshared_resnet_params(args, auxs, postfixes)

    # prepare training
    if net == 'vgg':
        if config.TRAIN.FINETUNE:
            fixed_param_prefix = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        else:
            fixed_param_prefix = ['conv1', 'conv2']
    elif net == 'resnet':
        if config.TRAIN.FINETUNE:
            fixed_param_prefix = ['conv1', 'res2', 'res3', 'res4']
        else:
            fixed_param_prefix = ['conv1', 'res2', 'res3']
        fixed_param_prefix.append('bn')
    else:
        print 'Unknown net', net, ', use VGG fixed_aram'
        if config.TRAIN.FINETUNE:
            fixed_param_prefix = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        else:
            fixed_param_prefix = ['conv1', 'conv2']
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    batch_end_callback = Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    if config.TRAIN.HAS_RPN is True:
        eval_metric = AccuracyMetric(use_ignore=True, ignore=-1)
        cls_metric = LogLossMetric(use_ignore=True, ignore=-1)
    else:
        eval_metric = AccuracyMetric()
        cls_metric = LogLossMetric()
    bbox_metric = SmoothL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        # 'lr_scheduler': mx.lr_scheduler.FactorScheduler(30000, 0.1),
                        'rescale_grad': (1.0 / config.TRAIN.BATCH_SIZE)}

    # train
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=work_load_list,
                        max_data_shapes=max_data_shape, fixed_param_prefix=fixed_param_prefix)
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kv_store,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=args, aux_params=auxs, begin_epoch=begin_epoch, num_epoch=end_epoch)

    # edit params and save
    # for epoch in range(begin_epoch + 1, end_epoch + 1):
    #     arg_params, aux_params = load_checkpoint(prefix, epoch)
    #     arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds)).T
    #     arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
    #     save_checkpoint(prefix + '-edit_params', epoch, arg_params, aux_params)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN Network')
    parser.add_argument('--image_set', dest='image_set', help='can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'rcnn'), type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=8, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--finetune', dest='finetune', help='second round finetune', action='store_true')
    parser.add_argument('--resume', dest='resume', help='continue training', action='store_true')
    parser.add_argument('--proposal', dest='proposal', help='can be ss for selective search or rpn',
                        default='rpn', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    if args.finetune:
        config.TRAIN.FINETUNE = True
    train_rcnn(args.image_set, args.year, args.root_path, args.devkit_path, args.pretrained, args.epoch,
               args.prefix, ctx, args.begin_epoch, args.end_epoch, args.frequent,
               args.kv_store, args.work_load_list, args.resume, args.proposal)
