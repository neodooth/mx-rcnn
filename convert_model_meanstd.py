#coding:utf8

import sys
import cPickle as cp
import mxnet as mx
from utils.load_model import load_checkpoint
from utils.save_model import save_checkpoint


mean_stds = cp.load(open('model/resnet-101-fix123/rcnn3_means_stds.pkl', 'rb'))
means = mean_stds[0]
stds = mean_stds[1]

prefix = sys.argv[1]
epoch = int(sys.argv[2])

raw_input('convert {} epoch {}'.format(prefix, epoch))
arg_params, aux_params = load_checkpoint(prefix, epoch)
arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds)).T
arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
save_checkpoint(prefix + '-edit_params', epoch, arg_params, aux_params)
