import mxnet as mx
import numpy as np
# import ipdb as pdb

from rcnn.config import config


class AccuracyMetric(mx.metric.EvalMetric):
    def __init__(self, use_ignore=False, ignore=None, joint_rpn=False, name='Accuracy'):
        super(AccuracyMetric, self).__init__(name)
        self.use_ignore = use_ignore
        self.ignore = ignore
        self.has_rpn = config.TRAIN.HAS_RPN and config.TRAIN.END2END == 0
        if self.has_rpn:
            assert self.use_ignore and self.ignore is not None
        self.joint_rpn = joint_rpn

    def update(self, labels, preds):
        if self.has_rpn:
            pred_label = mx.ndarray.argmax_channel(preds[0]).asnumpy().astype('int32')
            label = labels[0].asnumpy().astype('int32')
            non_ignore_inds = np.where(label != self.ignore)
            pred_label = pred_label[non_ignore_inds]
            label = label[non_ignore_inds]
        else:
            if config.TRAIN.END2END == 0:
                last_dim = preds[0].shape[-1]
                pred_label = preds[0].asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
                label = labels[0].asnumpy().reshape(-1,).astype('int32')
            elif self.joint_rpn:
                # pdb.set_trace()
                # last_dim = preds[1].shape[-1]
                # pred_label = preds[1].asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
                # label = preds[0].asnumpy().reshape(-1,).astype('int32')
                pred_label = mx.ndarray.argmax_channel(preds[1]).asnumpy().astype('int32')
                label = labels[0].asnumpy().astype('int32')
                non_ignore_inds = np.where(label != self.ignore)
                pred_label = pred_label[non_ignore_inds]
                label = label[non_ignore_inds]
            else:
                last_dim = preds[3].shape[-1]
                pred_label = preds[3].asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
                label = preds[0].asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


class LogLossMetric(mx.metric.EvalMetric):
    def __init__(self, use_ignore=False, ignore=None, joint_rpn=False, name='LogLoss'):
        super(LogLossMetric, self).__init__(name)
        self.use_ignore = use_ignore
        self.ignore = ignore
        self.has_rpn = config.TRAIN.HAS_RPN and config.TRAIN.END2END == 0
        if self.has_rpn:
            assert self.use_ignore and self.ignore is not None
        self.joint_rpn = joint_rpn

    def update(self, labels, preds):
        if self.has_rpn:
            # pdb.set_trace()
            pred_cls = preds[0].asnumpy()[0]
            label = labels[0].asnumpy().astype('int32')[0]
            non_ignore_inds = np.where(label != self.ignore)[0]
            label = label[non_ignore_inds]
            cls = pred_cls[label, non_ignore_inds]
        else:
            if config.TRAIN.END2END == 0:
                last_dim = preds[0].shape[-1]
                pred_cls = preds[0].asnumpy().reshape(-1, last_dim)
                label = labels[0].asnumpy().reshape(-1,).astype('int32')
                cls = pred_cls[np.arange(label.shape[0]), label]
            elif self.joint_rpn:
                # pdb.set_trace()
                # last_dim = preds[1].shape[-1]
                # pred_cls = preds[1].asnumpy().reshape(-1, last_dim)
                # label = labels[0].asnumpy().reshape(-1,).astype('int32')
                # non_ignore_inds = np.where(label != self.ignore)[0]
                # label = label[non_ignore_inds]
                # cls = pred_cls[label, non_ignore_inds]
                pred_cls = preds[1].asnumpy()[0]
                label = labels[0].asnumpy().astype('int32')[0]
                non_ignore_inds = np.where(label != self.ignore)[0]
                label = label[non_ignore_inds]
                cls = pred_cls[label, non_ignore_inds]
            else:
                last_dim = preds[3].shape[-1]
                pred_cls = preds[3].asnumpy().reshape(-1, last_dim)
                label = preds[0].asnumpy().reshape(-1,).astype('int32')
                cls = pred_cls[np.arange(label.shape[0]), label]
        cls += config.EPS
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class SmoothL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, joint_rpn=False, name='SmoothL1Loss'):
        super(SmoothL1LossMetric, self).__init__(name)
        self.has_rpn = config.TRAIN.HAS_RPN and config.TRAIN.END2END != 1
        self.joint_rpn = joint_rpn

    def update(self, labels, preds):
        bbox_loss = preds[1].asnumpy()
        if self.has_rpn:
            bbox_loss = bbox_loss.reshape((bbox_loss.shape[0], -1))
        else:
            if config.TRAIN.END2END == 0:
                first_dim = bbox_loss.shape[0] * bbox_loss.shape[1]
                bbox_loss = bbox_loss.reshape(first_dim, -1)
            elif self.joint_rpn:
                bbox_loss = preds[2].asnumpy()
                first_dim = bbox_loss.shape[0]
                bbox_loss = bbox_loss.reshape(first_dim, -1)
            else:
                bbox_loss = preds[-1].asnumpy()
                first_dim = bbox_loss.shape[0] * bbox_loss.shape[1]
                bbox_loss = bbox_loss.reshape(first_dim, -1)
        self.num_inst += bbox_loss.shape[0]
        bbox_loss = np.sum(bbox_loss)
        self.sum_metric += bbox_loss
