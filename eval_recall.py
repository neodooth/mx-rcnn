import numpy as np
import scipy.sparse
from helper.processing.bbox_regression import bbox_overlaps
import time


def evaluate_recall(roidb, candidate_boxes=None, thresholds=None, area='all', limit=None):
    """
    evaluate detection proposal recall metrics
    record max overlap value for each gt box; return vector of overlap values
    :param roidb: used to evaluate
    :param candidate_boxes: if not given, use roidb's non-gt boxes
    :param thresholds: array-like recall threshold
    :param area: index in area ranges
    :param limit: limit of bounding box evaluated
    :return: None
    ar: average recall, recalls: vector recalls at each IoU overlap threshold
    thresholds: vector of IoU overlap threshold, gt_overlaps: vector of all ground-truth overlaps
    """
    areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
             '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
    area_ranges = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2],
                   [96**2, 128**2], [128**2, 256**2], [256**2, 512**2], [512**2, 1e5**2]]
    assert areas.has_key(area), 'unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for i in range(349319):
        # check for max_overlaps == 1 avoids including crowd annotations
        max_gt_overlaps = roidb[i]['gt_overlaps'].toarray().max(axis=1)
        gt_inds = np.where((roidb[i]['gt_classes'] > 0) & (max_gt_overlaps == 1))[0]
        gt_boxes = roidb[i]['boxes'][gt_inds, :]
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
        valid_gt_inds = np.where((gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
        gt_boxes = gt_boxes[valid_gt_inds, :]
        num_pos += len(valid_gt_inds)

        if candidate_boxes is None:
            # default is use the non-gt boxes from roidb
            non_gt_inds = np.where(roidb[i]['gt_classes'] == 0)[0]
            boxes = roidb[i]['boxes'][non_gt_inds, :]
        else:
            boxes = candidate_boxes[i]
        if boxes.shape[0] == 0:
            continue
        if limit is not None and boxes.shape[0] > limit:
            boxes = boxes[:limit, :]

        overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        for j in range(gt_boxes.shape[0]):
            # find which proposal maximally covers each gt box
            argmax_overlaps = overlaps.argmax(axis=0)
            # get the IoU amount of coverage for each gt box
            max_overlaps = overlaps.max(axis=0)
            # find which gt box is covered by most IoU
            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()
            assert (gt_ovr >= 0)
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the IoU coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert (_gt_overlaps[j] == gt_ovr)
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        # append recorded IoU coverage level
        gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    gt_overlaps = np.sort(gt_overlaps)
    if thresholds is None:
        step = 0.05
        thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)

    # compute recall for each IoU threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    ar = recalls.mean()

    # print results
    print 'average recall: {:.3f}'.format(ar)
    for threshold, recall in zip(thresholds, recalls):
        print 'recall @{:.2f}: {:.3f}'.format(threshold, recall)
