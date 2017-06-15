import numpy as np
import cv2
import os
import cPickle
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
# import ipdb as pdb

from rcnn.config import config
from helper.processing import image_processing
from helper.processing.nms import nms


def pred_eval(detector, test_data, imdb, vis=False, start=0, num=-1, iter=1, suffix=''):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param detector: Detector
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :return:
    """
    assert not test_data.shuffle

    end = test_data.size
    if num > 0:
        end = min(end, start + num)
    print 'Generate detections from {} to {}'.format(start, end)
    ids = range(start, end)

    thresh = 0.05
    # limit detections to max_per_image over all classes
    max_per_image = 100
    roi_num = 300
    print 'using', roi_num, 'rois'

    bbox_refine = False
    bbox_refine_merge_stages = False
    print 'bbox_refine', bbox_refine, 'bbox_refine_merge_stages', bbox_refine_merge_stages

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # load pre-detected result
    # det_file = 'data/cache_rpn3/ilsvrc_2016_val/ilsvrc_2016_val_detections-0-20121.pkl-epoch9-1000proposal'
    # print 'loading', det_file
    # with open(det_file, 'rb') as f:
    #     all_boxes = cPickle.load(f)

    i = 0
    start_time = time.time()
    count = 0
    num_img = len(ids)
    # for databatch in test_data:
    for i in ids:
        if count % 10 == 0:
            print 'testing {}/{} {} seconds'.format(count, num_img, time.time() - start_time)
            start_time = time.time()

        print 'index', i
        databatch = test_data.get_i(i)
        if config.TEST.HAS_RPN:
            scores, boxes = detector.im_detect(databatch.data['data'], im_info=databatch.data['im_info'])
            scale = databatch.data['im_info'][0, 2]
        else:
            if vis:
                # vis_roi(databatch.data['data'], databatch.data['rois'])
                pass
            rois = databatch.data['rois'][:roi_num]
            scores, boxes = detector.im_detect(databatch.data['data'], roi_array=rois)

            # bbox refinement. use boxes with a score in top 300 as proposals
            if bbox_refine:
                refine_thresh = np.sort(scores, axis=None)[-roi_num - 1]
                proposal_indices = np.argwhere(scores > refine_thresh)
                refined_proposals = np.zeros((proposal_indices.shape[0], 5))

                for proposal_ind in range(refined_proposals.shape[0]):
                    pair = proposal_indices[proposal_ind]
                    refined_proposals[proposal_ind, 1:] = boxes[pair[0], pair[1] * 4: pair[1] * 4 + 4]

                # pdb.set_trace()
                if vis:
                    # 1st step regression result
                    im_path = imdb.image_path_from_index(imdb.image_set_index[i])
                    im = cv2.imread(im_path)
                    im_height = im.shape[0]
                    scale = float(databatch.data['data'].shape[2]) / float(im_height)
                    keep = np.argwhere(proposal_indices[:, 1] > 0).flatten()
                    vis_box(im, refined_proposals[keep, 1:] / scale)

                # pdb.set_trace()
                refined_scores, refined_boxes = detector.im_detect(databatch.data['data'], roi_array=refined_proposals)

                if bbox_refine_merge_stages: # merge boxes from 2 stages
                    scores = np.vstack((scores, refined_scores))
                    boxes = np.vstack((boxes, refined_boxes))
                else: # only use stage 2 boxes
                    scores = refined_scores
                    boxes = refined_boxes

                if vis:
                    # 2nd step regression result
                    refine_thresh = np.sort(refined_scores, axis=None)[-301]
                    proposal_indices = np.argwhere(refined_scores > refine_thresh)
                    refined_proposals = np.zeros((proposal_indices.shape[0], 5))

                    for proposal_ind in range(refined_proposals.shape[0]):
                        pair = proposal_indices[proposal_ind]
                        refined_proposals[proposal_ind, 1:] = refined_boxes[pair[0], pair[1] * 4: pair[1] * 4 + 4]
                    keep = np.argwhere(proposal_indices[:, 1] > 0).flatten()
                    vis_box(im, refined_proposals[keep, 1:] / scale)

            # we used scaled image & roi to train, so it is necessary to transform them back
            # visualization should also be from the original size
            im_path = imdb.image_path_from_index(imdb.image_set_index[i])
            im = cv2.imread(im_path)
            im_height = im.shape[0]
            scale = float(databatch.data['data'].shape[2]) / float(im_height)
        for j in range(1, imdb.num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j]
            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4] / scale
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
            keep = nms(cls_dets, config.TEST.NMS)
            all_boxes[j][i] = cls_dets[keep, :]
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            # pdb.set_trace()
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        boxes_this_image = [[]] + [all_boxes[j][i] for j in range(1, imdb.num_classes)]
        # pdb.set_trace()
        if vis:
            # visualize the testing scale
            for box in boxes_this_image:
                if isinstance(box, np.ndarray):
                    box[:, :4] *= scale
            vis_all_detection(databatch.data['data'], boxes_this_image,
                              imdb_classes=imdb.classes, thresh=0.5, index=i)
        count += 1

    cache_folder = os.path.join(imdb.cache_path, imdb.name)
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    det_file = os.path.join(cache_folder, imdb.name + '_detections{}-{}-{}.pkl'.format(suffix, start, end))
    print 'writing detection result to', det_file
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f)
    print 'wrote detection result to', det_file

    imdb.evaluate_detections(all_boxes, suffix=suffix + '-{}-{}'.format(start, num))


def vis_box(im, detections):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param imdb_classes: list of names in imdb
    :param thresh: threshold for valid detections
    :return:
    """
    plt.imshow(im)
    dets = detections
    for i in range(detections.shape[0]):
        color = (random.random(), random.random(), random.random())  # generate a random color
        bbox = dets[i]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='blue', linewidth=1)
        plt.gca().add_patch(rect)
        plt.gca().text(bbox[0], bbox[1] - 2,
                       '{}'.format(i),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()

def vis_roi(im_array, detections):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param imdb_classes: list of names in imdb
    :param thresh: threshold for valid detections
    :return:
    """
    im = image_processing.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    dets = detections
    for i in range(10):
        color = (random.random(), random.random(), random.random())  # generate a random color
        bbox = dets[i, 1:]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.gca().text(bbox[0], bbox[1] - 2,
                       '{}'.format(i),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def vis_all_detection(im_array, detections, imdb_classes=None, thresh=0.7, index=0):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param imdb_classes: list of names in imdb
    :param thresh: threshold for valid detections
    :return:
    """
    im = image_processing.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j in range(1, len(imdb_classes)):
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            if score > thresh:
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=3.5)
                plt.gca().add_patch(rect)
                plt.gca().text(bbox[0], bbox[1] - 2,
                               '{:s} {:.3f}'.format(imdb_classes[j], score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()
    # plt.savefig('detections/{}'.format(index))
    # plt.cla()
