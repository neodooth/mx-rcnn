#coding:utf8

import os
import numpy as np
import scipy.sparse
import scipy.io
import cPickle
from imdb_imagenet import IMDBImagenet
from voc_eval import voc_eval
from helper.processing.bbox_process import unique_boxes, filter_small_boxes


class ILSVRC(IMDBImagenet):
    def __init__(self, image_set, year, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val
        :param year: 2016
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        super(ILSVRC, self).__init__('ilsvrc_' + year + '_' + image_set)  # set self.name
        self.image_set = image_set
        self.year = year
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'ILSVRC' + year)

        self.classes = self.load_imagenet_label2words()
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        print len(self.image_set_index), 'images'
        self.image_sizes = self.load_image_sizes() # [(w, h), ...]
        self.remove_negative_data()
        self.num_images = len(self.image_set_index)

        self.extra_ratio_image_index_dic = self.replace_extra_ratio_data()

        self.config = {'comp_id': 'comp4',
                       'min_size': 2}

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def load_imagenet_label2words(self):
        filename = os.path.join(self.data_path, 'devkit', 'data', 'map_det.txt')
        assert os.path.exists(filename), \
               'label2words not found at: {}'.format(filename)
        with open(filename) as f:
            lines = f.read().splitlines()
        classes = tuple([l.split(' ')[0] for l in lines])
        result = ('__background__',) + classes
        return result

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'DET', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()] #[:150000]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'Data', 'DET', self.image_set, index + '.JPEG')
        if index in self.extra_ratio_image_index_dic:
            image_file = os.path.join(self.devkit_path, 'data_to_replace', 'Data', 'DET', self.image_set, index + '.JPEG')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def load_image_sizes(self):
        filename = os.path.join(self.devkit_path, 'image_sizes_' + self.image_set + '.txt')
        assert os.path.exists(filename), \
               'image_sizes_{}.txt not found at: {}'.format(self.image_set, filename)
        with open(filename) as f:
            lines = f.read().splitlines()
        sizes = []
        for l in lines:
            sp = l.split(' ')
            sizes.append((int(sp[0]), int(sp[1])))
        return sizes

    def remove_negative_data(self):
        positive_data_indices = [ind for (ind, name) in zip(range(len(self.image_set_index)), self.image_set_index) if 'extra' not in name]
        self.image_set_index = [self.image_set_index[x] for x in positive_data_indices]
        self.image_sizes = [self.image_sizes[x] for x in positive_data_indices]
        print 'Removed negative data,', len(self.image_set_index), len(self.image_sizes), 'remained'
        for i in range(len(self.image_set_index)):
            print i, self.image_set_index[i]

    def replace_extra_ratio_data(self):
        if self.image_set != 'train':
            return {}
        filename = os.path.join(self.devkit_path, 'image_sizes_train_replace.txt')
        assert os.path.exists(filename), \
               'image_sizes_train_replace.txt not found at: {}'.format(filename)
        print 'Replace extra ratio data according to {}'.format(filename)

        with open(filename) as f:
            lines = f.read().splitlines()
        d = {}
        extras = {}
        for i, s in enumerate(self.image_set_index):
            d[s] = i
        for l in lines:
            sp = l.split(' ')
            w, h, im_name = sp[0], sp[1], sp[2]
            if im_name in d:
                self.image_sizes[d[im_name]] = (int(w), int(h))
                extras[im_name] = d[im_name]
                print 'extra', d[im_name], im_name, self.image_sizes[d[im_name]]
        return extras

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_imagenet_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_imagenet_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET
        if index in self.extra_ratio_image_index_dic:
            print 'Extra large ratio image', index
            filename = os.path.join(self.devkit_path, 'data_to_replace', 'Annotations', 'DET', self.image_set, index + '.xml')
        else:
            filename = os.path.join(self.data_path, 'Annotations', 'DET', self.image_set, index + '.xml')

        # Images without any annotated objects may not have a corresponding xml file.
        if not os.path.exists(filename):
            print 'Did not found', filename
            tree = ET.fromstring('<useless></useless>')
        else:
            tree = ET.parse(filename)

        objs = tree.findall('object')
        num_objs = len(objs)
        if num_objs == 0:
            print 'No objects found from {}'.format(index)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        img_w = int(tree.find('size').find('width').text)
        img_h = int(tree.find('size').find('height').text)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            if x1 >= x2 or y1 >= y2:
                print "Malformed bounding box wxh:{} {} {} {} {} {} {}".format(
                        img_w, img_h, x1, x2, y1, y2, index)
                continue

            if x2 > img_w - 1:
                x2 = img_w - 1
            if y2 > img_h - 1:
                y2 = img_h - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def roidb(self, gt_roidb):
        return self.selective_search_roidb(gt_roidb)

    def load_selective_search_roidb(self, gt_roidb):
        """
        turn selective search proposals into selective search roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        matfile = os.path.join(self.root_path, 'selective_search_data', self.name + '.mat')
        assert os.path.exists(matfile), 'selective search data does not exist: {}'.format(matfile)
        raw_data = scipy.io.loadmat(matfile)['boxes'].ravel()  # original was dict ['images', 'boxes']

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1  # pascal voc dataset starts from 1.
            keep = unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_roidb(self, gt_roidb):
        """
        get selective search roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :return: roidb of selective search (ground truth included)
        """
        cache_file = os.path.join(self.cache_path, self.name + '_ss_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self.image_set != 'test':
            ss_roidb = self.load_selective_search_roidb(gt_roidb)
            roidb = IMDBImagenet.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self.load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def load_rpn_roidb(self, gt_roidb):
        """
        turn rpn detection boxes into roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        rpn_file = os.path.join(self.root_path, 'rpn_data', self.name + '_rpn.pkl')
        print 'loading {}'.format(rpn_file)
        assert os.path.exists(rpn_file), 'rpn data not found at {}'.format(rpn_file)
        with open(rpn_file, 'rb') as f:
            box_list = cPickle.load(f)
        print 'loaded {}'.format(rpn_file)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_roidb(self, gt_roidb):
        """
        get rpn roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :return: roidb of rpn (ground truth included)
        """
        if self.image_set != 'test':
            rpn_roidb = self.load_rpn_roidb(gt_roidb)
            roidb = IMDBImagenet.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            print 'rpn database need not be used in test'
            roidb = self.load_rpn_roidb(gt_roidb)
        return roidb

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'ILSVRC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'ILSVRC' + self.year, 'DET')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_imagenet_results(detections)

    def get_result_file_template(self):
        res_file_folder = os.path.join(self.devkit_path, 'results', 'ILSVRC' + self.year, 'DET')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_imagenet_results(self, all_boxes):
        filename = self.get_ilsvrc_results_file_template()
        with open(filename, 'wt') as f:
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                print 'Writing {} ILSVRC results file'.format(cls)
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the ILSVRCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, cls_ind, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
