#coding:utf8

vid_fn = 'data/ILSVRCdevkit/ILSVRC2016/devkit/data/map_vid.txt'
vid_dic = {}
with open(vid_fn) as f:
    for l in f:
        sp = l.strip().split()
        vid_dic[sp[0]] = sp[1]

det_fn = 'data/ILSVRCdevkit/ILSVRC2016/devkit/data/map_det.txt'
det_dic = {}
with open(det_fn) as f:
    for l in f:
        sp = l.strip().split()
        det_dic[sp[1]] = sp[0]

frameindex_fn = 'data/ILSVRCdevkit/ILSVRC2016/ImageSets/VID/test.txt-77-map'
frameindex_dic = {}
with open(frameindex_fn) as f:
    for l in f:
        sp = l.strip().split()
        frameindex_dic[sp[0]] = sp[1]

result_fn = 'data/ILSVRCdevkit/results/ILSVRC2016/VID/comp4_det_test.txt_multiscaleroi-epoch5-scale600-77'
fout = open(result_fn + '-transformed', 'w')
with open(result_fn) as f:
    for l in f:
        sp = l.strip().split()
        clsname = det_dic[sp[1]]
        if clsname not in vid_dic:
            continue
        ori_class = vid_dic[det_dic[sp[1]]]
        ori_frameindex = frameindex_dic[sp[0]]
        print >> fout, ori_frameindex, ori_class, ' '.join(sp[2:])

