#!/usr/bin/env python
import numpy as np
import cPickle as cp
import sys

IMAGE_NUM = 143056
IOU = 0.4
all_scores = []
for i in range(IMAGE_NUM):
  scores = [[] for i in range(200)]
  all_scores.append(scores)
with open(sys.argv[1], 'r') as fp:
  for l in fp:
    line = l.strip().split(' ')
    img_id = int(line[0]) - 1
    idx = int(line[1]) - 1
    all_scores[img_id][idx].append(map(lambda x:float(x), line[2:]))

all_scores1 = []
for i in range(IMAGE_NUM):
  scores = [[] for i in range(200)]
  all_scores1.append(scores)
with open(sys.argv[2], 'r') as fp:
  for l in fp:
    line = l.strip().split(' ')
    img_id = int(line[0]) - 1
    idx = int(line[1]) - 1
    all_scores1[img_id][idx].append(map(lambda x:float(x), line[2:]))

# all_scores2 = []
# for i in range(IMAGE_NUM):
#   scores = [[] for i in range(200)]
#   all_scores2.append(scores)
# with open(sys.argv[3], 'r') as fp:
#   for l in fp:
#     line = l.strip().split(' ')
#     img_id = int(line[0]) - 1
#     idx = int(line[1]) - 1
#     all_scores2[img_id][idx].append(map(lambda x:float(x), line[2:]))

def cal_iou(box1, box2):
  """calculate IoU of two boxes.
  box1,box2: tuple,(xmin1,ymin1,xmax1,ymax1)
  """
  width = min(box1[2],box2[2]) - max(box1[0],box2[0]) +1
  height = min(box1[3],box2[3]) - max(box1[1],box2[1]) +1
  if width < 0 or height < 0:
    return 0
  else:
    S_inter = width*height
    S_box1 = (box1[2]-box1[0]+1) * (box1[3]-box1[1]+1)
    S_box2 = (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1)
    S_union = S_box1 + S_box2 - S_inter
    IoU = S_inter / S_union
    return IoU

wr = open(sys.argv[3], 'w')
for m in range(len(all_scores)):
  scores = all_scores[m]
  scores1 = all_scores1[m]
  # scores2 = all_scores2[m]
  if m % 1000 == 0:
    print '{}/{}'.format(str(m+1), IMAGE_NUM)
  for i in range(200):
    tmp_scores = np.require(scores[i], dtype=np.float32)
    tmp_scores1 = np.require(scores1[i], dtype=np.float32)
    # tmp_scores2 = np.require(scores2[i], dtype=np.float32)
    if len(tmp_scores) == 0 and len(tmp_scores) == 0 and len(tmp_scores) == 0:
      continue
    inds = np.ones((len(tmp_scores), ))
    inds1 = np.ones((len(tmp_scores1), ))
    # inds2 = np.ones((len(tmp_scores2), ))
    for j in range(len(tmp_scores)):
      box1 = tmp_scores[j][1:]
      merge_boxes = []
      for k in range(len(tmp_scores1)):
        box2 = tmp_scores1[k][1:]
        iou = cal_iou(box1, box2)
        if iou > 0.5:
          # print i+1, tmp_scores[j][0], tmp_scores1[k][0], iou
          merge_boxes.append(box2)
          inds1[k] = 0
      # for k in range(len(tmp_scores2)):
      #   box2 = tmp_scores2[k][1:]
      #   iou = cal_iou(box1, box2)
      #   if iou > IOU:
      #     # print i+1, tmp_scores[j][0], tmp_scores1[k][0], iou
      #     merge_boxes.append(box2)
      #     inds2[k] = 0
      if len(merge_boxes) > 0:
        inds[j] = 0
        for tmp_box in merge_boxes:
          box1[0] += tmp_box[0]
          box1[1] += tmp_box[1]
          box1[2] += tmp_box[2]
          box1[3] += tmp_box[3]
        # print len(merge_boxes)
        box1 /= (len(merge_boxes)+1)
        wr.write("{} {} {:.5f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(str(m+1), str(i+1), tmp_scores[j][0], box1[0], box1[1], box1[2], box1[3]))
    # print inds, inds1, inds2
    idxes = np.where(inds[:] > 0)
    el_scores = tmp_scores[idxes]
    idxes1 = np.where(inds1[:] > 0)
    el_scores1 = tmp_scores1[idxes1]
    # idxes2 = np.where(inds2[:] > 0)
    # el_scores2 = tmp_scores2[idxes2]
    tmp_merge_scores = [el_scores, el_scores1]#, el_scores2]
    merge_scores = []
    for u in tmp_merge_scores:
      if u == []:
        continue
      for u_idx in range(u.shape[0]):
        merge_scores.append(u[u_idx])
    # print merge_scores
    merge_scores = np.require(merge_scores)
    if merge_scores.shape[0] == 0:
      continue
    # print el_scores.shape, el_scores1.shape, el_scores2.shape
    topn = merge_scores.shape[0]
    # topn /= 2
    # if topn == 0:
    #   topn += 1
    # topn += 1
    el_inds = np.argsort(-merge_scores[:, 0])[0:topn]
    for idx in el_inds:
      score = merge_scores[idx][0]
      box = merge_scores[idx][1:]
      wr.write("{} {} {:.5f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(str(m+1), str(i+1), score, box[0], box[1], box[2], box[3]))

wr.close()

