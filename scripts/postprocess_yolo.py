#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = "person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball glove skateboard surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed dining table toilet tv laptop mouse remote keyboard cell phone microwave oven toaster sink refrigerator book clock vase scissors teddy bear hair drier toothbrush".split()

def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    a = (box[2]-box[0]) * (box[3]-box[1])
    b = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / np.maximum(a+b-inter, 1e-9)

p = argparse.ArgumentParser()
p.add_argument("image", type=Path); p.add_argument("output", type=Path)
p.add_argument("--conf", type=float, default=.25); p.add_argument("--iou", type=float, default=.45)
args = p.parse_args()
pred = np.fromfile(args.output, np.float32).reshape(25200, 85)
classes = pred[:,5:].argmax(1); scores = pred[:,4] * pred[np.arange(len(pred)), classes+5]
keep = scores >= args.conf
pred, classes, scores = pred[keep], classes[keep], scores[keep]
xywh = pred[:,:4]; boxes = np.column_stack((xywh[:,0]-xywh[:,2]/2, xywh[:,1]-xywh[:,3]/2, xywh[:,0]+xywh[:,2]/2, xywh[:,1]+xywh[:,3]/2))
selected = []
for cls in np.unique(classes):
    ids = np.where(classes == cls)[0]; ids = ids[np.argsort(scores[ids])[::-1]]
    while len(ids):
        selected.append(ids[0]); ids = ids[1:][iou(boxes[ids[0]], boxes[ids[1:]]) <= args.iou]
w,h = Image.open(args.image).size; scale=min(640/w,640/h); nw,nh=round(w*scale),round(h*scale); left,top=(640-nw)//2,(640-nh)//2
for i in sorted(selected, key=lambda x: scores[x], reverse=True):
    b=boxes[i].copy(); b[[0,2]]=(b[[0,2]]-left)/scale; b[[1,3]]=(b[[1,3]]-top)/scale; b[[0,2]]=np.clip(b[[0,2]],0,w); b[[1,3]]=np.clip(b[[1,3]],0,h)
    print(f"{NAMES[classes[i]]:12s} {scores[i]:.4f} box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")
