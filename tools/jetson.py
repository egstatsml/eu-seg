#!/usr/bin/env python3

import sys

sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import scipy

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


from collections import namedtuple

import torch.onnx
import time

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)



net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval_bayes')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
weight_var = torch.load('weight_var.pt')
bias_var = torch.load('bias_var.pt')

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)#.cuda()

org_size = im.size()[2:]
new_size = [math.ceil(el / 64) * 32 for el in im.size()[2:]]
print('new size', new_size)

# inference
im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')



torch.onnx.export(net, im, "bisenet2_bayes.onnx", verbose=False)

net.half().cuda().eval()
im = im.half().cuda()
start_time = time.time()

with torch.no_grad():
    for i in range(0, 1000):
        logit_mean, logit_var = net(im)

end_time = time.time()
print(end_time - start_time)
print(logit_mean.shape)
print(logit_var.shape)


