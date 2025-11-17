import rawpy
import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

def raw_rgb(input):
    raw = rawpy.imread(input)
    rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=16)
    raw.close()
    return rgb


input_dir = '/home/xd/liulu/LLIE/Sony/Sony/short/'
gt_dir = '/home/xd/liulu/LLIE/Sony/Sony/long/'

ot_input_dir = '/home/xd/liulu/LLIE/Sony/sony_png/short/'
ot_gt_dir = '/home/xd/liulu/LLIE/Sony/sony_png/long/'

train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    train_ids.append(int(test_fn[0:5]))




for i in range(len(input)):

    ot_input = raw_rgb(input)
    ot_gt = raw_rgb(gt)
    ot_input_names = ot_input_dir + '{}.png'.format(str(i).zfill(1))
    ot_gt_names = ot_gt_dir + '{}.png'.format(str(i).zfill(1))
    imageio.imsave(ot_input_names, ot_input)
    imageio.imsave(ot_gt_names, ot_gt)