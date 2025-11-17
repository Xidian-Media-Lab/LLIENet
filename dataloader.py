import torch
from torchvision import datasets, transforms
from torch.utils import data
import os
from cwan.cwannet.utils.rgb2lab import LAB
from os import listdir
from PIL import Image
import glob
import numpy as np
import cv2
from saturation import sat

class TrainImageFolder(datasets.ImageFolder):
    # def __init__(self, imagedir, gtdir, transform=True):

    def __init__(self, imagedir, size_crop, stride_crop, transform=None):   #with crop
        super(TrainImageFolder, self).__init__(imagedir)
        self.image_high = glob.glob(os.path.join(imagedir,'long','%05d_00*.ARW'))
        self.image_low = glob.glob(os.path.join(imagedir,'short','%05d_00*.ARW'))
        # self.image_attmap = glob.glob(os.path.join(imagedir,'attmap/attmap','*.png'))

        self.image_high.sort()
        self.image_low.sort()
        # self.image_attmap.sort()

        self.transform = transform

        # if need image crop
        self.new_im_high, list_im_high = img_crop_rgb(self.image_high, size_crop, stride_crop)
        self.new_im_low, list_im_low = img_crop_rgb(self.image_low, size_crop, stride_crop)
        # self.new_map, list_map = img_crop_attmap(self.image_attmap, size_crop, stride_crop)

        # self.new_im_940, list_im_940 = img_crop_gray(self.image_940, size_crop, stride_crop)
        # self.new_gt, list_gt = img_crop_rgb(self.gt_filenames, size_crop, stride_crop)
        del self.new_im_high[list_im_high:len(self.new_im_high)]
        del self.new_im_low[list_im_low:len(self.new_im_low)]
        # del self.new_map[list_map:len(self.new_map)]




    def __len__(self):
        # return len(self.image_filenames)   # without crop
        return len(self.new_im_high)   # with crop

    def __getitem__(self, index):
        img_high = self.new_im_high[index]
        img_low = self.new_im_low[index]
        # attmap = self.new_map[index]

        # img = Image.open(path_set).convert('L')
        # img_gt = Image.open(path_gt).convert('RGB')
        img_high = self.transform(img_high)
        img_low = self.transform(img_low)
        # attmap = self.transform(attmap)
        # img_high = torch.from_numpy(img_high.transpose(2,0,1)).float()
        # img_low = torch.from_numpy(img_low.transpose(2,0,1)).float()
        # attmap = torch.from_numpy(attmap.transpose(2,0,1)).float()


        return img_high, img_low

def img_crop_gray(img_list, size, stride):
    """
    :param img_list: 图片名的列表
    :param size: 所裁图片块的大小，即 patch_size
    :param stride: 步长
    :return: 存储所裁图片块的列表，每个块都是ndarray格式
    """
    img_patches = [None]*250000
    index = 0
    for i in range(len(img_list)):
        img = Image.open(img_list[i]).convert('L')   # 这里是PIL格式
        img = np.array(img)
        # img = np.expand_dims(np.array(img), axis=2)    # 转化为ndarray, H * W * C
        for h in range(1, img.shape[0]-size+1, stride):
            for w in range(1, img.shape[1]-size+1, stride):
                img_patch = img[h:h+size, w:w+size]
                img_patches[index] = img_patch
                index += 1
    return img_patches, index

def img_crop_rgblow(img_list, size, stride):
    """
    :param img_list: 图片名的列表
    :param size: 所裁图片块的大小，即 patch_size
    :param stride: 步长
    :return: 存储所裁图片块的列表，每个块都是ndarray格式
    """
    img_patches = [None]*250000
    index = 0
    for i in range(len(img_list)):
        # img = Image.open(img_list[i]).convert('YCbCr')   # 这里是PIL格式
        # img = np.array(img)
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        key = 0.5 + 0.4*np.random.random()
        key = -key
        img = sat(img,key)
        # img = np.expand_dims(np.array(img), axis=2)    # 转化为ndarray, H * W * C
        for h in range(1, img.shape[0]-size+1, stride):
            for w in range(1, img.shape[1]-size+1, stride):
                img_patch = img[h:h+size, w:w+size, :]
                img_patches[index] = img_patch
                index += 1
    return img_patches, index

def img_crop_rgb(img_list, size, stride):
    """
    :param img_list: 图片名的列表
    :param size: 所裁图片块的大小，即 patch_size
    :param stride: 步长
    :return: 存储所裁图片块的列表，每个块都是ndarray格式
    """
    img_patches = [None]*250000
    index = 0
    for i in range(len(img_list)):
        # img = Image.open(img_list[i]).convert('YCbCr')   # 这里是PIL格式
        # img = np.array(img)
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = np.expand_dims(np.array(img), axis=2)    # 转化为ndarray, H * W * C
        for h in range(1, img.shape[0]-size+1, stride):
            for w in range(1, img.shape[1]-size+1, stride):
                img_patch = img[h:h+size, w:w+size, :]
                img_patches[index] = img_patch
                index += 1
    return img_patches, index

def dataloader(traindir, batch_size, num_workers,size_crop, stride_crop):
    transform = transforms.Compose([
        transforms.ToTensor()])

    train_dataset = TrainImageFolder(traindir, size_crop, stride_crop, transform=transform)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,pin_memory=True,
                                   num_workers=num_workers)
    return train_loader