from torch.utils.data import Dataset
import imageio as io
import cv2
import torch
from scipy.ndimage import filters
from numpy import *
import scipy.io as sio
import os
import numpy as np


class ImageList_DrFixD_rainy(Dataset):
    def __init__(self, args, imgs, for_train=False):
        self.root = args.root
        self.imgs = imgs

        self.img_shape = args.img_shape
        self.for_train = for_train

    def __getitem__(self, index):
        img_name = self.imgs[index]
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])
        img_name = 'trafficframe/' + self.imgs[index]

        image_name = os.path.join(self.root, img_name)
        img = io.imread(image_name)
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0

        lab_img = getLabel(self.root, vid_index, frame_index, self.img_shape)

        if self.for_train:
            img, lab_img = transform(img, lab_img)

        img = img.transpose(2, 0, 1)
        lab_img = lab_img[None, ...]
        img = np.ascontiguousarray(img)

        lab_img = np.ascontiguousarray(lab_img)

        img_tensor, lab_img_tensor = torch.from_numpy(img), torch.from_numpy(lab_img)
        # del img, lab_img
        return img_tensor, lab_img_tensor

    def __len__(self):
        return len(self.imgs)


class ImageList_DrFixD_rainy_Continuous(Dataset):
    def __init__(self, args, imgs, for_train=False):
        self.args = args
        self.root = args.root
        self.imgs = imgs
        self.for_train = for_train
        self.seq_len = args.seq_len

    def __getitem__(self, index):
        img_name = self.imgs[index]

        imgarr = []
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])
        for m in range(self.seq_len):
            fra_index = frame_index - m
            vid_index = vid_index
            img_name = 'trafficframe/' + '%02d' % (vid_index) + "/" + '%06d' % (fra_index) + '.jpg'
            image_name = os.path.join(self.root, img_name)
            img = io.imread(image_name)
            img = cv2.resize(img, self.args.img_shape, interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            imgarr.append(torch.from_numpy(img))
        imgarr = torch.stack(imgarr)
        imgarr = imgarr.float() / 255.0
        mask = getLabel_Continuous(self.root, vid_index, frame_index, self.args.img_shape)

        if self.for_train:
            img, mask = transform(img, mask)

        mask = mask[None, ...]
        mask = np.ascontiguousarray(mask)
        return imgarr, torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)


def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y


def getLabel(root, vid_index, frame_index, img_shape):
    fixdatafile = (root + '/fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    fix_x = fix_x.astype('int')
    fix_y = fix_y.astype('int')
    mask = np.zeros((720, 1280), dtype='float32')

    for i in range(len(fix_x)):
        if (fix_x[i] < 0) or (fix_x[i] >= 720) or (fix_y[i] < 0) or (fix_y[i] >= 1280):
            continue
        mask[int(fix_x[i]), int(fix_y[i])] = 1

    mask = filters.gaussian_filter(mask, 40)
    mask = np.array(mask, dtype='float32')
    mask = cv2.resize(mask, img_shape, interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0

    if mask.max() == 0:
        pass
        # print(mask.max())
    else:
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def getLabel_Continuous(root, vid_index, frame_index, img_shape):
    fixdatafile = (root + '/fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    fix_x = fix_x.astype('int')
    fix_y = fix_y.astype('int')
    mask = np.zeros((720, 1280), dtype='float32')
    # print(len(fix_x),vid_index, frame_index)
    for i in range(len(fix_x)):
        # print(fix_x[i],fix_y[i])
        mask[fix_x[i], fix_y[i]] = 1
    mask = filters.gaussian_filter(mask, 40)
    mask = np.array(mask, dtype='float32')
    mask = cv2.resize(mask, img_shape, interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0

    if mask.max() == 0:
        # print(mask.max())
        # print img_name
        pass
    else:
        mask = mask / mask.max()
    return mask