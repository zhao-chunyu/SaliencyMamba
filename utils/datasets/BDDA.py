from torch.utils.data import Dataset
import imageio as io
import cv2
import torch
from numpy import *
import os
import numpy as np


class ImageList_BDDA(Dataset):
    def __init__(self, args, imgs, for_train=False):
        self.root = args.root
        self.imgs = imgs
        self.label_root = args.root
        self.labels = imgs
        self.img_shape = args.img_shape
        self.for_train = for_train

    def __getitem__(self, index):
        img_name = 'camera_frames/' + self.imgs[index]
        img_name = os.path.join(self.root, img_name)


        img = io.imread(img_name)
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0

        lab_img_name = 'gazemap_frames/' + self.labels[index]
        lab_img_name = os.path.join(self.label_root, lab_img_name)

        lab_img = cv2.imread(lab_img_name, 0)
        lab_img = cv2.resize(lab_img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        lab_img = lab_img.astype('float32') / 255.0

        if np.max(lab_img) < 0.1:
            print(lab_img_name, np.max(lab_img))

        if self.for_train:
            img, lab_img = transform(img, lab_img)

        img = img.transpose(2, 0, 1)
        lab_img = lab_img[None, ...]
        img = np.ascontiguousarray(img)
        lab_img = np.ascontiguousarray(lab_img)

        img_tensor, lab_img_tensor = torch.from_numpy(img), torch.from_numpy(lab_img)
        del img, lab_img
        return img_tensor, lab_img_tensor

    def __len__(self):
        return len(self.imgs)


class ImageList_BDDA_Continuous(Dataset):
    def __init__(self, args, imgs, for_train=False):
        self.root = args.root
        self.imgs = imgs
        self.for_train = for_train
        self.label_root = args.root
        self.labels = imgs
        self.seq_len = args.seq_len
        self.img_shape = args.img_shape

    def __getitem__(self, index):
        img_name = self.imgs[index]
        vid_index, frame_index = img_name.split('/')

        vid_index = int(vid_index)
        frame_index = int(frame_index[:-4])

        imgarr = []
        for m in range(self.seq_len):
            temp_frame_index = frame_index - m

            if temp_frame_index < 0:
                continue  # 跳过无效帧索引

            img_name = f'camera_frames/{vid_index:04d}/{temp_frame_index:04d}.jpg'
            image_name = os.path.join(self.root, img_name)

            if not os.path.exists(image_name):
                raise FileNotFoundError(f"Image not found: {image_name}")

            img = io.imread(image_name)
            img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
            imgarr.append(torch.from_numpy(img))

        img_name = f'gazemap_frames/{vid_index:04d}/{frame_index:04d}.jpg'
        label_name = os.path.join(self.label_root, img_name)
        label = cv2.imread(label_name, 0)

        if label is None:
            raise FileNotFoundError(f"Label not found at {label_name}")

        label = cv2.resize(label, self.img_shape, interpolation=cv2.INTER_CUBIC)
        label = np.ascontiguousarray(label)
        label = torch.from_numpy(label).unsqueeze(0)

        imgarr = torch.stack(imgarr)
        imgarr = imgarr.float() / 255.0
        label_arr = label.float() / 255.0

        return imgarr, label_arr

    def __len__(self):
        return len(self.imgs)


def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y