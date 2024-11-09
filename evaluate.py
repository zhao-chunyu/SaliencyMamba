import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import random
import warnings
import logging
import numpy as np
import json
import gc
from datetime import datetime

from utils.mid_metrics import cc, sim, kldiv
from utils.options import parser
from utils.bulid_models import build_model
from utils.build_datasets import build_dataset

warnings.simplefilter("ignore")
os.environ['NUMEXPR_MAX_THREADS'] = '64'

args = parser.parse_args()

def predict_mat(valid_loader, model, args):
    import scipy.io as sio

    test_imgs = [json.loads(line) for line in open(args.root + '/test.json')]


    # switch to evaluate mode
    model.eval()

    save_path = ckpts+'/model_out/'
    for i, (input, target) in enumerate(valid_loader):
        save_name = test_imgs[i].replace('jpg', 'mat')

        target = target.squeeze(0).squeeze(0)
        input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output, _ = model(input_var)
        output = output.squeeze(0).squeeze(0)

        tar = target.detach().cpu().numpy()
        pre = output.detach().cpu().numpy()
        tar_mat_dict = {'tar': tar}
        pre_mat_dict = {'pre': pre}

        # path exsit
        path = (save_path + 'tar/' + save_name)[:-8]
        path2 = path.replace('tar', 'pre')

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path2):
            os.makedirs(path2)

        sio.savemat(save_path + 'tar/' + save_name, tar_mat_dict)
        sio.savemat(save_path + 'pre/' + save_name, pre_mat_dict)

        if (i+1) % 10 == 0:
            msg = 'Predicting ---> Iter/Len = {:03d}/{:03d}'.format(i + 1, len(valid_loader))
            print(msg)


def predict_img(valid_loader, model, root):
    import cv2

    test_imgs = [json.loads(line) for line in open(root + 'test.json')]


    # switch to evaluate mode
    model.eval()

    save_path = ckpts + 'model_out_img/'
    from torchvision.transforms import Resize
    torch_resize = Resize([256, 256])
    for i, (input, target) in enumerate(valid_loader):
        save_name = test_imgs[i]

        input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output = model(input_var)
        output = output.squeeze(0).squeeze(0)

        tar = torch_resize(target).squeeze(0).squeeze(0)
        tar = tar.detach().cpu().numpy()

        pre = output.detach().cpu().numpy()
        # path exsit
        path = (save_path + 'pre/' + save_name)[:-8]
        path2 = path.replace('pre', 'tar')

        pre_name = save_path + 'pre/' + save_name
        tar_name = save_path + 'tar/' + save_name

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(path2):
            os.makedirs(path2)

        # print('test1 ', np.max(tar), np.min(tar))
        tar = (tar * 255).astype('float32')
        # print('test2 ', np.max(tar), np.min(tar))

        # array_uint8 = (array * 255).astype(np.uint8)
        pre = (pre * 255).astype('float32')  # 将数据类型转换为 uint8
        resized_pre = pre
        # target_size = (256, 256)  # 宽 x 高
        # # target_size = (256, 256)  # 宽 x 高
        # resized_pre = cv2.resize(pre, target_size, interpolation=cv2.INTER_LINEAR)

        # 将数据归一化到0-255
        min_val = np.min(resized_pre)
        max_val = np.max(resized_pre)
        normalized_arr = (((resized_pre - min_val) / (max_val - min_val)) * 255).astype('float32')

        # print(type(normalized_arr))
        # print(np.max(normalized_arr), np.min(normalized_arr))
        # exit(0)

        cv2.imwrite(pre_name, normalized_arr)
        cv2.imwrite(tar_name, tar)
        # print(save_name2)

        if (i+1) % 10 == 0:
            msg = 'Predicting ---> Iter/Len = {:03d}/{:03d}'.format(i + 1, len(valid_loader))
            print(msg)


if __name__ == '__main__':
    # #  test------>next
    _, _, test_loader = build_dataset(args=args)
    ckpts = f'ckpts/{args.test_weight}/'
    file_name = ckpts + f'model_best_{args.network}.tar'
    model = build_model(args=args)
    model = model.cuda()

    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['state_dict'])
    predict_mat(test_loader, model, args)
