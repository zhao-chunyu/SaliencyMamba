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

from utils.options import parser
from utils.bulid_models import build_model
from utils.build_datasets import build_dataset

warnings.simplefilter("ignore")
os.environ['NUMEXPR_MAX_THREADS'] = '64'

args = parser.parse_args()


def predict_metrics(test_loader, model):
    # switch to evaluate mode
    model.eval()

    cc_ = np.empty((0,))
    sim = np.empty((0,))
    kld = np.empty((0,))

    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze(0).squeeze(0)
        input = input.cuda()

        output, _ = model(input)

        output = output.squeeze(0).squeeze(0)

        tar = target.detach().cpu().numpy()
        pre = output.detach().cpu().numpy()

        temp_cc = cc(pre, tar)
        temp_sim = similarity(pre, tar)
        temp_kld =kldiv(pre, tar)

        cc_ = np.append(cc_, temp_cc)
        sim = np.append(sim, temp_sim)
        kld = np.append(kld, temp_kld)

        if (i + 1) % 200 == 0:
            print(f'Iter {i+1:5d}/{len(test_loader):5d}, '
                  f'cc={cc_.mean():.4f}, '
                  f'sim={sim.mean():.4f}, '
                  f'kld={kld.mean():.4f}.')
    print('*' * 50)
    print(f'Finally, cc={cc_.mean():.4f}, sim={sim.mean():.4f}, kld={kld.mean():.4f}')
    print('*' * 50)


def similarity(s_map, gt):
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    return np.sum(np.minimum(s_map, gt))


def cc(s_map, gt):
    a = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt))/(np.std(gt) + 1e-7)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum() + 1e-7)
    return r


def kldiv(s_map, gt):
    s_map = s_map / (np.sum(s_map) * 1.0)
    gt = gt / (np.sum(gt) * 1.0)
    eps = 2.2204e-16
    res = np.sum(gt * np.log(eps + gt / (s_map + eps)))
    return res


if __name__ == '__main__':
    # #  test------>next
    _, _, test_loader = build_dataset(args=args)
    ckpts = f'ckpts/{args.test_weight}/'
    file_name = ckpts + f'model_best_{args.network}.tar'
    model = build_model(args=args)
    model = model.cuda()

    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['state_dict'])  # 单卡训练加载方式
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})  # 多卡训练加载方式
    predict_metrics(test_loader, model)
