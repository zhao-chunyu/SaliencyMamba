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
torch.cuda.set_device(int(args.gpu))
  

def write_to_file(log_path, message):
    with open(log_path, 'a') as f:
        f.write(message + '\n')


def predict_metrics(test_loader, model, ckpts):
    # init logging
    log_path = os.path.join(ckpts, 'test_log_metrics.txt')

    # switch to evaluate mode
    model.eval()

    cc_ = np.empty((0,))
    sim = np.empty((0,))
    kld = np.empty((0,))

    # total_samples = sum(input.shape[0] for input, _ in test_loader)
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze(1)
        input = input.cuda()

        with torch.no_grad():
            input_var = input
            output, _ = model(input_var)
            output = output.squeeze(1)

            batch = output.shape[0]  # real batch 
            for j in range(batch):
                index = i * test_loader.batch_size + j + 1  # global batch

            # for j in range(batch):
                # index = (batch*i)+(j+1)
                tem_out = output[j]
                tem_tar = target[j]
                # print(tem_out.shape)

                tar = tem_tar.detach().cpu().numpy()
                pre = tem_out.detach().cpu().numpy()

                temp_cc = cc(pre, tar)
                temp_sim = similarity(pre, tar)
                temp_kld =kldiv(pre, tar)

                cc_ = np.append(cc_, temp_cc)
                sim = np.append(sim, temp_sim)
                kld = np.append(kld, temp_kld)
                
                # print(i,j,index)
                write_to_file(log_path, f'Iter {index:5d}/{len(test_loader.dataset):5d}, '
                                        f'cc={temp_cc:.4f}, '
                                        f'sim={temp_sim:.4f}, '
                                        f'kld={temp_kld:.4f}, '
                                        f'a_cc={cc_.mean():.4f}, '
                                        f'a_sim={sim.mean():.4f}, '
                                        f'a_kld={kld.mean():.4f}.')

                if index % 200 == 0:
                    print(f'Iter {index:5d}/{len(test_loader.dataset):5d}, '
                        f'cc={cc_.mean():.4f}, '
                        f'sim={sim.mean():.4f}, '
                        f'kld={kld.mean():.4f}.')

    write_to_file(log_path, '*' * 50)
    write_to_file(log_path, f'Finally, cc={cc_.mean():.4f}, sim={sim.mean():.4f}, kld={kld.mean():.4f}')
    write_to_file(log_path, '*' * 50)

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
    # test------>next
    _, _, test_loader = build_dataset(args=args)
    ckpts = f'ckpts/{args.category}/{args.test_weight}/'
    file_name = ckpts + f'model_best_{args.network}2.tar'
    model = build_model(args=args)
    model = model.cuda()

    checkpoint = torch.load(file_name)
    # model.load_state_dict(checkpoint['state_dict'])  # single-GPU
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})  # multi-GPU

    if any(key.startswith('module.') for key in checkpoint['state_dict'].keys()):
        # multi-GPU: move 'module.'
        print("[Start Testing] Detected multi-GPU training. Loading")
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    else:
        # single-GPU：
        print("[Start Testing] Detected single-GPU training. Loading")
        model.load_state_dict(checkpoint['state_dict'])

    predict_metrics(test_loader, model, ckpts)
