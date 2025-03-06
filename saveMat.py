import os
import time
import torch

import random
import warnings
import logging
import numpy as np
import json
from datetime import datetime

from utils.options import parser
from utils.bulid_models import build_model
from utils.build_datasets import build_dataset
from tqdm import tqdm

warnings.simplefilter("ignore")
os.environ['NUMEXPR_MAX_THREADS'] = '64'

args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))

def predict_mat(test_loader, model):
    import scipy.io as sio

    test_imgs = [json.loads(line) for line in open(args.root + '/test.json')]

    # switch to evaluate mode
    model.eval()

    save_path = ckpts + 'outputMat/'
    
    total_samples = len(test_imgs)

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="matFile Saving", ncols=90)

    for i, (input, target) in progress_bar:
        target = target.squeeze(1)
        input = input.cuda()

        # compute output
        output, _ = model(input)
        output = output.squeeze(1)

        target = target.detach().cpu().numpy()
        output = output.detach().cpu().numpy()

        batch = output.shape[0]

        for j in range(batch):
            index = i * test_loader.batch_size + j
            save_name = test_imgs[index].replace('jpg', 'mat')

            tar = target[j]
            pre = output[j]

            tar_mat_dict = {'tar': tar}
            pre_mat_dict = {'pre': pre}

            # create directories if not exist
            # print(save_name)
            path = save_path + 'tar/' + save_name
            path2 = path.replace('tar', 'pre')

            path_ = os.path.dirname(path)
            path2_ = os.path.dirname(path2)

            # print(path, path2)
            # exit(0)
            os.makedirs(path_, exist_ok=True)
            os.makedirs(path2_, exist_ok=True)

            sio.savemat(path, tar_mat_dict)
            sio.savemat(path2, pre_mat_dict)
        
            # update
            # progress_bar.set_postfix(Iter="{:03d}/{:03d}".format(i + 1, len(valid_loader)))

if __name__ == '__main__':
    # save------>next
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

    predict_mat(test_loader, model)
