import os
import time
import torch

import warnings
import numpy as np
import json

from utils.options import parser
from utils.bulid_models import build_model
from utils.build_datasets import build_dataset

from torch.utils.data import DataLoader
import imageio as io
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.simplefilter("ignore")
os.environ['NUMEXPR_MAX_THREADS'] = '64'

args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))



def img2tensor(img_path, rgb=True):
    if rgb:
        img = io.imread(img_path)
        # print(img.shape)
        img = cv2.resize(img, args.img_shape, interpolation=cv2.INTER_CUBIC)
        # print(img.shape)
        img_ = img.astype('float32')/255.0
        img_ = img_.transpose(2, 0, 1)
        # print(img_.shape)
        # exit(0)
        img_ = np.ascontiguousarray(img_)
        img_tesor = torch.from_numpy(img_)

    else:
        img = io.imread(img_path, 0)
        img = cv2.resize(img, args.img_shape, interpolation=cv2.INTER_CUBIC)
        img_ = img.astype('float32')/255.0
        img_ = np.ascontiguousarray(img_)
        img_tesor = torch.from_numpy(img_)
    return img, img_tesor


def savePlot(img_ori=None, img_sal=None, saveFullName=None):
    if img_ori is None or img_sal is None:
        img = img_ori if img_ori is not None else img_sal
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        plt.savefig(saveFullName, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.imshow(img_ori)
        plt.imshow(img_sal, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.show()
        plt.savefig(saveFullName, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def onevisual(model, input_data, saveFolder, save_ori=True, save_pre=True):
    """
    type of input_data: str, tensor
    example:
            str : '0001/0001.jpg'
            list: [index, nump-->in_img, nump-->tar_img, nump-->out_img]
    """
    if isinstance(input_data, str):
        saveBaseName = os.path.basename(input_data)
        saveFolder = f'{saveFolder}/temp/'
        os.makedirs(saveFolder, exist_ok=True)

        img, img_tesor = img2tensor(input_data)
        img_tesor = img_tesor.cuda()
        img_tesor = img_tesor.unsqueeze(0)

        
        pre_mask, _ = model(img_tesor)

        pre_mask = pre_mask.detach().cpu().numpy()
        if len(pre_mask.shape) == 4:
            pre_mask = pre_mask.squeeze(0).squeeze(0)

        # save ori
        if save_ori:
            saveFullName = f'{saveFolder}/ori_{saveBaseName}'
            savePlot(img_ori=img, saveFullName=saveFullName)
        # save pre
        if save_pre:
            saveFullName = f'{saveFolder}/sal_{saveBaseName}'
            savePlot(img_sal=pre_mask, saveFullName=saveFullName)
        # save ori+pre
        saveFullName = f'{saveFolder}/add_{saveBaseName}'
        savePlot(img_ori=img, img_sal=pre_mask, saveFullName=saveFullName)

    else:
        # save_ori = False
        # save_pre = False
        saveBaseNames = [json.loads(line) for line in open(args.root + '/test.json')]
        index, in_img, tar_img, out_img = input_data[0], input_data[1], input_data[2], input_data[3]
        saveBaseName = saveBaseNames[index]
        # print(saveBaseName)
        saveBaseNameList = saveBaseName.split('/')
        # print(saveBaseNameList)
        in_img = in_img.squeeze(0).transpose(1, 2, 0)

        # save ori
        if save_ori:
            saveFolder_ = f'{saveFolder}/{args.category}/ori/{saveBaseNameList[0]}'
            os.makedirs(saveFolder_, exist_ok=True)
            saveFullName = f'{saveFolder_}/{saveBaseNameList[1]}'
            savePlot(img_ori=in_img, saveFullName=saveFullName)
        # save pre
        if save_pre:
            saveFolder_ = f'{saveFolder}/{args.category}/pre/{saveBaseNameList[0]}'
            os.makedirs(saveFolder_, exist_ok=True)
            saveFullName = f'{saveFolder_}/{saveBaseNameList[1]}'
            savePlot(img_sal=out_img, saveFullName=saveFullName)
        # save ori+pre
        saveFolder_ = f'{saveFolder}/{args.category}/add/{saveBaseNameList[0]}'
        os.makedirs(saveFolder_, exist_ok=True)
        saveFullName = f'{saveFolder_}/{saveBaseNameList[1]}'
        savePlot(img_ori=in_img, img_sal=out_img, saveFullName=saveFullName)


def visualization(model, input_data, saveFolder='visualResults', save_ori=True, save_pre=True):
    """
    type of input_data: str, list, .txt, dataloader
    example:
            str : '0001/0001.jpg'
            list: ['0001/0001.jpg', '0001/0002.jpg', ......]
            dataloader: by our build_dataset.
    """
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise ValueError(f"Path: {input_data} does not exist.")

        onevisual(model, input_data=input_data, saveFolder=saveFolder, save_ori=True, save_pre=True)

    elif isinstance(input_data, list):
        for path in input_data:
            if not isinstance(path, str):
                raise TypeError(f"The element must be string, but {path} is {type(path)}")
            if not os.path.exists(path):
                raise ValueError(f"Path: {input_data} does not exist.")
        for path in input_data:
            onevisual(model, input_data=path, saveFolder=saveFolder, save_ori=True, save_pre=True)       

    elif isinstance(input_data, DataLoader):
        for i, (in_img, tar_img) in tqdm(enumerate(input_data), total=len(input_data), desc="Visualization Processing"):
            tar_img = tar_img.squeeze(0).squeeze(0)
            in_img = in_img.cuda()
            out_img, _ = model(in_img)
            out_img = out_img.squeeze(0).squeeze(0)

            in_img = in_img.detach().cpu().numpy()
            tar_img = tar_img.detach().cpu().numpy()
            out_img = out_img.detach().cpu().numpy()

            visual_np = [i, in_img, tar_img, out_img]
            onevisual(model, input_data=visual_np, saveFolder=saveFolder, save_ori=True, save_pre=True)
    else:
        raise TypeError("Parameter input_data must be a string path, a list of string paths, a txt file or a DataLoader.")


if __name__ == '__main__':
    # #  test------>next
    _, _, test_loader = build_dataset(args=args)
    ckpts = f'ckpts/{args.category}/{args.test_weight}/'
    file_name = ckpts + f'model_best_{args.network}.tar'

    checkpoint = torch.load(file_name, map_location="cpu")

    model = build_model(args=args).cuda()
    model.load_state_dict(checkpoint['state_dict'])  # single-gpu
    # print(model.device)
    # input_str = '/data/dataset/BDDA/images/0002/001.jpg'
    # input_list = ['/data/dataset/BDDA/images/0002/002.jpg', '/data/dataset/BDDA/images/0002/003.jpg']
    input_dataloader = test_loader
    saveFolder = 'visualResults'
    visualization(model, input_dataloader, saveFolder)
