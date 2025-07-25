import argparse
import torch
from models.SalMM import SalMM
from clip import *


device = torch.device('cuda:0')
backbone = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
id = 4

'''
[official note] The original text contains an error. 
[official note] We wrongly wrote the model with id=4 as 'RN50x16'.
[official note] In fact, the backbone used by CLIP is 'ViT-B/32'.
[official note] reference: https://arxiv.org/pdf/2502.16214
'''


def Restore_model(
    model: torch.nn.Module,
    path: str,
    device='cpu',
    backbone=backbone[id],
    save_path=None
):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict']

    # load model
    model.load_state_dict(state_dict, strict=False)

    # restore complete model
    clip_model, _ = clip.load(backbone)
    model.CrossModelAtt.model = clip_model.eval()

    if save_path:
        full_state_dict = model.state_dict()
        checkpoint['state_dict'] = full_state_dict
        torch.save(checkpoint, save_path)
        print(f"Successfully Restored. Saving on: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restore SalMM model by reloading CLIP weights.")
    parser.add_argument('--inc_path', type=str, required=True, help="Path to incomplete model (.tar) without CLIP")
    parser.add_argument('--c_path', type=str, required=True, help="Path to save complete model (.tar)")

    args = parser.parse_args()

    # 初始化模型
    model = SalMM().to(device)

    # 恢复模型并保存
    Restore_model(
        model=model,
        path=args.inc_path,
        save_path=args.c_path
    )





