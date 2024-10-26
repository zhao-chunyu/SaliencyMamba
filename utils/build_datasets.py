"""
Author: xxxxxxxx
Date: 2024-10-10
Description:
    - This module performs dataset-building for TrafficGze, DrFixD_rainy and BDDA.
    - You can add new dataset-building.
References:
    - Some relevant reference or source
    - paper:
"""
import json
from torch.utils.data import DataLoader
from .datasets.TrafficGaze import *
from .datasets.DrFixD_rainy import *
from .datasets.BDDA import *


def build_dataset(args=None):

    if args.seq_len > 1:
        dataset_classes = {
            'TrafficGaze': ImageList_TrafficGaze_Continuous,
            'DrFixD_rainy': ImageList_DrFixD_rainy_Continuous,
            'BDDA': ImageList_BDDA_Continuous
        }
    else:
        dataset_classes = {
            'TrafficGaze': ImageList_TrafficGaze,
            'DrFixD_rainy': ImageList_DrFixD_rainy,
            'BDDA': ImageList_BDDA
        }

    train_imgs = [json.loads(line) for line in open(args.root + '/train.json')]
    valid_imgs = [json.loads(line) for line in open(args.root + '/valid.json')]
    test_imgs = [json.loads(line) for line in open(args.root + '/test.json')]

    dataset_class = dataset_classes.get(args.category)

    if dataset_class is None:
        raise ValueError(f"Unknown category: {args.category}")

    train_loader = DataLoader(
        dataset_class(args, train_imgs, for_train=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    valid_loader = DataLoader(
        dataset_class(args, valid_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = DataLoader(
        dataset_class(args, test_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('SalMAE training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--seq_len', default=2, type=int)
    parser.add_argument('--img_shape', default=(224, 224), type=lambda s: tuple(map(int, s.split(','))))
    parser.add_argument('--category', default='BDDA', type=str)
    parser.add_argument('--root', default='F:/Build_dataset/BDDA', type=str)

    args = parser.parse_args()

    train_loader, valid_loader, test_loader = build_dataset(args=args)
    for i, (input, target) in enumerate(train_loader):
        print('input: ', input.shape)
        print('target: ', target.shape)
        exit(0)
