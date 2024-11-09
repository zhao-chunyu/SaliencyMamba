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

torch.cuda.set_device(int(args.gpu))


date_time = datetime.now().strftime('%Y%m%d-%H:%M:%S')

ckpts = f'ckpts/{args.network}_{date_time}/'
if not os.path.exists(ckpts):
    os.makedirs(ckpts)

log_file = os.path.join(ckpts + "/train_log_%s.txt" % (args.network, ))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)

# saving config list
logging.info('*'*50)
logging.info(f'[Train Model] ----> {ckpts[6:-1]}')
for arg, value in vars(args).items():
    logging.info(f'%-15s: %s', arg, value)
logging.info('*'*50)


def main():
    # global args, best_score
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = args.gpu

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args=args)
    model = model.cuda()

    params = model.parameters()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # args.start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, valid_loader, _ = build_dataset(args=args)

    criterion = nn.BCELoss().cuda()

    best_loss = float('inf')
    file_name = os.path.join(ckpts, 'model_best_%s.tar' % (args.network, ))
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # train for one epoch
        train_loss = train(
                train_loader, model, criterion, optimizer, epoch)

        valid_loss = validate(
                valid_loader, model, criterion)

        # remember best lost and save checkpoint
        best_loss = min(valid_loss, best_loss)
        file_name_last = os.path.join(ckpts, 'model_epoch_%d.tar' % (epoch + 1, ))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'valid_loss': valid_loss,
        }, file_name_last)

        if valid_loss == best_loss:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, file_name)

        msg = 'Epoch: {:02d} Train loss {:.4f} | Valid loss {:.4f}'.format(
                epoch+1, train_loss, valid_loss)
        logging.info(msg)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    train_len = len(train_loader)
    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output, _ = model(input)
        loss = criterion(output, target)
        losses.update(loss, target.size(0))

        optimizer.zero_grad()
        # print(loss)
        loss.backward()

        optimizer.step()

        del input, target, output

        interval = 10
        if (i+1) % interval == 0:
            msg = 'Training Epoch {:03d}  Iter/Total = {:04d}/{:04d}  Loss {:.6f} in {:.3f}s'.format(epoch+1, i+1, train_len, losses.avg, time.time() - start)
            start = time.time()
            logging.info(msg)

    return losses.avg


def validate(valid_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()
    metrics = [0, 0, 0]
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output, _ = model(input)

            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.data, target.size(0))

            # valid metrics printing
            output = output.squeeze(1)
            target = target.squeeze(1)
            metrics[0] = metrics[0] + cc(output, target)
            metrics[1] = metrics[1] + sim(output, target)
            metrics[2] = metrics[2] + kldiv(output, target)

            msg = 'Validating Iter {:03d} Loss {:.6f} || CC {:4f}  SIM {:4f}  KLD {:4f} in {:.3f}s'.format(i + 1,
                                                                                                           losses.avg,
                                                                                                           metrics[0] / (i + 1),
                                                                                                           metrics[1] / (i + 1),
                                                                                                           metrics[2] / (i + 1),
                                                                                                           time.time() - start)
            start = time.time()
               # logging.info(msg)

            del input, target, output
            # gc.collect()

            interval = 10
            if (i + 1) % interval == 0:
                logging.info(msg)

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs//3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # train
    main()

    # evaluate
    # from evaluate import predict_mat
    # file_name = f'{ckpts}model_best_{args.network}.tar'
    # _, _, test_loader = build_dataset(args=args)
    #
    # model = build_model(args=args)
    # model = model.cuda()
    #
    # checkpoint = torch.load(file_name)
    # model.load_state_dict(checkpoint['state_dict'])
    #
    # predict_mat(test_loader, model, args)
