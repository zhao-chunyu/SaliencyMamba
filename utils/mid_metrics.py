import torch
import numpy as np

eps = np.finfo(np.float32).eps

def kldiv(s_map, gt):
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_gt.size() == gt.size()

    # eps = 2.2204e-16

    # added eps to prevent division from zero
    s_map_temp = s_map / ((expand_s_map + eps) * 1.0)
    gt_temp = gt / ((expand_gt + eps) * 1.0)

    s_map_temp = s_map_temp.view(batch_size, -1)
    gt_temp = gt_temp.view(batch_size, -1)

    result = gt_temp * torch.log(eps + gt_temp / (s_map_temp + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    # return torch.mean(torch.sum(result, 1))
    return torch.mean(torch.sum(result, 1))


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map


def sim(s_map, gt):
    ''' For single image metric
        Size of Image - WxH or 1xWxH
        gt is ground truth saliency map
    '''
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    s_map_norm = normalize_map(s_map)
    gt_norm = normalize_map(gt)

    sum_s_map = torch.sum(s_map_norm.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map_norm.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map_norm = s_map_norm / (expand_s_map * 1.0)
    gt_norm = gt / (expand_gt * 1.0)

    s_map_norm = s_map_norm.view(batch_size, -1)
    gt_norm = gt_norm.view(batch_size, -1)
    # return torch.mean(torch.sum(torch.min(s_map, gt), 1))
    return torch.mean(torch.sum(torch.min(s_map_norm, gt_norm), 1))


def cc(s_map, gt):
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map_norm = (s_map - mean_s_map) / std_s_map
    gt_norm = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map_norm * gt_norm).view(batch_size, -1), 1)
    aa = torch.sum((s_map_norm * s_map_norm).view(batch_size, -1), 1)
    bb = torch.sum((gt_norm * gt_norm).view(batch_size, -1), 1)

    # return torch.mean(ab / (torch.sqrt(aa*bb)))
    return torch.mean(ab / (torch.sqrt(aa * bb)))
