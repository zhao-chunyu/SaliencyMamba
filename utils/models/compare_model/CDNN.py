import os
import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class CDNN(nn.Module):
    def __init__(self):
        n, m = 24, 3

        super(CDNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.convd1 = conv3x3(1 * m, 1 * n)
        self.convd2 = conv3x3(1 * n, 2 * n)
        self.convd3 = conv3x3(2 * n, 4 * n)
        self.convd4 = conv3x3(4 * n, 4 * n)

        self.convu3 = conv3x3(8 * n, 4 * n)
        self.convu2 = conv3x3(6 * n, 2 * n)
        self.convu1 = conv3x3(3 * n, 1 * n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        x1 = x
        # print("1st branch in:",x1.size())
        x1 = self.convd1(x1)
        # print(x1.size())

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)
        # print(x2.size())

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)
        # print(x3.size())

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)
        # print(x4.size())

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3)
        # print(y3.size())

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)
        # print(y2.size())

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)
        # print(y1.size())

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)
        # print(y1.size())
        # exit(0)
        return y1, None


if __name__ == '__main__':
    import time
    b = 1
    img = torch.randn(size=(b, 3, 256, 256)).cuda()
    model = CDNN().cuda()
    output, _ = model(img)
    print(output.size())
    para = sum([param.nelement() for param in model.parameters()])
    # print('paramaters = ', (para-clip_para)/1000000, 'M')

    # model = CDNN().cuda()
    # 测量运行时间
    for i in range(10):
        b = 2 ** i
        print(i)
        img = torch.randn(size=(b, 3, 256, 256)).cuda()
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            _ = model(img)

        torch.cuda.synchronize()
        end_time = time.time()

        print(f'运行时间: {(end_time - start_time) * 1000:.2f} ms')

