import torch
import torch.nn as nn
import torch.nn.functional as F
from img_utils import *


class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channels=64, kernel_size=3, dropout=0.):
        super(Model, self).__init__()
        padding = int(kernel_size/2)
        ops = [
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=channels,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(channels),
            nn.Conv2d(in_channels=channels, out_channels=channels*2,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),

            nn.BatchNorm2d(channels*2),
            nn.Conv2d(in_channels=channels*2, out_channels=channels*4,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(channels*4),
            nn.Conv2d(in_channels=channels*4, out_channels=channels*8,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(channels*8),
            nn.Conv2d(in_channels=channels*8, out_channels=channels*4,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),


            nn.BatchNorm2d(channels*4),
            nn.Conv2d(in_channels=channels*4, out_channels=channels*2,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.BatchNorm2d(channels*2),
            nn.Conv2d(in_channels=channels*2, out_channels=channels,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),


            nn.Upsample([res[1],res[0]]),
            nn.Conv2d(in_channels=channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.Sigmoid(), ]
        self.ops = nn.ModuleList(ops)

    def forward(self, x):
        # print(x.shape)
        for op in self.ops:
            x = op(x)
            # print(x.shape)
        return x


print('Using Naive FCN')

if __name__ == "__main__":
    model = Model()
    model = nn.DataParallel(model).cuda()
    print(model.eval())
    sig = torch.zeros([1, 3, res[1], res[0]]).cuda()
    ret = model(sig)
    print(ret.shape)
