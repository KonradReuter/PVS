#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.models.SANet.res2net import Res2Net50, weight_init
from scripts.utils import SingleImageModelWrapper

class Model(nn.Module):
    def __init__(self, snapshot = None):
        super(Model, self).__init__()
        self.snapshot = snapshot
        self.bkbone  = Res2Net50()
        self.linear5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
        pred = self.predict(pred)

        pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)

        #if not self.training:
        #    for i, p in enumerate(pred):
        #        p[torch.where(p>0)] /= (p>0).float().mean()
        #        p[torch.where(p<0)] /= (p<0).float().mean()
        #        pred[i] = p

        return pred

    def initialize(self):
        if self.snapshot:
            self.load_state_dict(torch.load(self.snapshot))
        else:
            weight_init(self)

def get_SANet():
    return SingleImageModelWrapper(Model())

if __name__ == '__main__':
    from scripts.utils import SingleImageModelWrapper
    model = SingleImageModelWrapper(Model())
    input = torch.randn(1, 5, 3, 256, 256)
    out = model(input)
    print(out.shape)