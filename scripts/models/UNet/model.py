import torch
import torch.nn as nn
import torch.nn.functional as F
#from config.config import logger, device
from scripts.models.PolypSwin.convlstm import ConvLSTM, BidirectionalConvLSTM
from scripts.models.UNet.modules import *
#from scripts.PNSPlusModule import NS_Block

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters = [64, 128, 256, 512, 1024], bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor)
        self.up1 = Up(filters[4], filters[3] // factor, bilinear)
        self.up2 = Up(filters[3], filters[2] // factor, bilinear)
        self.up3 = Up(filters[2], filters[1] // factor, bilinear)
        self.up4 = Up(filters[1], filters[0], bilinear)
        self.output_layer = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output_layer(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.output_layer = torch.utils.checkpoint(self.output_layer)

if __name__ == '__main__':
    from scripts.utils import SingleImageModelWrapper
    sample_input = torch.randn(2, 5, 3, 256, 256)
    model = SingleImageModelWrapper(UNet(3, 1))
    pred = model(sample_input)
    print(pred.shape)
