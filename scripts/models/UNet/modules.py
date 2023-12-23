import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DoubleConv3D(nn.Module):
    """Applies 3D convolutions in order to fuse temporal information"""

    def __init__(self, channels_in: int, channels_out: int = None, channels_mid: int = None, reduce_frame_dim: bool = False) -> None:
        """Initialization

        Args:
            channels_in (int): Number of channels in the input tensor.
            channels_out (int, optional): Number of output channels. Will equal input channels if not provided. Defaults to None.
            reduce_frame_dim (bool, optional): If true, the frame dimension will not be padded. Defaults to False.
        """
        super().__init__()
        if channels_out == None:
            channels_out = channels_in
        if channels_mid == None:
            channels_mid = channels_in
        padding = (0, 1, 1) if reduce_frame_dim else (1, 1, 1)

        self.double_conv_3d = nn.Sequential(nn.Conv3d(channels_in, channels_mid, kernel_size=3, padding = padding, bias=False),
                                            nn.BatchNorm3d(channels_mid),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(channels_mid, channels_out, kernel_size=3, padding = padding, bias=False),
                                            nn.BatchNorm3d(channels_out),
                                            nn.ReLU(inplace=True)
                                            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies double 3D convolution

        Args:
            x (torch.Tensor): Input. Expected to have shape BxCxFxHxW

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.double_conv_3d(x)
        return out.squeeze(2)

class ConcatAndFuse(nn.Module):
    """Concatenates input along channel dimension and applies double convolution."""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None) -> None:
        """Initialization

        Args:
            in_channels (int): Channels of the input tensor.
            out_channels (int): Channels of the output tensor.
            mid_channels (int, optional): Mid channels for the double convolution. Defaults to None.
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate channels then double convolution. Reshapes the output back to input shape afterwards.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        shape = x.shape
        x = x.view(shape[0], -1, *shape[3:])
        x = self.conv(x)
        x = x.view(*shape)
        return x

if __name__ == '__main__':
    sample_input = torch.randn(2, 5, 32, 256, 256)
    model = DoubleConv3D(32)
    out = model(sample_input.permute(0, 2, 1, 3, 4))
    print(out.shape)