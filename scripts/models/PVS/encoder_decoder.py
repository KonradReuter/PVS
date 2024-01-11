import torch
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from scripts.models.UNet.modules import Up
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNextEncoder(torch.nn.Module):
    def __init__(self, backbone: any = convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT), stages: int = 4) -> None:
        """Initialization
        """
        super(ConvNextEncoder, self).__init__()
        assert 1 <= stages <= 4, "Value for number of stages out of bounds. Make sure it is between 1 and 4"
        self.stages = stages
        self.feature_extractor = backbone.features[:stages*2]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Generate encoded states.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            tuple[torch.Tensor]: Encoded states after the first, second and third encoding layer
        """
        # incoming image: b x f x c x h x w
        in_shape = x.shape
        out = []
        # flatten batch and frame layer
        x = x.view(-1, *x.shape[2:])
        for i in range(self.stages):
            x = self.feature_extractor[(i*2):(i+1)*2](x)
            features = x.view(*in_shape[:2], *x.shape[1:])
            out.append(features)
        return out


class SwinEncoder(torch.nn.Module):
    def __init__(self, backbone: any = swin_v2_t(weights = Swin_V2_T_Weights.DEFAULT), stages: int = 4) -> None:
        """Initialization
        """
        super(SwinEncoder, self).__init__()
        assert 1 <= stages <= 4, "Value for number of stages out of bounds. Make sure it is between 1 and 4"
        self.stages = stages
        self.feature_extractor = backbone.features[:stages*2]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Generate encoded states.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            tuple[torch.Tensor]: Encoded states after the first, second and third encoding layer
        """
        # incoming image: b x f x c x h x w
        in_shape = x.shape
        out = []
        # flatten batch and frame layer
        x = x.view(-1, *x.shape[2:])
        for i in range(self.stages):
            x = self.feature_extractor[(i*2):(i+1)*2](x)
            features = x.view(*in_shape[:2], *x.shape[1:])
            features = features.permute(0, 1, 4, 2, 3).contiguous()
            out.append(features)
        return out

class Decoder3(torch.nn.Module):
    def __init__(self) -> None:
        super(Decoder3, self).__init__()
        self.up1 = Up(384, 192, bilinear=False)
        self.up2 = Up(192, 96, bilinear=False)
        self.up3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU()
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3 = x
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x)
        return x


class Decoder4(torch.nn.Module):
    """UNet inspired decoder with skip connections
    """
    def __init__(self) -> None:
        """Initialization
        """
        super(Decoder4, self).__init__()
        self.up1 = Up(768, 384, bilinear=False)
        self.up2 = Up(384, 192, bilinear=False)
        self.up3 = Up(192, 96, bilinear=False)
        self.up4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU()
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4 = x
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x)
        return x


class PatchExpansion(torch.nn.Module):
    """Module which reverses patch merging
    """
    def __init__(self, dim: int, norm_layer: any = torch.nn.LayerNorm) -> None:
        """Initialization

        Args:
            dim (int): Number of input channels
            norm_layer (any, optional): Normalization to be applied. Defaults to torch.nn.LayerNorm.
        """
        super(PatchExpansion, self).__init__()
        self.dim = dim
        self.expansion = torch.nn.Linear(dim // 4, dim //2, bias = False)
        self.norm = norm_layer(dim //2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies patch expansion and doubles channels afterwards

        Args:
            x (torch.Tensor): Input tensor of form b x h x w x c

        Returns:
            torch.Tensor: Output tensor of form bx 2h x 2w x c/2
        """
        x = _patch_expansion(x)
        x = self.expansion(x)
        x = self.norm(x)
        return(x)

def _patch_expansion(x: torch.Tensor) -> torch.Tensor:
    """Applies patch expansion

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Output tensor
    """
    H, W, C = x.shape[-3:]
    # pad input if necessary
    x = torch.nn.functional.pad(x, (0, (4-C%4) if C%4 != 0 else 0, 0, W % 2, 0, H % 2))
    # split along channel dimension
    x1, x2, x3, x4 = x.chunk(4, dim = -1)
    # insert dimension for concatenation in height dimension
    x1 = x1.unsqueeze(-3)
    x2 = x2.unsqueeze(-3)
    x3 = x3.unsqueeze(-3)
    x4 = x4.unsqueeze(-3)
    # concat and flatten additional dimension
    x12 = torch.concat((x1, x2), -3).reshape(x1.shape[0], -1, *x1.shape[3:5])
    x34 = torch.concat((x3, x4), -3).reshape(x3.shape[0], -1, *x3.shape[3:5])
    # insert another dimension for concatenation in width dimension
    x12 = torch.unsqueeze(x12, -2)
    x34 = torch.unsqueeze(x34, -2)
    # concat and flatten additional dimension
    x1234 = torch.concat((x12, x34), -2).reshape(*x12.shape[0:2], -1, x12.shape[-1])
    return x1234

class PatchExpansionDecoder(torch.nn.Module):
    """Simple upsampling decoder using patch expansion
    """
    def __init__(self) -> None:
        """Initialization
        """
        super(PatchExpansionDecoder, self).__init__()
        self.up1 = PatchExpansion(768)
        self.up2 = PatchExpansion(384)
        self.up3 = PatchExpansion(192)
        self.up4 = PatchExpansion(96)
        self.up5 = PatchExpansion(48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies upsampling

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Upsampled output tensor
        """
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x
