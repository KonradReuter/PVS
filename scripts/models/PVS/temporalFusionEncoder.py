import torch
from torchvision.models import swin_v2_t, Swin_V2_T_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from scripts.models.PVS.fusion_modules import FusionSimple, Fusion3D, FusionLSTM, FusionAttention, FusionNSA

class ConvTempEncoder(torch.nn.Module):
    def __init__(self, backbone: any = convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT), stages: int = 4) -> None:
        super(ConvTempEncoder, self).__init__()
        assert 1 <= stages <= 4, f"Value for number of stages ({stages}) out of bounds. Make sure it is between 1 and 4"
        self.stages = stages
        self.feature_extractor = backbone.features[:stages*2]
        self.filters = [self.feature_extractor[1+2*i][-1].block[5].out_features for i in range(stages)]
        self.fusion_modules = None

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
            x = self.feature_extractor[(0+i*2):(i+1)*2](x)
            x = x.view(*in_shape[:2], *x.shape[1:])
            x = self.fusion_modules[i]([x])[0]
            x = x.view(*in_shape[:2], *x.shape[1:])
            out.append(x)
            x = x.view(-1, *x.shape[2:])
        return out

class ConvTempEncoderSimple(ConvTempEncoder):
    def __init__(self, stages = 3) -> None:
        super(ConvTempEncoderSimple, self).__init__(stages=stages)
        self.fusion_modules = torch.nn.ModuleList([FusionSimple([f], n_frames=5) for f in self.filters])

class ConvTempEncoder3D(ConvTempEncoder):
    def __init__(self, stages = 3, n = 1) -> None:
        super(ConvTempEncoder3D, self).__init__(stages=stages)
        self.fusion_modules = torch.nn.ModuleList([Fusion3D([f], n = n) for f in self.filters])

class ConvTempEncoderLSTM(ConvTempEncoder):
    def __init__(self, stages = 3) -> None:
        super(ConvTempEncoderLSTM, self).__init__(stages=stages)
        self.fusion_modules = torch.nn.ModuleList([FusionLSTM([f]) for f in self.filters])

class ConvTempEncoderAttention(ConvTempEncoder):
    def __init__(self, stages = 3) -> None:
        super(ConvTempEncoderAttention, self).__init__(stages=stages)
        self.fusion_modules = torch.nn.ModuleList([FusionAttention([f], n_heads=[4]) for f in self.filters])

class ConvTempEncoderNSA(ConvTempEncoder):
    def __init__(self, stages = 3) -> None:
        super(ConvTempEncoderNSA, self).__init__(stages=stages)
        sizes = [4*2**(4-s) for s in range(stages)]
        self.fusion_modules = torch.nn.ModuleList([FusionNSA([f], 4, [size], [size]) for f, size in zip(self.filters, sizes)])


class SwinTempEncoder(torch.nn.Module):
    def __init__(self, backbone: any = swin_v2_t(weights = Swin_V2_T_Weights.DEFAULT), stages: int = 4) -> None:
        super(SwinTempEncoder, self).__init__()
        assert 1 <= stages <= 4, f"Value for number of stages ({stages}) out of bounds. Make sure it is between 1 and 4"
        self.stages = stages
        self.feature_extractor = backbone.features[:stages*2]
        self.filters = [self.feature_extractor[1+2*i][-1].mlp[3].out_features for i in range(stages)]
        self.fusion_modules = torch.nn.ModuleList([FusionSimple([f], 5) for f in self.filters])

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
            x = self.feature_extractor[(0+i*2):(i+1)*2](x)
            x = x.view(*in_shape[:2], *x.shape[1:])
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            x = self.fusion_modules[i]([x])[0]
            x = x.view(*in_shape[:2], *x.shape[1:])
            out.append(x)
            x = x.permute(0, 1, 3, 4, 2)
            x = x.view(-1, *x.shape[2:])
        return out


class TemporalFusionEncoder3D(SwinTempEncoder):
    def __init__(self, backbone: any = swin_v2_t(weights = Swin_V2_T_Weights.DEFAULT), stages = 4) -> None:
        super(TemporalFusionEncoder3D, self).__init__(backbone=backbone, stages=stages)
        self.fusion_modules = torch.nn.ModuleList([Fusion3D([f]) for f in self.filters])

class TemporalFusionEncoderLSTM(SwinTempEncoder):
    def __init__(self, backbone: any = swin_v2_t(weights = Swin_V2_T_Weights.DEFAULT), stages = 4) -> None:
        super(TemporalFusionEncoderLSTM, self).__init__(backbone=backbone, stages=stages)
        self.fusion_modules = torch.nn.ModuleList([FusionLSTM([f], bidirectional=True) for f in self.filters])

class TemporalFusionEncoderAttention(SwinTempEncoder):
    def __init__(self, backbone: any = swin_v2_t(weights = Swin_V2_T_Weights.DEFAULT), stages = 4) -> None:
        super(TemporalFusionEncoderAttention, self).__init__(backbone=backbone, stages=stages)
        self.fusion_modules = torch.nn.ModuleList([FusionAttention([f], 4, spatial = False) for f in self.filters])

class TemporalFusionEncoderNSA(SwinTempEncoder):
    def __init__(self, backbone: any = swin_v2_t(weights =Swin_V2_T_Weights.DEFAULT), stages = 4) -> None:
        super(TemporalFusionEncoderNSA, self).__init__(backbone = backbone, stages = stages)
        sizes = [4*2**(4-s) for s in range(stages)]
        self.fusion_modules = torch.nn.ModuleList([FusionNSA([f], 4, [size], [size], original=True) for f, size in zip(self.filters, sizes)])

if __name__ == '__main__':
    input = torch.randn(1, 5, 3, 256, 256)
    model = ConvTempEncoderSimple()
    print(model(input)[0].shape)