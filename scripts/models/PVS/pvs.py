import torch
from config.config import logger, args
from scripts.models.PVS.encoder_decoder import *
from scripts.models.PVS.fusion_modules import NoFusion, FusionSimple, Fusion3D, FusionLSTM, FusionAttention, FusionNSA
from scripts.models.PVS.convNext import ConvNextEncoder
from scripts.models.PVS.temporalFusionEncoder import *
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

class PolypSwin(torch.nn.Module):
    """PolypSwin base class"""
    def __init__(self, encoder, fusion_module, decoder, return_all_frames = True) -> None:
        """Initialization
        """
        super(PolypSwin, self).__init__()
        self.encoder = encoder
        self.fusion = fusion_module
        self.decoder = decoder
        self.out_layer = torch.nn.Conv2d(24, 1, 1)
        self.identity = torch.nn.Identity()
        self.return_all_frames = return_all_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swap second and third dimension if we are calculating attention maps as the inputs are in form b x c x f x h x w
        if args["save_attention_maps"]:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        # x shape: b x f x c x h x w
        origin_shape = x.shape

        # get encoded states list[x1, x2, x3, (...)]
        x = self.encoder(x)
        # shape: b x f x c x h x w

        # apply module for temporal information fusion (takes and returns list[x1, x2, x3, (...)])
        x = self.fusion(x)
        # x shape: b*f x c x h x w

        # apply decoder module (takes list[x1, x2, x3, (...)], returns tensor)
        x = self.decoder(x)
        # x shape: b*f x c x h x w

        # use output layer to reduce the number of channels
        x = self.out_layer(x)

        # reshape back to temporal form
        if self.return_all_frames:
            x = x.view(*origin_shape[:2], *x.shape[1:])
        else:
            x = x.view(origin_shape[0], 1, *x.shape[1:])
        # x shape: b x f x c x h x w

        # reverse change from the beginning if we are calculating attention
        if args["save_attention_maps"]:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            # identity layer because attention maps can only be calculated from a specific layer backwards. So we need an additional layer after the dimension swap.
            x = self.identity(x)
        return x

def ConvNext_base():
    return PolypSwin(ConvNextEncoder(stages=4), NoFusion(), Decoder4())

def ConvNext_base3():
    return PolypSwin(ConvNextEncoder(stages=3), NoFusion(), Decoder3())

# Experiment 1 #############################################################################################

def ConvNext_simple():
    return PolypSwin(ConvNextEncoder(stages=3), FusionSimple([384], 5), Decoder3())

def ConvNext_3D():
    return PolypSwin(ConvNextEncoder(stages=3), Fusion3D([384], n = 1), Decoder3())

def ConvNext_LSTM():
    return PolypSwin(ConvNextEncoder(stages=3), FusionLSTM([384]), Decoder3())

def ConvNext_Attention():
    return PolypSwin(ConvNextEncoder(stages=3), FusionAttention([384], [4]), Decoder3())

def ConvNext_NSA():
    return PolypSwin(ConvNextEncoder(stages=3), FusionNSA([384], 4, [16], [16]), Decoder3())

# Experiment 2 ############################################################################################

def ConvNext_simple_skip():
    return PolypSwin(ConvNextEncoder(stages=3), FusionSimple([96, 192, 384], n_frames = 5), Decoder3())

def ConvNext_3D_skip():
    return PolypSwin(ConvNextEncoder(stages=3), Fusion3D([96, 192, 384], n = 1), Decoder3())

def ConvNext_LSTM_skip():
    return PolypSwin(ConvNextEncoder(stages=3), FusionLSTM([96, 192, 384]), Decoder3())

def ConvNext_Attention_skip():
    return PolypSwin(ConvNextEncoder(stages=3), FusionAttention([96, 192, 384], n_heads=[4, 4, 4]), Decoder3())

def ConvNext_NSA_skip():
    return PolypSwin(ConvNextEncoder(stages=3), FusionNSA([96, 192, 384], n_heads=4, heights=[64, 32, 16], widths=[64, 32, 16]), Decoder3())

# Experiment 3 #############################################################################################

def ConvNext_simple_enc():
    return PolypSwin(ConvTempEncoderSimple(), NoFusion(), Decoder3())

def ConvNext_3D_enc():
    return PolypSwin(ConvTempEncoder3D(), NoFusion(), Decoder3())

def ConvNext_LSTM_enc():
    return PolypSwin(ConvTempEncoderLSTM(), NoFusion(), Decoder3())

def ConvNext_Attention_enc():
    return PolypSwin(ConvTempEncoderAttention(), NoFusion(), Decoder3())

def ConvNext_NSA_enc():
    return PolypSwin(ConvTempEncoderNSA(), NoFusion(), Decoder3())

############################################################################################################
# TESTS

def test_model():
    return PolypSwin(ConvNextEncoder(backbone=convnext_small(weights=ConvNeXt_Small_Weights), stages=3), FusionLSTM(filters=[384]), Decoder3())

def ConvNext_LSTM_single():
    return PolypSwin(ConvNextEncoder(stages = 3), FusionLSTM([384], bidirectional=False), Decoder3())

############################################################################################################
def PolypSwin_base():
    return PolypSwin(SwinEncoder(stages = 4), NoFusion(), Decoder4())

def PolypSwin_base3():
    return PolypSwin(SwinEncoder(stages = 3), NoFusion(), Decoder3())

def PolypSwin_simple():
    return PolypSwin(SwinEncoder(stages = 4), FusionSimple([768], 5), Decoder4())

def PolypSwin_simple_skip():
    return PolypSwin(SwinEncoder(stages = 4), FusionSimple([96, 192, 384, 768], 5), Decoder4())

def PolypSwin_3D():
    return PolypSwin(SwinEncoder(stages = 4), Fusion3D([768], n = 2), Decoder4())

def PolypSwin_3D_skip():
    return PolypSwin(SwinEncoder(stages = 4), Fusion3D([96, 192, 384, 768], n=2), Decoder4())

def PolypSwin_LSTM():
    return PolypSwin(SwinEncoder(stages = 4), FusionLSTM([768]), Decoder4())

def PolypSwin_LSTM_skip():
    return PolypSwin(SwinEncoder(stages = 4), FusionLSTM([96, 192, 384, 768], bidirectional = False), Decoder4(), return_all_frames=False)

def PolypSwin_Attention():
    return PolypSwin(SwinEncoder(stages = 4), FusionAttention([768], 4, spatial = True), Decoder4())

def PolypSwin_Attention_skip():
    return PolypSwin(SwinEncoder(stages = 3), FusionAttention([96, 192, 384], 4), Decoder3())

def PolypSwin_Attention_skip_4():
    return PolypSwin(SwinEncoder(stages = 4), FusionAttention([96, 192, 384, 768], 4, spatial=True), Decoder4())

def PolypSwin_NSA():
    return PolypSwin(SwinEncoder(stages = 4), FusionNSA([768], 4, [8], [8], original=False, n = 1), Decoder4())

def PolypSwin_NSA_skip():
    return PolypSwin(SwinEncoder(stages = 4), FusionNSA([96, 192, 384, 768], 4, [64, 32, 16, 8], [64, 32, 16, 8], original=False, n = 1), Decoder4())

def PolypSwin_NSA_original():
    return PolypSwin(SwinEncoder(stages = 4), FusionNSA([768], 4, [8], [8], original=True, n = 1), DecoderNSA())

def PolypSwin_NSA_original_skip():
    return PolypSwin(SwinEncoder(stages=4), FusionNSA([96, 192, 384, 768], 4, [64, 32, 16, 8], [64, 32, 16, 8], original=True), DecoderNSA_skip())

def PolypSwin_test():
    return PolypSwin(TemporalFusionEncoderAttention(), NoFusion(), Decoder4())

if __name__ == "__main__":
    model = ConvNext_NSA()
    input = torch.randn(1, 5, 3, 256, 256)
    print(model(input).shape)

