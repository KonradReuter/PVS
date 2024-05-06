import torch
from config.config import args
from scripts.models.PVS.encoder_decoder import *
from scripts.models.PVS.fusion_modules import NoFusion, FusionSimple, Fusion3D, FusionLSTM, FusionAttention, FusionNSA
from scripts.models.PVS.temporalFusionEncoder import *
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

class PVS(torch.nn.Module):
    """PolypVideoSegmentation base class"""
    def __init__(self, encoder, fusion_module, decoder, return_all_frames = True) -> None:
        """Initialization
        """
        super(PVS, self).__init__()
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

# Base Models ##############################################################################################

def ConvNext_base():
    return PVS(ConvNextEncoder(stages=4), NoFusion(), Decoder4())

def ConvNext_base3():
    return PVS(ConvNextEncoder(stages=3), NoFusion(), Decoder3())

def PolypSwin_base():
    return PVS(SwinEncoder(stages = 4), NoFusion(), Decoder4())

def PolypSwin_base3():
    return PVS(SwinEncoder(stages = 3), NoFusion(), Decoder3())

# Experiment 1 #############################################################################################

def ConvNext_simple():
    return PVS(ConvNextEncoder(stages=3), FusionSimple([384], 5), Decoder3())

def ConvNext_3D():
    return PVS(ConvNextEncoder(stages=3), Fusion3D([384], n = 1), Decoder3())

def ConvNext_LSTM():
    return PVS(ConvNextEncoder(stages=3), FusionLSTM([384]), Decoder3())

def ConvNext_Attention():
    return PVS(ConvNextEncoder(stages=3), FusionAttention([384], [4]), Decoder3())

def ConvNext_NSA():
    return PVS(ConvNextEncoder(stages=3), FusionNSA([384], 4, [16], [16]), Decoder3())

# Experiment 2 ############################################################################################

def ConvNext_simple_skip():
    return PVS(ConvNextEncoder(stages=3), FusionSimple([96, 192, 384], n_frames = 5), Decoder3())

def ConvNext_3D_skip():
    return PVS(ConvNextEncoder(stages=3), Fusion3D([96, 192, 384], n = 1), Decoder3())

def ConvNext_LSTM_skip():
    return PVS(ConvNextEncoder(stages=3), FusionLSTM([96, 192, 384]), Decoder3())

def ConvNext_Attention_skip():
    return PVS(ConvNextEncoder(stages=3), FusionAttention([96, 192, 384], n_heads=[4, 4, 4]), Decoder3())

def ConvNext_NSA_skip():
    return PVS(ConvNextEncoder(stages=3), FusionNSA([96, 192, 384], n_heads=4, heights=[64, 32, 16], widths=[64, 32, 16]), Decoder3())

# Experiment 3 #############################################################################################

def ConvNext_simple_enc():
    return PVS(ConvTempEncoderSimple(), NoFusion(), Decoder3())

def ConvNext_3D_enc():
    return PVS(ConvTempEncoder3D(), NoFusion(), Decoder3())

def ConvNext_LSTM_enc():
    return PVS(ConvTempEncoderLSTM(), NoFusion(), Decoder3())

def ConvNext_Attention_enc():
    return PVS(ConvTempEncoderAttention(), NoFusion(), Decoder3())

def ConvNext_NSA_enc():
    return PVS(ConvTempEncoderNSA(), NoFusion(), Decoder3())

############################################################################################################
# TESTS

def test_model():
    return PVS(ConvNextEncoder(backbone=convnext_small(weights=ConvNeXt_Small_Weights), stages=3), FusionLSTM(filters=[384]), Decoder3())

def ConvNext_LSTM_single():
    return PVS(ConvNextEncoder(stages = 3), FusionLSTM([384], bidirectional=False), Decoder3())

def ConvNext_LSTM_unpruned():
    return PVS(ConvNextEncoder(stages = 4), FusionLSTM([768], bidirectional=True, skip_connection=False), Decoder4())


if __name__ == "__main__":
    model = ConvNext_LSTM_unpruned()
    input = torch.randn(1, 5, 3, 256, 256)
    print(model)
    print(model(input).shape)

