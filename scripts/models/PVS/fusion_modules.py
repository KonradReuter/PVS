import torch
from scripts.models.UNet.modules import DoubleConv, DoubleConv3D
from scripts.models.PVS.convlstm import BidirectionalConvLSTM, ConvLSTM
from scripts.models.PNSPlus.LightRFB import LightRFB
#from scripts.models.PNSPlus.PNSPlusModule import NS_Block
from torchvision.models.swin_transformer import SwinTransformerBlockV2

def temporal2spatial(x: torch.Tensor):
    return x.view(-1, *x.shape[2:])

def spatial2temporal(x:torch.Tensor, origin_shape: torch.Size):
    return x.view(*origin_shape[:2], *x.shape[1:])

class NoFusion(torch.nn.Module):
    """Module to be inserted if no temporal fusion should be applied.
    """
    def __init__(self) -> None:
        """Initialization
        """
        super(NoFusion, self).__init__()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Simply flattens batch and time dimension into one.

        Args:
            x (torch.Tensor): Input tensor of shape b x f x c x h x w

        Returns:
            torch.Tensor: Output tensor of shape b*f x c x h x w
        """

        return [temporal2spatial(t) for t in x]

class FusionSimple(torch.nn.Module):
    """Temporal fusing by concatenation along channel dimension and double convolution
    """
    def __init__(self, filters: list[int], n_frames: int, skip_connection: bool = True, squeeze: bool = True) -> None:
        """Initialization

        Args:
            filters (int): Number of input filters (=output filters)
            n_frames (int): Number of frames to be fused
        """
        super(FusionSimple, self).__init__()
        self.skip_connection = skip_connection
        if squeeze:
            self.fusion_modules = torch.nn.ModuleList([DoubleConv(n_frames*f, n_frames*f, f) for f in filters])
        else:
            self.fusion_modules = torch.nn.ModuleList([DoubleConv(n_frames*f, n_frames*f, n_frames*f) for f in filters])

    def _fuse(self, x, idx):
        shape = x.shape
        fw = x.contiguous().view(shape[0], -1, *shape[3:])
        fw = self.fusion_modules[idx](fw)
        fw = fw.view(*shape)
        if self.skip_connection:
            return x+fw
        else:
            return fw

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Concatenate along channel dimension and double convolution

        Args:
            x (torch.Tensor): Input tensor in form b x f x c x h x w

        Returns:
            torch.Tensor: Output tensor in form b*f x c x h x w
        """
        out = []
        if len(self.fusion_modules) == 1:
            out += [temporal2spatial(t) for t in x[:-1]]
            out.append(temporal2spatial(self._fuse(x[-1], 0)))
        else:
            for i, t in enumerate(x):
                out.append(temporal2spatial(self._fuse(t, i)))
        return out

class Fusion3D(torch.nn.Module):
    """Temporal Fusion using 3D convolutions
    """
    def __init__(self, filters: list[int], n: int = 1, reduce_frame_dim: bool = False, skip_connection: bool = False, squeeze: bool = False, n_frames: int = 5) -> None:
        """Initialization

        Args:
            filters (int): Input filters (= output filters)
        """
        super(Fusion3D, self).__init__()
        self.skip_connection = skip_connection
        if reduce_frame_dim:
            assert n == 1, f"Illegal Fusion3D configuration. n must be 1, if reduce_frame_dim is set to true!"
        if not squeeze:
            self.fusion_modules = torch.nn.ModuleList([torch.nn.Sequential(*[DoubleConv3D(f, f, f, reduce_frame_dim=reduce_frame_dim) for _ in range(n)]) for f in filters])
        else:
            self.fusion_modules = torch.nn.ModuleList([torch.nn.Sequential(*[DoubleConv3D(f, f, f//n_frames, reduce_frame_dim=reduce_frame_dim) for _ in range(n)]) for f in filters])


    def _fuse(self, x, idx):
        fw = x.permute(0, 2, 1, 3, 4).contiguous()
        fw = self.fusion_modules[idx](fw)
        fw = fw.permute(0, 2, 1, 3, 4).contiguous()
        if self.skip_connection:
            return x+fw
        else:
            return fw

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuses temporal information using 3D convolutions

        Args:
            x (torch.Tensor): Input tensor in form b x f x c x h x w

        Returns:
            torch.Tensor: Output tensor in form b*f x c x h x w
        """
        out = []
        if len(self.fusion_modules) == 1:
            out += [temporal2spatial(t) for t in x[:-1]]
            out.append(temporal2spatial(self._fuse(x[-1], 0)))
        else:
            for i, t in enumerate(x):
                out.append(temporal2spatial(self._fuse(t, i)))
        return out


class FusionLSTM(torch.nn.Module):
    "Temporal Fusing using bidirectional LSTM"
    def __init__(self, filters: list[int], bidirectional = True, skip_connection = False) -> None:
        """Initialization

        Args:
            filters (int): Number of input filters (=output filters)
        """
        super(FusionLSTM, self).__init__()
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fusion_modules = torch.nn.ModuleList([BidirectionalConvLSTM(f, hidden_dim=[f//2], kernel_size=(3,3), num_layers=1, batch_first=True, merge_mode='concat', skip_connection=skip_connection) for f in filters])
        else:
            self.fusion_modules = torch.nn.ModuleList([ConvLSTM(f, hidden_dim=[f], kernel_size=[(3,3)], num_layers=1, batch_first=True) for f in filters])


    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply temporal fusion

        Args:
            x (torch.Tensor): Input tensor in form b x f x c x h x w

        Returns:
            torch.Tensor: Output tensor in form b*f x c x h x w
        """
        out = []
        if len(self.fusion_modules) == 1:
            out += [temporal2spatial(t) for t in x[:-1]]
            if self.bidirectional:
                out.append(temporal2spatial(self.fusion_modules[0](x[-1])))
            else:
                out.append(temporal2spatial(self.fusion_modules[0](x[-1])[0][-1]))
        else:
            for i, t in enumerate(x):
                if self.bidirectional:
                    out.append(temporal2spatial(self.fusion_modules[i](t)))
                else:
                    out.append(temporal2spatial(self.fusion_modules[i](t)[0][-1]))
        return out


class FusionAttention(torch.nn.Module):
    """Temporal fusing using a temporal and a spatial attention module
    """
    def __init__(self, filters: list[int], n_heads: list[int]) -> None:
        """Initialization

        Args:
            filters (list[int]): Number of input filters (= output filters)
            n_heads (list[int]): Number of heads for multi head attention
        """
        super(FusionAttention, self).__init__()
        self.fusion_modules = torch.nn.ModuleList([FusionAttentionSingle(f, h) for f, h in zip(filters, n_heads)])

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Applies temporal local attention

        Args:
            x (torch.Tensor): Input tensor in form b x f x c x h x w

        Returns:
            torch.Tensor: Output tensor in form b*f x c x h x w
        """
        out = []
        if len(self.fusion_modules) == 1:
            out += [temporal2spatial(t) for t in x[:-1]]
            out.append(temporal2spatial(self.fusion_modules[0](x[-1])))
        else:
            for i, t in enumerate(x):
                out.append(temporal2spatial(self.fusion_modules[i](t)))

        return out


class FusionAttentionSingle(torch.nn.Module):
    def __init__(self, filters, n_heads)->None:
        super(FusionAttentionSingle, self).__init__()
        self.fuse1 = torch.nn.MultiheadAttention(filters, n_heads, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(filters)
        self.ln2 = torch.nn.LayerNorm(filters)
        self.ff1 = torch.nn.Sequential(
            torch.nn.Linear(filters, filters*4),
            torch.nn.ReLU(),
            torch.nn.Linear(filters*4, filters),
            torch.nn.ReLU(),
        )
        self.fuse2 = SwinTransformerBlockV2(filters, n_heads, window_size=[8, 8], shift_size=[0, 0])
        self.fuse3 = torch.nn.MultiheadAttention(filters, n_heads, batch_first=True)
        self.ln3 = torch.nn.LayerNorm(filters)
        self.ln4 = torch.nn.LayerNorm(filters)
        self.ff2 = torch.nn.Sequential(
            torch.nn.Linear(filters, filters*4),
            torch.nn.ReLU(),
            torch.nn.Linear(filters*4, filters),
            torch.nn.ReLU(),
        )
        self.fuse4 = SwinTransformerBlockV2(filters, n_heads, window_size=[8, 8], shift_size=[4, 4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape into b x h x w x f x c
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x_shape = x.shape
        # reshape into b*h*w x f x c
        fw = x.view(-1, *x_shape[3:])
        # apply temporal MHA
        fw = self.fuse1(fw, fw, fw)[0]
        # back to original shape
        fw = fw.view(*x_shape)
        # layernorm and skip connection
        x = self.ln1(fw)+x
        # feedforward
        fw = self.ff1(x)
        # layernorm and skip connection
        x = self.ln2(fw)+x
        # reorder to b x f x h x w x c
        x = x.permute(0, 3, 1, 2, 4).contiguous()

        x_shape = x.shape
        # reshape into b*f x h x w x c
        x = x.view(-1, *x.shape[2:])
        # apply spatial MHA
        x = self.fuse2(x)
        # reshape back to b x f x h x w x c
        x = x.view(*x_shape)

        # reshape into b h w f c
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x_shape = x.shape
        # reshape into b*h*w x f x c
        fw = x.view(-1, *x.shape[3:])
        fw = self.fuse3(fw, fw, fw)[0]
        fw = fw.view(*x_shape)
        x = self.ln3(fw)+x
        fw = self.ff2(x)
        x = self.ln4(fw)+x
        x = x.permute(0, 3, 1, 2, 4).contiguous()

        x_shape = x.shape
        x = x.view(-1, *x.shape[2:])
        x = self.fuse4(x)
        x = x.view(*x_shape)

        # reorder back to b x f x c x h x w
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x

class FusionNSA(torch.nn.Module):
    """Temporal fusing using normalized self attention blocks
    """
    def __init__(self, filters: list[int], n_heads: int, heights: list[int], widths: list[int], skip_connection=True) -> None:
        """Initialization

        Args:
            filters (int): Number of input filters (=output filters)
            n_heads (int): Number of heads for the attention blocks
            h (int): Input height
            w (int): Input width
        """
        super(FusionNSA, self).__init__()
        self.fusion_modules = torch.nn.ModuleList([FusionNSASingle(f, n_heads, h, w, skip_connection=skip_connection) for f, h, w in zip(filters, heights, widths)])

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Applies normalized self attention blocks

        Args:
            x (torch.Tensor): Input tensor of shape b x f x c x h x w

        Returns:
            torch.Tensor: Output tensor of shape b*f x c x h x w
        """
        out = []
        if len(self.fusion_modules) == 1:
            out += [temporal2spatial(t) for t in x[:-1]]
            out.append(temporal2spatial(self.fusion_modules[0](x[-1])))
        else:
            for i, t in enumerate(x):
                out.append(temporal2spatial(self.fusion_modules[i](t)))

        return out

class FusionNSASingle(torch.nn.Module):
    def __init__(self, filters, n_heads, height, width, skip_connection) -> None:
        super(FusionNSASingle, self).__init__()
        self.RFB = LightRFB(filters, filters//4, filters//8)
        self.fuse1 = NS_Block(channels_in=filters//8, n_head=n_heads, h = height, w = width, radius = [3, 3, 3, 3], dilation=[3, 4, 3, 4])
        self.fuse2 = NS_Block(channels_in=filters//8, n_head=n_heads, h = height, w = width, radius = [3, 3, 3, 3], dilation=[1, 2, 1, 2])
        self.excitation = torch.nn.Sequential(torch.nn.Linear(filters//8, filters), torch.nn.LayerNorm(filters), torch.nn.ReLU())
        self.skip_connection = skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape
        fw = x.view(-1, *x.shape[2:])
        fw = self.RFB(fw)
        fw = fw.view(*origin_shape[:2], *fw.shape[1:])
        fw2 = self.fuse1(fw, fw)+fw
        fw = self.fuse2(fw2, fw2)+fw2+fw
        fw = fw.contiguous()
        fw = fw.permute(0, 1, 3, 4, 2)
        fw = self.excitation(fw)
        fw = fw.permute(0, 1, 4, 2, 3)
        if self.skip_connection:
            x = fw+x
            return x
        else:
            return fw

if __name__ == '__main__':
    pass
