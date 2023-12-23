import torch
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