from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import ResNet50_Weights
from scripts.utils import SingleImageModelWrapper

#import torch

def get_DeepLab():
    #model = SingleImageModelWrapper(deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.DEFAULT, num_classes = 21))
    #model.model.classifier[4] = torch.nn.Conv2d(256, 1, 1)
    #return model

    return SingleImageModelWrapper(deeplabv3_resnet50(weights_backbone = ResNet50_Weights.DEFAULT, num_classes = 1))

if __name__ == '__main__':
    import torch
    model = get_DeepLab()
    input = torch.randn(2, 5, 3, 256, 256)
    out = model(input)
    print(out.shape)