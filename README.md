# Implementation and Evaluation of Deep Learning Methods for Colorectal Polyp Video Segmentation

The repository is currently in the state how it was during my master thesis. Adjustments to make it more user-friendly will follow.

## Data preparation

Clone the repository and create a data folder:
 ```
 git clone https://github.com/KonradReuter/PVS.git
 cd PVS
 mkdir data
 ```

Request and prepare the SUN-SEG dataset as explained [here](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md) and move it to the created data folder.


## Create Virtual Environment and Install Dependencies

```
python3 -m venv venv
venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e
```

## Training & Evaluating

There are four scripts for training and evaluating models:
- **train_models.sh**: Trains and evaluates all constructed models using 5-fold cross validation.
- **train_num_frames.sh**: Trains and evaluates our SOTA model with varying number of input frames using 5-fold cross validation.
- **train_single.sh**: Trains a model using a single direction ConvLSTM instead of a bidirectional one using 5-fold cross validation.
- **train_sota.sh**: Trains and evaluates 10 state-of-the-art models using 5-fold cross validation.

The training progress and evaluation results are logged in WandB.

### HybridNet

The code for the [Hybird2D/3D model](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_29) is not publicly available. We want to thank the authors for providing it to us. If you want to repeat our experiments, please send them a request as well.

The model will need some adjustments to run in our environment. For more information, contact us once you received the original model code.

### PNSNet/PNSPlusNet

The PNSNet and PNSPlusNet are based on the NS-Block module. To be able to use it, a CUDA extension has to be installed. [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) is required.

```
cd scripts/models/PNSPlus
python setup.py build develop
```

### SOTA backbone weights

CASCADE, COSNet and TransFuse use pretrained backbone weights, which are not downloaded automatically. Please refer to the according repository linked below in order to get the weight files and move them into the according model folder.

## Code from other repositories

This repository includes code from various other repository. Many thanks to the authors for providing their code:

- [Pytorch-Unet](https://github.com/milesial/Pytorch-UNet)
- [PraNet](https://github.com/DengPingFan/PraNet)
- [SANet](https://github.com/weijun88/SANet)
- [TransFuse](https://github.com/Rayicer/TransFuse)
- [CASCADE](https://github.com/SLDGroup/CASCADE)
- [COSNet](https://github.com/carrierlxk/COSNet)
- [PNSNet](https://github.com/GewelsJI/PNS-Net)
- [PNSPlusNet](https://github.com/GewelsJI/VPS)
- [SSTAN](https://github.com/ShinkaiZ/SSTAN-VPS)
- [ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch/)
- [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch)