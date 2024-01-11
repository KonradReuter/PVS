import random
from pathlib import Path
import os
import time
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
from PIL import Image, ImageStat
from torch.utils.data import DataLoader
import nibabel
from collections import OrderedDict
from config.config import args

from config.config import DATA_DIR, device, CHECKPOINT_DIR, ATT_MAP_DIR


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def print_example(dataloader: DataLoader, figsize: tuple[int, int] = (2, 2)) -> None:
    """Prints an example from the given dataloader

    Args:
        dataloader (DataLoader): Dataloader from which an example should be shown.
        figsize (tuple[int, int], optional): Size of the matplotlib figure. Defaults to (2, 2).
    """
    # get a image and mask example
    img, mask, _ = next(iter(dataloader))
    img = img[0]
    mask = mask[0]
    img_mask_list = []

    # Append all images and masks to one list
    # If it is a Video dataset (img.shape = 4), we have multiple images per example and add them one by one
    if len(img.shape) == 4:
        for i in range(img.shape[0]):
            img_mask_list.append(img[i])
        for i in range(mask.shape[0]):
            img_mask_list.append(mask[i])
    # If it is a Image dataset, we only have one image and mask and can directy append them
    elif len(img.shape) == 3:
        img_mask_list.append(img)
        img_mask_list.append(mask)
    else:
        print("UNKNOWN IMAGE FORMAT!")
        return
    # Create a new figure and add the images in the upper row and the masks in the lower row
    figure = plt.figure(figsize=figsize)
    for i in range(len(img_mask_list)):
        figure.add_subplot(2, len(img_mask_list) // 2, i + 1)
        plt.imshow(img_mask_list[i].permute(1, 2, 0))
        plt.axis("off")
    plt.show()


def make_reproducible(seed: int = 42) -> None:
    """Sets seed and configurations for reproducibility

    Args:
        seed (int, optional): Seed number. Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # deactivate benchmarking and use deterministic algorithms. This may limit the performance
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only = True)
    # For deterministic behavior of RNNS


def calculate_mean_std(dataset_list: list) -> tuple[list[float], list[float]]:
    """Calculates the channelwise mean and standard deviation over all images in a list of datasets.
    Each dataset is expected to have a list "image_list" where the path to each UNIQUE image is stored.

    Args:
        dataset_list (list): A list of datasets

    Returns:
        tuple[list[float], list[float]]: The mean and standard deviation over all images.
    """
    num_imgs = 0
    total_mean = [0, 0, 0]
    total_std = [0, 0, 0]
    # iterate through all datasets
    for dataset in dataset_list:
        # iterate through all images per dataloader
        for img_path in dataset.image_list:
            # open the image and calculate mean and std dev
            img = Image.open(img_path)
            stats = ImageStat.Stat(img)
            total_mean = np.add(total_mean, stats.mean)
            total_std = np.add(total_std, stats.stddev)
            num_imgs += 1

    return total_mean / num_imgs, total_std / num_imgs


def create_imgs_from_videos(path: str) -> None:
    """Splits videos into images.

    Args:
        path (str): Path to the folder which contains the videos.
    """
    # get a list of all files
    file_list = Path(path).glob("*.*")
    # iterate trough the files
    for video_file in file_list:
        print(video_file.name)
        # create directories
        Path(path, video_file.stem, "images_" + video_file.stem).mkdir(parents=True, exist_ok=True)
        # create video object
        video = cv2.VideoCapture(str(Path(path, video_file).resolve()))
        frame_count = 0
        # calculate the maximum amount of digits for the frame number
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        digits = len(str(int(total_frames)))
        # iterate through the video frames
        ret, frame = video.read()
        while ret:
            # fill frame count with leading zeros
            frame_count_str = str(frame_count).zfill(digits)
            # save the image and continue with the next frame
            cv2.imwrite(
                str(
                    Path(
                        path,
                        video_file.stem,
                        "images_" + video_file.stem,
                        video_file.stem + f"_{frame_count_str}.png",
                    ).resolve()
                ),
                frame,
            )
            frame_count += 1
            ret, frame = video.read()


def create_image_subfolders(path: str) -> None:
    """Creates image subfolders for all sequences under the given path and moves all the files into this folder.

    Args:
        path (str): Path for which a image subfolder should be created.
    """
    # get everything in the directory
    seq_list = Path(path).glob("*")
    # only take the sequence folders
    seq_list = [p for p in seq_list if p.is_dir()]
    # iterate through the sequences
    for seq in seq_list:
        # get a list of all files
        file_list = seq.glob("*.*")
        # create the new subfolder
        Path(seq, "images_" + seq.stem).mkdir(exist_ok=True)
        # move all files into that folder
        for file in file_list:
            file.rename(str(Path(seq, "images_" + seq.stem, file.name).resolve()))

def get_inference_speed(model: any, img_size: tuple, num_iterations: int) -> None:
    """Calculates and prints the inference speed of a model

    Args:
        model (any): Model to be evaluated.
        img_size (tuple): Shape of the model input.
        num_iterations (int): How many inferences should be run.
    """
    sample_image = torch.rand(img_size).to(device)
    start = time.time()
    for _ in range(num_iterations):
        model(sample_image)
    end = time.time()
    print(f"Needed {end-start} seconds for {num_iterations} inferences. This corresponds to {num_iterations/(end-start)} FPS.")

def cutHyperKvasirVid(n_frames: int) -> None:
    """Cuts the HyperKvasir Videos to the given length.

    Args:
        n_frames (int): How many frames should be kept.
    """
    # get video directory list
    HyperKvasir_dir = './data/HyperKvasir_vid'
    dir_list = os.listdir(HyperKvasir_dir)
    # remove items from the list which are no directories
    dir_list = [dir for dir in dir_list if os.path.isdir(os.path.join(HyperKvasir_dir, dir))]
    # iterate through videos
    for seq in dir_list:
        # define origin and target directory
        origin_dir = os.path.join(HyperKvasir_dir, seq, "images_" + seq)
        target_dir = f"./data/HyperKvasir_vid_{n_frames}/"+seq+"/images_"+seq
        os.makedirs(target_dir, exist_ok=True)
        # iterate through origin directory until the maximum number of frames is reached and copy items into target directory
        for i, img, in enumerate(os.listdir(origin_dir)):
            if i >= n_frames:
                break
            shutil.copyfile(os.path.join(origin_dir, img), os.path.join(target_dir, img))

def count_parameters(model: any) -> int:
    """Count trainable parameters in a given model

    Args:
        model (any): Model to be evaluated.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_nii_images(path: str) -> None:
    """Prints multiple images contained in a .nii file

    Args:
        path (str): Path to nii file.
    """
    imgs = nibabel.load(path).get_fdata()
    n_imgs = imgs.shape[-1]
    for i in range(n_imgs):
        plt.subplot(1, n_imgs, i+1)
        plt.imshow(imgs[..., i])
    plt.show()

def get_optimizer(optimizer: str) -> any:
    if optimizer == "Adam":
        return torch.optim.Adam
    if optimizer == "AdamW":
        return torch.optim.AdamW
    raise ModuleNotFoundError(f"The specified optimizer ({optimizer}) could not be found!")

class SingleImageModelWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super(SingleImageModelWrapper, self).__init__()
        self.model = model
        self.identity = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if args["save_attention_maps"]:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        origin_shape = x.shape
        x = x.view(-1, *x.shape[2:])
        x = self.model(x)
        if type(x) == tuple or type(x) == list:
            x = [i.view(*origin_shape[:2], *i.shape[1:]) for i in x]
        elif type(x) == OrderedDict:
            x = x["out"]
            x = x.view(*origin_shape[:2], *x.shape[1:])
        else:
            x = x.view(*origin_shape[:2], *x.shape[1:])
        if args["save_attention_maps"]:
            if type(x) == tuple or type(x) == list:
                x = [self.identity(i.permute(0, 2, 1, 3, 4).contiguous()) for i in x]
            else:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                x = self.identity(x)
        return x

def adjust_args(model_name: str, args: dict):
    """Automatically adjust args for specific SOTA models

    Args:
        model_name (str): Name of the Model
        args (dict): Args to be adjusted
    """
    model_name = model_name.split(".")[0]

    if model_name == "Conv_NSA":
        args["amp"] = False
    if model_name == "Conv_NSA_skip":
        args["amp"] = False
    if model_name == "Conv_NSA_enc":
        args["amp"] = False
    if model_name == "TransFuse":
        args["loss_factors"] = [0.2, 0.3, 0.5]
    if model_name == "PraNet":
        args["loss_factors"] = [1.0, 1.0, 1.0, 1.0]
    if model_name == "CASCADE":
        args["loss_factors"] = [1.0, 1.0, 1.0, 1.0]
    if model_name == "COSNet":
        args["num_frames"] = 2
        args["output_frames"] = -2
        args["time_interval"] = 4
    if model_name == "HybridNet":
        args["loss_factors"] = [1.0, 1.0]
        args["output_frames"] = 2
        args["unique"] = False
    if model_name == "PNSNet":
        args["amp"] = False
    if model_name == "PNSPlusNet":
        args["amp"] = False
        args["num_frames"] = 6
        args["anchor_frame"] = True
    if model_name == "VACSNet":
        args["loss_factors"] = [0.2, 0.2, 0.2, 0.2, 0.2]

    return args

if __name__ == "__main__":
    #from scripts.swin_model import PolypSwin
    #model = PolypSwin().to(device)
    #input_size = (1, 5, 3, 256, 256)
    #num_iterations = 100
    #get_inference_speed(model, input_size, num_iterations)
    #get_inference_speed(model, input_size, num_iterations)
    print_nii_images(str(ATT_MAP_DIR)+'/identity/attention_map_0_0_0.nii')
