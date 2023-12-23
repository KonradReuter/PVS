import copy
import os
from pathlib import Path

import natsort
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from config import config
from scripts.transforms import *


class ImageDataset(Dataset):
    def __init__(self, path: str, transform: any = None, masked: bool = True) -> None:
        """Initilization, generates lists for all image and mask items.

        Args:
            path (str): Path to the base folder of the dataset. The base folder should contain a "images" and a "masks" folder. The mask belonging to an image is expected to have the same name.
            transform (any, optional): Transforms that should be applied to the images/masks. Defaults to None.
            masked (bool, optional): Wether the image set is masked or not. Defaults to True.
        """
        self.image_list = natsort.natsorted(os.listdir(os.path.join(path, "images")))
        self.mask_list = copy.copy(self.image_list)
        self.image_list = [os.path.join(path, "images", img) for img in self.image_list]
        self.mask_list = [os.path.join(path, "masks", mask) for mask in self.mask_list]
        self.transform = transform
        self.masked = masked

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the Dataset.
        """
        return len(self.image_list)

    def __getitem__(self, idx: int) -> tuple[any, any]:
        """Get one item from the dataset

        Args:
            idx (int): Index of the items to return.

        Returns:
            tuple[any, any]: Image and mask corresponding to the index. Type depends on the given transformations.
        """
        image = Image.open(self.image_list[idx])
        # if a mask exists open it, otherwise create a pseudo mask to apply the transformations
        if self.masked:
            mask = Image.open(self.mask_list[idx]).convert("L")
        else:
            mask = Image.new("L", image.size)
        if self.transform:
            image, mask = self.transform([image], [mask])
        return image[0], mask[0]


class VideoDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform: any = None,
        num_frames: int = 5,
        time_interval: int = 1,
        anchor_frame: bool = True,
        masked: bool = True,
    ) -> None:
        """Initialization. The base path is expected to have a folder for each sequence.
        Each sequence folder is expected to have a images_<seq_name> and a masks_<seq_name> folder.
        The mask corresponding to an image should have the same name with an "_mask" at the end.

        Args:
            path (str): Path to the sequence data folder.
            transform (transformations, optional): Transformations to be applied on the images and the masks. Defaults to None.
            num_frames (int, optional): How many frames should be given at once. Defaults to 5.
            time_interval (int, optional): Defines the time span between two frames. E.g.: 1 -> consecutive frames, 2 -> every second frame. Defaults to 1.
            anchor_frame (bool, optional): If true, the first frame of a datapoint is always the first frame of the sequence. Defaults to True.
            masked (bool, optional): Wether masks exist for the images. Defaults to True.
        """
        # The image_list is only used to create a list of unique images for easier calculation of the mean/stddev over all image datasets.
        self.image_list = []
        self.mask_list = []
        self.train_list = []
        self.path = path
        self.transform = transform
        self.num_frames = num_frames
        self.time_interval = time_interval
        self.anchor_frame = anchor_frame
        self.masked = masked

        # create a dict for all files and a list of all sequences
        seq_file_list = {}
        seq_list = os.listdir(path)
        seq_list = [dir for dir in seq_list if os.path.isdir(os.path.join(self.path, dir))]

        # iterate through the sequences and create lists of all images and masks
        for seq in seq_list:
            seq_file_list[seq] = []
            img_dir = os.path.join(path, seq, "images_" + seq)
            mask_dir = os.path.join(path, seq, "masks_" + seq)
            img_list = natsort.natsorted(os.listdir(img_dir))
            mask_list = [
                os.path.splitext(img)[0] + "_mask" + os.path.splitext(img)[1] for img in img_list
            ]

            # append the image/mask paths as tuples to the seq_file_list
            for img_file, mask_file in zip(img_list, mask_list):
                seq_file_list[seq].append(
                    (os.path.join(img_dir, img_file), os.path.join(mask_dir, mask_file))
                )
                # append the image path to the list of unique images
                self.image_list.append(os.path.join(img_dir, img_file))
                self.mask_list.append(os.path.join(mask_dir, mask_file))

        # Set the start and consecutive_frames variable based on anchor_frame.
        if self.anchor_frame:
            start = 1
            consecutive_frames = self.num_frames - 1
        else:
            start = 0
            consecutive_frames = self.num_frames

        # Iterate through the sequence dictionary.
        for seq in seq_list:
            file_list = seq_file_list[seq]

            # For each sequence create datapoints with all possible values for begin.
            for begin in range(start, len(file_list) - (consecutive_frames-1) * time_interval):
                frame_block = []

                # if an anchor frame is used, append the path to the first (image, mask) path
                if self.anchor_frame:
                    frame_block.append(file_list[0])

                # append the remaining paths
                for t in range(consecutive_frames):
                    frame_block.append(file_list[begin + time_interval * t])

                # append the datapoint to the train_list
                self.train_list.append(frame_block)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.train_list)

    def __getitem__(self, idx: int) -> tuple[any, any]:
        """Get the items (image, mask) of a specific index.

        Args:
            idx (int): Index

        Returns:
            any: images and masks for the index. Type depends on the transformations.
        """
        img_mask_files = self.train_list[idx]
        IMG = None
        MASK = None
        img_list = []
        mask_list = []

        # load the images and masks into separate lists
        for idx, (img_path, mask_path) in enumerate(img_mask_files):
            img = Image.open(img_path)
            # if no mask path is given, the example is negative, so a black image is generated.
            if self.masked:
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", img.size)
            img_list.append(img)
            mask_list.append(mask)

        # apply the transformations
        img_list, mask_list = self.transform(img_list, mask_list)

        # rearrange the images and masks into two tensors and return them
        for idx, (img, mask) in enumerate(zip(img_list, mask_list)):
            if idx == 0:
                IMG = torch.zeros(len(img_list), *(img.shape))
                MASK = torch.zeros(len(mask_list), *(mask.shape))
                IMG[idx, :, :, :] = img
                MASK[idx, :, :, :] = mask
            else:
                IMG[idx, :, :, :] = img
                MASK[idx, :, :, :] = mask
        return IMG, MASK


class SUNSEG(Dataset):
    def __init__(self, path: str, set_list: list = None, transform: any = None, num_frames: int = 5, time_interval: int = 1, anchor_frame: bool = True, unique:bool = False) -> None:
        """Initialize SUNSEG dataset

        Args:
            path (str): Path to dataset directory.
            set_list (list, optional): If a list of sets if given, only these will be included into the dataset. Set names must match the folder names under path.
            transform (any, optional): Transformations to be applied to image/mask. Defaults to None.
            num_frames (int, optional): Number of frames per data point. Defaults to 5.
            time_interval (int, optional): Interval between frames. Defaults to 1.
            anchor_frame (bool, optional): If true, the first frame for each datapoint is the first frame of the underlying video clip. Defaults to True.
            unique (bool, optional): If true, every image will only be returned once, so there will be no overlaps between the returned data.
                                     Only the last few items might contain overlaps, if the length of the sequence is not divisible by num_frames. Defaults to False.
        """
        super(SUNSEG, self).__init__()
        self.image_list = []
        self.mask_list = []
        self.path = path
        self.set_list = set_list
        self.time_interval = time_interval
        self.anchor_frame = anchor_frame
        self.time_clips = num_frames
        self.video_train_list = []

        # get image and mask root
        img_root = os.path.join(path, 'Frame')
        gt_root = os.path.join(path, 'GT')

        # get all clips in the dataset
        cls_list = os.listdir(img_root)
        if self.set_list:
            cls_list = [clip for clip in cls_list if clip in self.set_list]
        else:
            self.set_list = cls_list
        self.video_filelist = {}
        # iterate through the clips
        for cls in cls_list:
            self.video_filelist[cls] = []

            # get img and mask paths
            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            # get all image files and sort them
            tmp_list = os.listdir(cls_img_path)
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            # iterate through the image names and append image and mask file as tuple to list
            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                ))
                # additionally append them to separate lists (to match the structure of other dataset classes)
                self.image_list.append(os.path.join(cls_img_path, filename))
                self.mask_list.append(os.path.join(cls_label_path, filename.replace(".jpg", ".png")))
        # set variables based on anchor frame variable
        if anchor_frame:
            start = 1
            consecutive_frames = num_frames - 1
        else:
            start = 0
            consecutive_frames = num_frames
        # create file list for each data point
        # iterate through classes
        for cls in cls_list:
            li = self.video_filelist[cls]
            if unique:
                step = consecutive_frames*time_interval
                end = len(li) - step + 1
                for begin in range(start, end, step):
                    for beg in range(begin, begin+time_interval):
                        batch_clips = []
                        if anchor_frame:
                            batch_clips.append(li[0])
                        for t in range(consecutive_frames):
                            batch_clips.append(li[beg + time_interval*t])
                        self.video_train_list.append(batch_clips)
                # append frames missed by previous loop
                missed_frames = len(li)%step
                if missed_frames:
                    begin = len(li)-step
                    if missed_frames < time_interval:
                        begin = begin + (time_interval-missed_frames)
                    for beg in range(begin, len(li)-time_interval*(consecutive_frames-1)):
                        batch_clips = []
                        if anchor_frame:
                            batch_clips.append(li[0])
                        for t in range(consecutive_frames):
                            batch_clips.append(li[beg+time_interval*t])
                        self.video_train_list.append(batch_clips)
            else:
                step = 1
                end = len(li) - (consecutive_frames-1)*time_interval
                for begin in range(start, end, step):
                    batch_clips = []
                    if anchor_frame:
                        batch_clips.append(li[0])
                    for t in range(consecutive_frames):
                        batch_clips.append(li[begin+time_interval*t])
                    self.video_train_list.append(batch_clips)

        self.img_label_transform = transform

    def __getitem__(self, idx: int) -> tuple[any, any]:
        """Returns data point for the given index.

        Args:
            idx (int): Index

        Returns:
            tuple[any, any]: Tuple of images and masks.
        """
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        img_path_list = []
        # open and append all images and labels
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            img_path_list.append(img_path)
        # apply transformations
        img_li, label_li = self.img_label_transform(img_li, label_li)
        # Stack images and masks
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))

                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label

        return IMG, LABEL, img_path_list

    def __len__(self) -> int:
        """Get number of data points

        Returns:
            int: Number of data points
        """
        return len(self.video_train_list)

def get_subsets(train_set: Dataset, test_set: Dataset, train_size: float) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Generates subsets from a given train and test set. Train and test set are expected to contain the same data but with different transformations.

    Args:
        train_set (Dataset): Dataset with train transformations
        test_set (Dataset): Dataset with test transformations
        train_size (float): Size of the train set [0-1].

    Returns:
         tuple[torch.utils.data.Subset, torch.utils.data.Subset]: Train and test subset from the input datasets.
    """
    # create list of all indices
    indices = list(range(len(train_set)))
    # set seed for reproducibility and shuffle indices
    random.shuffle(indices)
    # generate index list for train and test set
    split_idx = int(np.floor(len(indices) * train_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    # create subsets
    train_subset = torch.utils.data.Subset(train_set, train_indices)
    test_subset = torch.utils.data.Subset(test_set, test_indices)
    return train_subset, test_subset

def get_dataloaders(args: dict) -> dict:
    """Function to generate dataloaders for the datasets

    Args:
        args (dict): Dictionary containg hyperparameters.

    Returns:
        dict: Dictionary with the different dataloaders.
    """

    # transforms for the masked datasets
    # for the train set
    transforms = Compose_img_mask(
        [
            Resize(args["img_height"], args["img_width"]),
            RandomCropResize(args["random_crop_size"]),
            RandomHorizontalFlip(args["random_flip_prob"]),
            RandomVerticalFlip(args["random_flip_prob"]),
            RandomRotation(args["random_rotation_prob"], args["random_rotation_range"]),
            RandomBlur(args["random_blur_prob"], args["random_blur_radius"]),
            RandomBrightnessShift(args["random_brightness_factor"], args["random_brightness_probability"]),
            toTensor(),
            Normalize(
                np.array(args["normalize_mean"]), np.array(args["normalize_std"])
            ),
        ]
    )
    # for validation/test set
    test_transforms = Compose_img_mask(
        [
            Resize(args["img_height"], args["img_width"]),
            toTensor(),
            Normalize(
                np.array(args["normalize_mean"]), np.array(args["normalize_std"])
            )
        ]
    )
    # transforms for the unmasked (pretraining) datasets
    # for the train set
    inpainting_transforms = Compose_img_mask(
        [
            Resize(args["img_height"], args["img_width"]),
            RandomCropResize(args["random_crop_size"]),
            RandomHorizontalFlip(args["random_flip_prob"]),
            RandomVerticalFlip(args["random_flip_prob"]),
            RandomRotation(args["random_rotation_prob"], args["random_rotation_range"]),
            RandomBlur(args["random_blur_prob"], args["random_blur_radius"]),
            RandomBrightnessShift(args["random_brightness_factor"], args["random_brightness_probability"]),
            toTensor(),
            Normalize(
                np.array(args["normalize_mean"]), np.array(args["normalize_std"])
            ),
            RandomRectDropout(args["rect_dropout"]),
            RandomImageDrop(args["img_dropout"])
        ]
    )
    # for validation set
    test_inpainting_transforms = Compose_img_mask(
        [
            Resize(args["img_height"], args["img_width"]),
            toTensor(),
            Normalize(
                np.array(args["normalize_mean"]), np.array(args["normalize_std"])
            ),
            RandomRectDropout(args["rect_dropout"]),
            RandomImageDrop(args["img_dropout"])
        ]
    )

    # Create the different image datasets
    # inpainting
    #HyperKvasir_img_train = ImageDataset(Path(config.DATA_DIR, "HyperKvasir_img"), transform=inpainting_transforms, masked=False)
    #HyperKvasir_img_test = ImageDataset(Path(config.DATA_DIR, "HyperKvasir_img"), transform=test_inpainting_transforms, masked=False)
    #HK_train, HK_test = get_subsets(HyperKvasir_img_train, HyperKvasir_img_test, args["pretrain_split"][0])

    # segmentation
    #KVASIR_SEG_train = ImageDataset(Path(config.DATA_DIR, "KVASIR_SEG"), transform=transforms)
    #KVASIR_SEG_test = ImageDataset(Path(config.DATA_DIR, "KVASIR_SEG"), transform=test_transforms)
    #SEG_train, SEG_test = get_subsets(KVASIR_SEG_train, KVASIR_SEG_test, args["pretrain_split"][0])

    # create datasets

    #HyperKvasir_vid_train = VideoDataset(
    #    Path(config.DATA_DIR, "HyperKvasir_vid_60/train"),
    #    transform=inpainting_transforms,
    #    anchor_frame=args["anchor_frame"],
    #    num_frames=args["num_frames"],
    #    time_interval=args["time_interval"],
    #    masked=False
    #)

    #HyperKvasir_vid_test = VideoDataset(
    #    Path(config.DATA_DIR, "HyperKvasir_vid_60/validation"),
    #    transform=test_inpainting_transforms,
    #    num_frames=args["num_frames"],
    #    time_interval=args["time_interval"],
    #    anchor_frame=args["anchor_frame"],
    #    masked=False
    #)

    SUNSEG_train_path = Path(config.DATA_DIR, "SUN-SEG/TrainDataset")
    #SUNSEG_sets = sorted(os.listdir(Path(SUNSEG_train_path, "Frame")), key=lambda x: int(x.split('e')[1].split('_')[0]))
    SUNSEG_sets = ['case1_1', 'case2_1', 'case4', 'case5_1', 'case6_1', 'case7_1',
                   'case15_1', 'case16', 'case17_1', 'case18_1', 'case20_1', 'case21',
                   'case22', 'case25_1', 'case26_1', 'case28', 'case33_1', 'case35_4',
                   'case38_1', 'case41', 'case44_2', 'case45_1', 'case46', 'case47_1',
                   'case49', 'case53', 'case55_1', 'case57', 'case58', 'case59_1',
                   'case61_1', 'case62_1', 'case63_1', 'case65', 'case66_1', 'case69',
                   'case71_3', 'case72_2', 'case73_1', 'case75_2', 'case76_1', 'case77',
                   'case78_1', 'case82', 'case83_1', 'case85_1', 'case87_1', 'case88_1',
                   'case90_2', 'case92_1', 'case98']
    SUNSEG_sets = np.array_split(SUNSEG_sets, args["num_folds"])
    SUNSEG_validation_set = list(SUNSEG_sets[args["validation_fold"]])
    SUNSEG_train_set = list(np.concatenate([SUNSEG_sets[i] for i in range(len(SUNSEG_sets)) if i != args["validation_fold"]]))

    SUNSEG_train = SUNSEG(
        SUNSEG_train_path,
        SUNSEG_train_set,
        transform=transforms,
        num_frames=args["num_frames"],
        time_interval=args["time_interval"],
        anchor_frame=args["anchor_frame"]
    )
    SUNSEG_valid = SUNSEG(
        SUNSEG_train_path,
        SUNSEG_validation_set,
        transform=test_transforms,
        num_frames=args["num_frames"],
        time_interval=args["time_interval"],
        anchor_frame=args["anchor_frame"]
    )

    SUNSEG_test_easy_seen = SUNSEG(
        Path(config.DATA_DIR, "SUN-SEG/TestEasyDataset/Seen"),
        transform=test_transforms,
        num_frames=args["num_frames"],
        time_interval=args["time_interval"],
        anchor_frame=args["anchor_frame"],
        unique=args["unique"]
    )

    SUNSEG_test_easy_unseen = SUNSEG(
        Path(config.DATA_DIR, "SUN-SEG/TestEasyDataset/Unseen"),
        transform=test_transforms,
        num_frames=args["num_frames"],
        time_interval=args["time_interval"],
        anchor_frame=args["anchor_frame"],
        unique=args["unique"]
    )

    SUNSEG_test_hard_seen = SUNSEG(
        Path(config.DATA_DIR, "SUN-SEG/TestHardDataset/Seen"),
        transform=test_transforms,
        num_frames=args["num_frames"],
        time_interval=args["time_interval"],
        anchor_frame=args["anchor_frame"],
        unique=args["unique"]
    )

    SUNSEG_test_hard_unseen = SUNSEG(
        Path(config.DATA_DIR, "SUN-SEG/TestHardDataset/Unseen"),
        transform=test_transforms,
        num_frames=args["num_frames"],
        time_interval=args["time_interval"],
        anchor_frame=args["anchor_frame"],
        unique=args["unique"]
    )

    # for reproducibility
    def seed_workers(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders from all datasets
    #HK_train_dl = DataLoader(HK_train, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    #HK_valid_dl = DataLoader(HK_test, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    #image_train_dl = DataLoader(SEG_train, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    #image_valid_dl = DataLoader(SEG_test, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    #HK_vid_train_dl = DataLoader(HyperKvasir_vid_train, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    #HK_vid_valid_dl = DataLoader(HyperKvasir_vid_test, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    SUNSEG_train_dl = DataLoader(SUNSEG_train, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    SUNSEG_valid_dl = DataLoader(SUNSEG_valid, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    SUNSEG_test_easy_seen_dl = DataLoader(SUNSEG_test_easy_seen, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    SUNSEG_test_easy_unseen_dl = DataLoader(SUNSEG_test_easy_unseen, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    SUNSEG_test_hard_seen_dl = DataLoader(SUNSEG_test_hard_seen, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)
    SUNSEG_test_hard_unseen_dl = DataLoader(SUNSEG_test_hard_unseen, batch_size=args["batch_size"], shuffle=args["shuffle_data"], num_workers=args["n_workers"], worker_init_fn=seed_workers, generator=g)


    # create and return dictionary with all datasets
    dataloaders = {
        #"unmasked_img_train": HK_train_dl,
        #"unmasked_img_valid": HK_valid_dl,
        #"unmasked_vid_train": HK_vid_train_dl,
        #"unmasked_vid_valid": HK_vid_valid_dl,
        #"masked_img_train": image_train_dl,
        #"masked_img_valid": image_valid_dl,
        "masked_vid_train": SUNSEG_train_dl,
        "masked_vid_valid": SUNSEG_valid_dl,
        "masked_vid_test_easy_seen": SUNSEG_test_easy_seen_dl,
        "masked_vid_test_easy_unseen": SUNSEG_test_easy_unseen_dl,
        "masked_vid_test_hard_seen": SUNSEG_test_hard_seen_dl,
        "masked_vid_test_hard_unseen": SUNSEG_test_hard_unseen_dl
    }

    return dataloaders

def get_datasets() -> list[Dataset]:
    """Returns a list of all datasets. Used to calculate mean and standard deviation.

    Returns:
        list[Dataset]: List of all datasets.
    """
    #HyperKvasir_img= ImageDataset(Path(config.DATA_DIR, "HyperKvasir_img"), masked=False)
    #Kvasir_SEG = ImageDataset(Path(config.DATA_DIR, "KVASIR_SEG"))
    #HyperKvasir_vid_train = VideoDataset(Path(config.DATA_DIR, "HyperKvasir_vid_60/train"), masked=False)
    #HyperKvasir_vid_test = VideoDataset(Path(config.DATA_DIR, "HyperKvasir_vid_60/validation"), masked=False)
    SUNSEG_train = SUNSEG(Path(config.DATA_DIR, "SUN-SEG/TrainDataset"))
    SUNSEG_valid = SUNSEG(Path(config.DATA_DIR, "SUN-SEG/ValidationDataset"))
    SUNSEG_test_easy_seen = SUNSEG(Path(config.DATA_DIR, "SUN-SEG/TestEasyDataset/Seen"))
    SUNSEG_test_easy_unseen = SUNSEG(Path(config.DATA_DIR, "SUN-SEG/TestEasyDataset/Unseen"))
    SUNSEG_test_hard_seen = SUNSEG(Path(config.DATA_DIR, "SUN-SEG/TestHardDataset/Seen"))
    SUNSEG_test_hard_unseen = SUNSEG(Path(config.DATA_DIR, "SUN-SEG/TestHardDataset/Unseen"))

    dataset_list = [#HyperKvasir_img,
                    #Kvasir_SEG,
                    #HyperKvasir_vid_train,
                    #HyperKvasir_vid_test,
                    SUNSEG_train,
                    SUNSEG_valid,
                    SUNSEG_test_easy_seen,
                    SUNSEG_test_easy_unseen,
                    SUNSEG_test_hard_seen,
                    SUNSEG_test_hard_unseen]

    return dataset_list

if __name__ == '__main__':
    import json
    args = json.load(open("./config/args.json"))
    dl = get_dataloaders(args)
    train_sets = dl["masked_vid_train"].dataset.image_list
    valid_sets = dl["masked_vid_valid"].dataset.image_list
    schnittmenge = [img for img in train_sets if img in valid_sets]
    print(schnittmenge)