import random

import torch
from numpy import ndarray
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor
from config.config import logger


class Compose_img_mask(object):
    def __init__(self, transforms: list) -> None:
        """Composes multiple transformations

        Args:
            transforms (list): List of transformations.
        """
        self.transforms = transforms

    def __call__(
        self, img: list[Image.Image | torch.Tensor], mask: list[Image.Image | torch.Tensor]
    ) -> tuple[list[Image.Image | torch.Tensor], list[Image.Image | torch.Tensor]]:
        """Applies all transformations to the given image and mask.

        Args:
            img (list[Image.Image | torch.Tensor]): list of images
            mask (list[Image.Image | torch.Tensor]): list of groundtruth masks

        Returns:
            tuple[list[Image.Image | torch.Tensor], list[Image.Image | torch.Tensor]]: The transformed images and masks.
        """
        # apply the list of transformation to the images and the masks
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Resize(object):
    def __init__(self, height: int, width: int) -> None:
        """Resizes an image and a mask to the given dimensions.

        Args:
            height (int): target height in pixels
            width (int): target width in pixels
        """
        self.height = height
        self.width = width

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Applies the transformation to a list of images and a list of masks

        Args:
            imgs (list[Image.Image]):list of images
            masks (list[Image.Image]): list of (groundtruth) masks

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: The lists of resized images and resized masks
        """
        res_img = []
        res_mask = []

        for img, mask in zip(imgs, masks):
            res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
            res_mask.append(mask.resize((self.width, self.height), Image.NEAREST))

        return res_img, res_mask

class MultiResolutionResize(object):
    def __init__(self, sizes: list[tuple]) -> None:
        """Resizes an image and a mask to the given dimensions.

        Args:
            sizes: list of tuples in form (height, width)
        """
        self.sizes = sizes

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Applies the transformation to a list of images and a list of masks

        Args:
            imgs (list[Image.Image]):list of images
            masks (list[Image.Image]): list of (groundtruth) masks

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: The lists of resized images and resized masks
        """
        res_img = []
        res_mask = []

        size_idx = random.randint(0, len(self.sizes-1))

        for img, mask in zip(imgs, masks):
            res_img.append(img.resize(self.sizes[size_idx], Image.BILINEAR))
            res_mask.append(mask.resize(self.sizes[size_idx], Image.NEAREST))

        return res_img, res_mask

class Normalize(object):
    def __init__(self, mean: ndarray, std: ndarray) -> None:
        """Normalizes a list of images with the given mean and standard deviation.

        Args:
            mean (ndarray): array of mean values for the three channels
            std (ndarray): array of standard deviations for each channel
        """
        self.mean = mean
        self.std = std

    def __call__(
        self, imgs: list[torch.Tensor], masks: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Normalizes the images. The masks are only passed for convenience and will not be transformed.

        Args:
            imgs (list[torch.Tensor]): list of images
            masks (list[torch.Tensor]): list of (groundtruth) masks

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Normalized images together with the masks
        """
        norm_img = []
        for img in imgs:
            for i in range(imgs[0].shape[0]):
                img[i, :, :] -= float(self.mean[i])
                img[i, :, :] /= float(self.std[i])
            norm_img.append(img)
        return norm_img, masks


class toTensor(object):
    def __init__(self) -> None:
        """Transforms images and masks to tensors."""
        self.toTensor = ToTensor()

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Transform a list of images and masks to tensors

        Args:
            imgs (list[Image.Image]): list of pillow images
            masks (list[Image.Image]): list of pillow masks

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: lists of tensors
        """
        tens_img = []
        tens_mask = []
        for img, mask in zip(imgs, masks):
            img = self.toTensor(img)
            mask = self.toTensor(mask)
            tens_img.append(img)
            tens_mask.append(mask)
        return tens_img, tens_mask


class RandomCropResize(object):
    def _randomCrop(
        self, img: Image.Image, mask: Image.Image, x: int, y: int
    ) -> tuple[Image.Image, Image.Image]:
        """Crops the image and mask

        Args:
            img (Image.Image): image
            mask (Image.Image): groundtruth mask
            x (int): number of pixels to cut off at the left and the right side
            y (int): number of pixels to cut off at the top and the bottom

        Returns:
            tuple[Image.Image, Image.Image]: Cropped and resized image and mask
        """
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, mask = img.crop(region), mask.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        mask = mask.resize((width, height), Image.NEAREST)
        return img, mask

    def __init__(self, crop_size: int) -> None:
        """Randomly cuts off pixels at the borders and resizes to the original image size

        Args:
            crop_size (int): maximum number of pixels to cut off
        """
        self.crop_size = crop_size

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Iterate through the lists and applies the transformation on each image/mask.

        Args:
            imgs (list[Image.Image]): list of images
            masks (list[Image.Image]): list of masks

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: Cropped and resized images.
        """
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)

        crop_img = []
        crop_mask = []
        for img, mask in zip(imgs, masks):
            img, mask = self._randomCrop(img, mask, x, y)
            crop_img.append(img)
            crop_mask.append(mask)

        return crop_img, crop_mask


class RandomHorizontalFlip(object):
    def __init__(self, prob: float) -> None:
        """Applies a random horizontal flip with a given probability.

        Args:
            prob (float): Probability with which the image is flipped.
        """
        self.prob = prob

    def _horizontal_flip(
        self, img: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """Flips the image and mask

        Args:
            img (Image.Image): Image to be flipped.
            mask (Image.Image): Mask to be flipped.

        Returns:
            tuple(Image.Image, Image.Image): Flipped image and mask.
        """
        return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Flips the image with the given probability

        Args:
            imgs (list[Image.Image]): list of images
            masks (list[Image.Image]): list of masks

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: The image and masks lists, either flipped or not changed.
        """
        if random.random() < self.prob:
            flip_img = []
            flip_mask = []
            for img, mask in zip(imgs, masks):
                img, mask = self._horizontal_flip(img, mask)
                flip_img.append(img)
                flip_mask.append(mask)
            return flip_img, flip_mask
        else:
            return imgs, masks

class RandomVerticalFlip(object):
    def __init__(self, prob: float) -> None:
        """Applies a random vertical flip with a given probability.

        Args:
            prob (float): Probability with which the image is flipped.
        """
        self.prob = prob

    def _vertical_flip(
        self, img: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """Flips the image and mask

        Args:
            img (Image.Image): Image to be flipped.
            mask (Image.Image): Mask to be flipped.

        Returns:
            tuple(Image.Image, Image.Image): Flipped image and mask.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Flips the image with the given probability

        Args:
            imgs (list[Image.Image]): list of images
            masks (list[Image.Image]): list of masks

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: The image and masks lists, either flipped or not changed.
        """
        if random.random() < self.prob:
            flip_img = []
            flip_mask = []
            for img, mask in zip(imgs, masks):
                img, mask = self._vertical_flip(img, mask)
                flip_img.append(img)
                flip_mask.append(mask)
            return flip_img, flip_mask
        else:
            return imgs, masks

class RandomRotation(object):
    "Randomly rotate about a random amount of degrees"
    def __init__(self, prob:float, degree_range:tuple[int, int]) -> None:
        """Initialization

        Args:
            prob (float): Probability with which should be rotated
            degree_range (tuple[int, int]): The amount of rotation is randomly picked from this interval.
        """
        self.prob = prob
        self.min_deg = degree_range[0]
        self.max_deg = degree_range[1]

    def __call__(self, imgs: list[Image.Image], masks: list[Image.Image]) -> tuple[list[Image.Image], list[Image.Image]]:
        """Apply random rotation

        Args:
            imgs (list[Image.Image]): List of images to be rotated.
            masks (list[Image.Image]): List of corresponding masks. Will be rotated in the same way as the images.

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: Images and masks. Either rotated or unchanged.
        """
        if random.random() < self.prob:
            rotated_imgs = []
            rotated_masks = []
            angle = random.randint(self.min_deg, self.max_deg)
            for img, mask in zip(imgs, masks):
                rotated_imgs.append(img.rotate(angle))
                rotated_masks.append(mask.rotate(angle))
            return rotated_imgs, rotated_masks
        else:
            return imgs, masks

class RandomBlur(object):
    """Applies random gaussian blurring
    """
    def __init__(self, prob:float, radius:int = 2) -> None:
        """Initialization

        Args:
            prob (float): Probability with which blurring should be applied.
            radius (int, optional): Radius with which blurring should be applied. Defaults to 2.
        """
        self.prob = prob
        self.radius = radius

    def __call__(self, imgs: list[Image.Image], masks: list[Image.Image]) -> tuple[list[Image.Image], list[Image.Image]]:
        """Applies random blurring

        Args:
            imgs (list[Image.Image]): List of images to be blurred.
            masks (list[Image.Image]): List of corresponding masks.

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: List of possibly blurred images and masks. Masks will always be unchanged.
        """
        if random.random() < self.prob:
            blur_imgs = []
            for img in imgs:
                blur_imgs.append(img.filter(ImageFilter.GaussianBlur(radius=self.radius)))
            return blur_imgs, masks
        else:
            return imgs, masks

class RandomBrightnessShift(object):
    """Randomly shifts the brightness of the image.
    """
    def __init__(
            self,
            factor: float=0.5,
            probability: float=0.5,
    ) -> None:
        """Initialization

        Args:
            factor (float, optional): Minimum factor the brightness values is multiplied with. Defaults to 0.5.
            probability(float, optional): probability that brightness shift is applied. Defaults to 0.5.
        """
        self.factor = factor
        self.probability = probability

    def __call__(self, imgs: list[Image.Image], masks: list[Image.Image]) -> tuple[list[Image.Image], list[Image.Image]]:
        """Applies random brightness shift

        Args:
            imgs (list[Image.Image]): List of images which should be shifted in brightness.
            masks (list[Image.Image]): List of corresponding masks.

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: Lists of brightness shifted images and unchanged masks.
        """
        if random.random() < self.probability:
            bright_imgs = []
            # randomly set factor to multiply the brightness value with
            fact = self.factor + random.random()
            # iterate through images
            for img in imgs:
                # convert from RGB to HSV space to make brightness value accessible
                img = img.convert("HSV")
                # separate channels
                h, s, v = img.split()
                # multiply v values with factor
                v = v.point(lambda i: i*fact)
                # limit values to a maximum of 255
                v = v.point(lambda j: (j:=255) if j > 255 else j)
                # but image back together
                img = Image.merge("HSV", (h, s, v))
                bright_imgs.append(img.convert("RGB"))
            return bright_imgs, masks
        else:
            return imgs, masks

class RandomRectDropout(object):
    def __init__(
        self,
        drop_rect: bool = True,
        min_region_dropped: float = 0.10,
        max_edge_ratio: float = 3.0,
        min_edge_length: float = 0.05,
        max_edge_length: float = 0.3,
    ) -> None:
        """Initialisation

        Args:
            drop_rect (bool, optional): Wether random rectangles should be dropped.
            min_region_dropped (float, optional): Percentage of the image to be dropped. Generation of rectangles will be stopped once above this value. Defaults to 0.2.
            max_edge_ratio (float, optional): Maximum ratio between the long and the short edge. Defaults to 3.0.
            min_edge_length (float, optional): Minimum rectangle edge length in percent from the short image edge. Defaults to 0.05.
            max_edge_length (float, optional): Maximum rectangle edge length in percent from the short image edge. Defaults to 0.4.
        """
        self.drop_rect = drop_rect
        self.min_region_dropped = min_region_dropped
        self.max_edge_ratio = max_edge_ratio
        self.min_edge_length = min_edge_length
        self.max_edge_length = max_edge_length

    def _get_rect(self, img_shape: torch.Size) -> dict:
        """Generates a rectangle for the given image size.

        Args:
            img_shape (torch.Size): Height and width of the input image.

        Returns:
            dict: Dictionary including the upper left point and shape of the rectangle.
        """
        # generate the ratio between long and short edge
        ratio = random.uniform(1, self.max_edge_ratio)
        # decide if the rectangle should be oriented horizontally or vertically
        orientation = random.randint(0, 1)  # 0: horizontal, 1: vertical
        # generate the size of the rectangle based on the short edge of the image
        short_edge_pxl_img = min(img_shape)
        short_edge_pxl = random.randint(
            int(self.min_edge_length * short_edge_pxl_img),
            int(self.max_edge_length * short_edge_pxl_img / ratio),
        )
        long_edge_pxl = int(short_edge_pxl * ratio)
        size_x = long_edge_pxl if orientation == 0 else short_edge_pxl
        size_y = short_edge_pxl if orientation == 0 else long_edge_pxl
        # generate the position of the rectangle
        pos_x = random.randint(0, img_shape[1] - size_x)
        pos_y = random.randint(0, img_shape[0] - size_y)
        # return values as a dict
        return {"pos_x": pos_x, "pos_y": pos_y, "size_x": size_x, "size_y": size_y}

    def __call__(
        self, imgs: list[torch.Tensor], masks: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generates masks of rectangles to be dropped up to the given percentage

        Args:
            imgs (list[torch.Tensor]): A list of images for which dropout masks should be generated
            masks (list[torch.Tensor]): Empty masks which will be filled with regions to be dropped

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: The input images and the generated masks.
        """
        cutout_masks = []
        if self.drop_rect:
            for image, mask in zip(imgs, masks):
                # variable for keeping track of the dropped percentage
                region_dropped = 0.0
                # loop until the desired drop percentage is reached
                while region_dropped < self.min_region_dropped:
                    # generate rectangle
                    rect = self._get_rect(image.shape[1:])
                    # add rectangle to mask
                    mask[
                        :,
                        rect["pos_y"] : (rect["pos_y"] + rect["size_y"]),
                        rect["pos_x"] : (rect["pos_x"] + rect["size_x"]),
                    ] = 1.0
                    # calculate dropped region
                    region_dropped += rect["size_x"] / image.shape[2] * rect["size_y"] / image.shape[1]
                cutout_masks.append(mask)
            return imgs, cutout_masks
        else:
            return imgs, masks

class Shuffle(object):
    def __init__(self, shuffle: bool =True) -> None:
        """Shuffles the images

        Args:
            shuffle (bool): Wether images should be shuffled. Defaults to true.
        """
        self.shuffle = shuffle

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Shuffles images and masks

        Args:
            imgs (list[Image.Image]): list of images
            masks (list[Image.Image]): list of (groundtruth) masks

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: Shuffled images together with the masks
        """
        if self.shuffle:
            img_mask = list(zip(imgs, masks))
            random.shuffle(img_mask)
            shuffled_imgs, shuffled_masks = zip(*img_mask)
            return shuffled_imgs, shuffled_masks
        else:
            return imgs, masks

class RandomImageDrop(object):
    def __init__(self, drop: bool = True) -> None:
        """Randomly drops one of the input images

        Args:
            drop (bool): Wether an image should be dropped. Defaults to true.
        """
        self.drop = drop

    def __call__(
        self, imgs: list[Image.Image], masks: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Shuffles images and masks

        Args:
            imgs (list[Image.Image]): list of images
            masks (list[Image.Image]): list of masks which should be filled with regions to be dropped

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: Original images and drop masks
        """
        if self.drop:
            drop_idx = random.randint(0, len(imgs)-1)
            masks[drop_idx][...] = 1.0
        return imgs, masks

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    im_list = [torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256)]
    mask_list = [torch.zeros(3, 256, 256), torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)]
    rect = RandomRectDropout()
    i, m = rect(im_list, mask_list)
    img_mask_list = i + m

    # Create a new figure and add the images in the upper row and the masks in the lower row
    figure = plt.figure()
    for i in range(len(img_mask_list)):
        figure.add_subplot(2, len(img_mask_list) // 2, i + 1)
        plt.imshow(img_mask_list[i].permute(1, 2, 0))
        plt.axis("off")
    plt.show()