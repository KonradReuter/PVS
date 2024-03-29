import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb
from config.config import device, logger, ATT_MAP_DIR, PRED_DIR
from scripts.transforms import *
from medpy.metric.binary import hd95
from math import isnan
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
from medcam.medcam_utils import save_attention_map
import pandas as pd

class Evaluator(object):
    def __init__(
        self,
        accuracy: bool = True,
        precision: bool = True,
        recall: bool = True,
        specificity: bool = False,
        IOU: bool = False,
        dice: bool = False,
        F1: bool = False,
        F2: bool = False,
        MAE: bool = False,
        hd95: bool = True,
        threshold: float = None,
    ) -> None:
        """Evaluator class, which can calculate different metrics

        Args:
            accuracy (bool, optional): Calculate accuracy. Defaults to True.
            precision (bool, optional): Calculate precision. Defaults to True.
            recall (bool, optional): Calculate recall. Defaults to True.
            specificity (bool, optional): Calculate specificity. Defaults to False.
            IOU (bool, optional): Calculate intersection over union. Defaults to False.
            dice (bool, optional): Calculate dice score. Defaults to False.
            F1 (bool, optional): Calculate F1 score. Defaults to False.
            F2 (bool, optional): Calculate F2 score. Defaults to False.
            MAE (bool, optional): Calculate mean average error. Defaults to False.
        """
        self.accuracy = accuracy
        self.accuracy_list = []
        self.precision = precision
        self.precision_list = []
        self.recall = recall
        self.recall_list = []
        self.specificity = specificity
        self.specificity_list = []
        self.IOU = IOU
        self.IOU_list = []
        self.dice = dice
        self.dice_list = []
        self.F1 = F1
        self.F1_list = []
        self.F2 = F2
        self.F2_list = []
        self.MAE = MAE
        self.MAE_list = []
        self.hd95 = hd95
        self.hd95_list = []
        self.zero_pred_counter = 0
        self.threshold = threshold
        self.paths_list = []

    def update_per_img(self, prediction: torch.Tensor, target: torch.Tensor, image_paths: list) -> None:
        assert prediction.shape[1] == target.shape[1] == len(image_paths)
        for b in range(len(prediction)):
            for f in range(prediction.shape[1]):
                pred = prediction[b, f, ...]
                tgt = target[b, f, ...]
                img_path = image_paths[f][b]
                self.update(pred, tgt, img_path)


    def update(self, prediction: torch.Tensor, target: torch.Tensor, image_path: str) -> None:
        """Update the metrics based on a new prediction

        Args:
            prediction (torch.Tensor): Prediction generated by the model
            target (torch.Tensor): Groundtruth
        """
        # Convert prediction into range [0, 1]
        prediction = torch.sigmoid(prediction)

        assert prediction.max() <= 1.
        assert prediction.min() >= 0.
        assert target.max() <= 1.
        assert target.min() >= 0.

        if image_path in self.paths_list and args["unique"]:
            idx = self.paths_list.index(image_path)
        else:
            idx = None
            self.paths_list.append(image_path)

        if self.threshold is not None:
            prediction = (prediction>self.threshold).float()

        prediction = prediction.to(torch.int)
        target = target.to(torch.int)

        if prediction.max() == 0:
            self.zero_pred_counter += 1

        # calculate true positive, false positive, false negative and true positive value
        TP = torch.mul(prediction, target).sum()
        FP = torch.mul((1 - target), prediction).sum()
        FN = torch.mul(target, (1 - prediction)).sum()
        TN = torch.mul((1 - target), (1 - prediction)).sum()

        # calculate the desired metrics and append to the lists
        if self.accuracy:
            acc = (TP + TN) / (TP + FP + FN + TN)
            if idx == None:
                self.accuracy_list.append(acc)
            else:
                self.accuracy_list[idx] = acc

        if self.precision:
            pre = (TP) / (TP + FP)
            if idx == None:
                self.precision_list.append(pre)
            else:
                self.precision_list[idx] = pre

        if self.recall:
            rec = (TP) / (TP + FN)
            if idx == None:
                self.recall_list.append(rec)
            else:
                self.recall_list[idx] = rec

        if self.specificity:
            spe = (TN) / (TN + FP)
            if idx == None:
                self.specificity_list.append(spe)
            else:
                self.specificity_list[idx] = spe

        if self.IOU:
            iou = (TP) / (TP + FP + FN)
            if idx == None:
                self.IOU_list.append(iou)
            else:
                self.IOU_list[idx] = iou

        if self.dice:
            dic = (2 * TP) / (2 * TP + FP + FN)
            if idx == None:
                self.dice_list.append(dic)
            else:
                self.dice_list[idx] = dic

        if self.F1:
            f1 = (TP) / (TP + 0.5 * (FP + FN))
            if idx == None:
                self.F1_list.append(f1)
            else:
                self.F1_list[idx] = f1

        if self.F2:
            f2 = (TP) / (TP + 0.2 * FP + 0.8 * FN)
            if idx == None:
                self.F2_list.append(f2)
            else:
                self.F2_list[idx] = f2

        if self.MAE:
            mae = (torch.abs(prediction - target)).sum() / len(prediction.reshape(-1))
            if idx == None:
                self.MAE_list.append(mae)
            else:
                self.MAE_list[idx] = mae

        if self.hd95:
            if prediction.max() == 0:
                hd = float('nan')
            else:
                hd = torch.tensor(hd95(prediction.detach().cpu().numpy(), target.detach().cpu().numpy()))
            if idx == None:
                self.hd95_list.append(hd)
            else:
                self.hd95_list[idx] = hd

    def getMetrics(self) -> dict:
        """Returns the metrics as a dictionary

        Returns:
            dict: Dictionary with all desired metrics.
        """
        metrics = {}
        if self.precision or self.hd95:
            metrics["zero predictions"] = self.zero_pred_counter
        if self.accuracy:
            metrics["accuracy"] = torch.mean(torch.stack(self.accuracy_list)).item()
            metrics["accuracy std"] = torch.std(torch.stack(self.accuracy_list)).item()
            #logger.info(f"accuracy list: {torch.stack(self.accuracy_list).cpu().detach().numpy()}")
        if self.precision:
            clean_precision_list = [i for i in self.precision_list if not isnan(i)]
            metrics["precision"] = torch.mean(torch.stack(clean_precision_list)).item()
            metrics["precision std"] = torch.std(torch.stack(clean_precision_list)).item()
            #logger.info(f"precision list: {torch.stack(self.precision_list).cpu().detach().numpy()}")
        if self.recall:
            metrics["recall"] = torch.mean(torch.stack(self.recall_list)).item()
            metrics["recall std"] = torch.std(torch.stack(self.recall_list)).item()
            #logger.info(f"recall list: {torch.stack(self.recall_list).cpu().detach().numpy()}")
        if self.specificity:
            metrics["specificity"] = torch.mean(torch.stack(self.specificity_list)).item()
            metrics["specificity std"] = torch.std(torch.stack(self.specificity_list)).item()
            #logger.info(f"specificity list: {torch.stack(self.specificity_list).cpu().detach().numpy()}")
        if self.IOU:
            metrics["IOU"] = torch.mean(torch.stack(self.IOU_list)).item()
            metrics["IOU std"] = torch.std(torch.stack(self.IOU_list)).item()
            #logger.info(f"IOU list: {torch.stack(self.IOU_list).cpu().detach().numpy()}")
        if self.dice:
            metrics["dice"] = torch.mean(torch.stack(self.dice_list)).item()
            metrics["dice std"] = torch.std(torch.stack(self.dice_list)).item()
            #logger.info(f"dice list: {torch.stack(self.dice_list).cpu().detach().numpy()}")
        if self.F1:
            metrics["F1"] = torch.mean(torch.stack(self.F1_list)).item()
            metrics["F1 std"] = torch.std(torch.stack(self.F1_list)).item()
            #logger.info(f"F1 list: {torch.stack(self.F1_list).cpu().detach().numpy()}")
        if self.F2:
            metrics["F2"] = torch.mean(torch.stack(self.F2_list)).item()
            metrics["F2 std"] = torch.std(torch.stack(self.F2_list)).item()
            #logger.info(f"F2 list: {torch.stack(self.F2_list).cpu().detach().numpy()}")
        if self.MAE:
            metrics["MAE"] = torch.mean(torch.stack(self.MAE_list)).item()
            metrics["MAE std"] = torch.std(torch.stack(self.MAE_list)).item()
            #logger.info(f"MAE list: {torch.stack(self.MAE_list).cpu().detach().numpy()}")
        if self.hd95:
            clean_hd95_list = [i for i in self.hd95_list if not isnan(i)]
            metrics["hd95"] = torch.mean(torch.stack(clean_hd95_list)).item()
            metrics["hd95 std"] = torch.std(torch.stack(clean_hd95_list)).item()
        return metrics


def _save_attention_maps(images, img_paths, attention, model_name, set_name, args):
    # images and attention maps come in form b x c x f x h x w => reshape
    images = images.permute(0, 2, 1, 3, 4)
    attention = attention.permute(0, 2, 1, 3, 4)

    # iterate through batch and frames
    for b in range(attention.shape[0]):
        for f in range(attention.shape[1]):
            # get current input image and "unnormalize"
            raw = images[b, f, ...]
            for channel in range(raw.shape[0]):
                raw[channel, ...] *= float(args["normalize_std"][channel])
                raw[channel, ...] += float(args["normalize_mean"][channel])
            # get current attention map
            curr_att = attention[b, f, ...]
            # get current image path and extract clip name and file name
            if args["anchor_frame"]:
                if f == 0:
                    img_path = img_paths[f][b].split(".")[0]+"_anchor.jpg"
                else:
                    img_path = img_paths[f-1][b]
            else:
                img_path = img_paths[f][b] if len(img_paths) > 1 else img_paths[0][b]

            img_path = "/".join(img_path.split("/")[-2:])
            # create save path
            if len(img_paths) == 1:
                save_path = Path(ATT_MAP_DIR, model_name, set_name, img_path.split(".")[0], str(f))
            else:
                save_path = Path(ATT_MAP_DIR, model_name, set_name, img_path.split(".")[0])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # save map
            save_attention_map(str(save_path), curr_att[0], heatmap=True, raw_input=raw)

def _save_predictions(pred, img_paths, model_name, set_name):
    # iterate through batch size and frames
    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            # get current predictions and apply threshold
            curr_pred = pred[b, f, ...]
            curr_pred = (curr_pred>0.5).float()
            # get current image paths and extract clip name and file name
            img_path = img_paths[f][b]
            img_path = "/".join(img_path.split("/")[-2:])
            # create save path
            save_path = Path(PRED_DIR, model_name, set_name, img_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # convert to pillow image and save
            img_pil = to_pil_image(curr_pred, mode="L")
            img_pil.save(save_path)


def test_loop(dataloader: DataLoader, model: any, loss_fn: any, args: dict, evaluator: Evaluator, model_name: str, set_name: str) -> None:
    """Evaluate the model with the data from the dataloader

    Args:
        dataloader (DataLoader): Dataloader
        model (pytorch model): model to evaluate
        loss_fn (loss function): Function to calculate the validation loss.
        args (dict): args for the current run
        evaluator (Evaluator): class for calculating evaluation metrics
    """
    # Set the model into evaluation mode
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # iterate through the dataset and accumulate the loss
    with torch.no_grad():
        for img, mask, img_paths in dataloader:
            img = img.to(device)

            # inputs different from # outputs
            if args["output_frames"] < 0:
                mask = mask[:, args["output_frames"]:, :, :, :]
                img_paths = img_paths[args["output_frames"]:]
            else:
                mask = mask[:, args["output_frames"], :, :, :]
                mask = mask.unsqueeze(1)
                img_paths = [img_paths[args["output_frames"]]]
            mask = mask.to(device)

            if args["save_attention_maps"]:
                # medcam expects input in shape b x c x f x h x w, therefore we need to reshape
                img = img.permute(0, 2, 1, 3, 4).contiguous()
                pred, attention = model(img)
                _save_attention_maps(img, img_paths, attention, model_name, set_name, args)
                if type(pred) == list:
                    pred = pred[-1]
                # reshape back to original form
                pred = pred.permute(0, 2, 1, 3, 4).contiguous()
            else:
                pred = model(img)
                if type(pred) == list:
                    pred = pred[-1]
            if args["save_predictions"]:
                _save_predictions(pred, img_paths, model_name, set_name)
            if type(evaluator) == list:
                for e in evaluator:
                    e.update_per_img(pred, mask, img_paths)
            else:
                evaluator.update_per_img(pred, mask, img_paths)
            loss = loss_fn(pred, mask)
            test_loss += loss.item()

    logger.info(f"Average test loss: {test_loss/num_batches}")
    if args["use_wandb"]:
        wandb.log({"Test loss": test_loss / num_batches})

if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor
    np.bool = bool
    tt = ToTensor()
    img1 = tt(Image.open('./data/CVC-ClinicDB/masks/1.png').convert('L'))
    img2 = tt(Image.open('./data/CVC-ClinicDB/masks/2.png').convert('L'))
    evaluator = Evaluator(
        accuracy=True,
        precision=True,
        recall=True,
        specificity=True,
        IOU=True,
        dice=True,
        F1=True,
        F2=True,
        MAE=True,
        hd95=True
    )
    evaluator.update(img1, img1, 'sample name')
    evaluator.update(img1, img2, 'img2')
    print(evaluator.getMetrics())
