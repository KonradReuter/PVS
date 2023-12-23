from pathlib import Path
import copy

import numpy as np
import pandas as pd
import typer
from medcam import medcam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR

import wandb
from config.config import PRED_DIR, CHECKPOINT_DIR, LOGS_DIR, logger, args, device
from scripts.dataloader import get_dataloaders, get_datasets
from scripts.loss import L2ReconstructionLoss, get_loss
from scripts.test import test_loop, Evaluator
from scripts.train import pre_train_loop, train_loop
from scripts.transforms import *
from scripts.utils import EarlyStopping, calculate_mean_std, make_reproducible, get_optimizer, SingleImageModelWrapper
from scripts.models.PVS.pvs import *
from scripts.models.PNSPlus.PNSPlusNetwork import PNSPlusNet
from scripts.models.PNSPlus.PNS_Network import PNSNet
from scripts.models.UNet.model import UNet
from torchvision.models import ResNet50_Weights
from scripts.models.PraNet.PraNet_Res2Net import get_PraNet
from scripts.models.SANet.model import get_SANet
from scripts.models.FCBFormer.models import get_FCBFormer
from scripts.models.CASCADE.networks import get_CASCADE
from scripts.models.SSTAN.vacs import VACSNet
from scripts.models.Hybrid2d3d.network import get_HybridNet
from scripts.models.COSNet.siamese_model_conf import CoattentionNet
from scripts.models.TransFuse.TransFuse import get_TransFuse
from scripts.models.DeepLabV3.deeplab import get_DeepLab

app = typer.Typer(pretty_exceptions_show_locals=False)
make_reproducible(42)
loss = get_loss(args["loss"])(smooth=0.0)
optimizer_module = get_optimizer(args["optimizer"])
pt_loss = L2ReconstructionLoss()
np.float = float # np.float deprecated but is needed for medcam ...
np.bool = bool # np.bool deprecated but needed for calculation of hausdorff distance.
get_model = ConvNext_base3

@app.command()
def pretrain(model_save_name:str = "model.pt", run_name:str = "run") -> None:
    model_save_name = model_save_name.split(".")[0]+"_pretrained.pt"
    run = wandb.init(project="pvs", config=args, name=run_name)
    dataloaders = get_dataloaders(args)
    model = get_model()
    model.out_layer = torch.nn.Conv2d(24, 3, 1)
    model.to(device)
    optimizer = optimizer_module(params=model.parameters(), lr=args["inpainting_learning_rate"], weight_decay=args["weight_decay"])
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args["scheduler_gamma"])

    pretrain_stopper = EarlyStopping(
        patience=args["patience"],
        verbose=True,
        path=Path(CHECKPOINT_DIR, model_save_name),
        trace_func=logger.info,
    )

    pre_train_loop(dataloaders["unmasked_vid_train"], dataloaders["unmasked_vid_valid"], model, pt_loss, optimizer, args["epochs"], pretrain_stopper, amp = args["amp"], scheduler=scheduler)

@app.command()
def train(model_save_name:str = "model.pt", run_name:str = "run", pretrained: bool = False):
    pt_model_name = model_save_name.split(".")[0]+"_pretrained.pt"
    model_save_name = model_save_name.split(".")[0]+str(args["validation_fold"])+".pt"
    run = wandb.init(project="pvs", config=args, name=run_name)
    dataloaders = get_dataloaders(args)
    model = get_model()
    if pretrained:
        model.out_layer = torch.nn.Conv2d(24, 3, 1)
        model.to(device)
        model.load_state_dict(torch.load(Path(CHECKPOINT_DIR, pt_model_name)))
        model.out_layer = torch.nn.Conv2d(24, 1, 1)
    model.to(device)
    optimizer = optimizer_module(params=model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args["scheduler_gamma"])
    train_stopper = EarlyStopping(
        patience=args["patience"],
        verbose=True,
        path=Path(CHECKPOINT_DIR, model_save_name),
        trace_func=logger.info,
    )
    train_loop(dataloaders["masked_vid_train"], dataloaders["masked_vid_valid"], model, loss, optimizer, args["epochs"], stopper=train_stopper, amp=args["amp"], scheduler=scheduler)


@app.command()
def multiThresholdEvaluation(model_name: str = "model.pt", test_set: str = "masked_vid_test_easy_seen"):
    run = wandb.init(project="pvs", config=args)
    dataloaders = get_dataloaders(args)
    model = PolypSwin(args["fusion_module"]).to(device)
    model.load_state_dict(torch.load(Path(CHECKPOINT_DIR, model_name)))

    evaluator_list = []

    evaluator_list.append(Evaluator(
            accuracy=True,
            precision=True,
            recall=True,
            specificity=True,
            IOU=True,
            dice=True,
            F1=True,
            F2=True,
            MAE=True,
            threshold=None))

    for t in np.arange(0.1, 0.9, 0.1):
        evaluator_list.append(Evaluator(
            accuracy=True,
            precision=True,
            recall=True,
            specificity=True,
            IOU=True,
            dice=True,
            F1=True,
            F2=True,
            MAE=True,
            threshold=t
    ))

    test_loop(dataloaders[test_set], model, loss, args, evaluator_list)

    results_list = []
    for e in evaluator_list:
        results = e.getMetrics()
        results["threshold"] = e.threshold
        results_list.append(results)

    df = pd.DataFrame(results_list)
    df.to_csv(Path(LOGS_DIR, "results.csv"), index=False)


@app.command()
def evaluate(model_name:str = "model.pt", run_name:str = "run"):
    model_name = model_name.split(".")[0]+str(args["validation_fold"])+".pt"
    run = wandb.init(project="pvs", config=args, name = run_name)
    dataloaders = get_dataloaders(args)
    model = get_model().to(device)
    model.load_state_dict(torch.load(Path(CHECKPOINT_DIR, model_name), map_location=device))
    if args["save_attention_maps"]:
        model = medcam.inject(model, backend="gcam", return_attention=True)

    evaluator_es = Evaluator(
        accuracy=False,
        precision=False,
        recall=False,
        specificity=False,
        IOU=False,
        dice=True,
        F1=False,
        F2=False,
        MAE=False,
        hd95=False,
        threshold=args["threshold"],
    )
    evaluator_eu = copy.deepcopy(evaluator_es)
    evaluator_hs = copy.deepcopy(evaluator_es)
    evaluator_hu = copy.deepcopy(evaluator_es)

    test_loop(dataloaders["masked_vid_test_easy_seen"], model, loss, args, evaluator_es, model_name.split(".")[0], "ES")
    test_loop(dataloaders["masked_vid_test_easy_unseen"], model, loss, args, evaluator_eu, model_name.split(".")[0], "EU")
    test_loop(dataloaders["masked_vid_test_hard_seen"], model, loss, args, evaluator_hs, model_name.split(".")[0], "HS")
    test_loop(dataloaders["masked_vid_test_hard_unseen"], model, loss, args, evaluator_hu, model_name.split(".")[0], "HU")

    performance_es = evaluator_es.getMetrics()
    performance_es["set"] = "easy seen"
    performance_eu = evaluator_eu.getMetrics()
    performance_eu["set"] = "easy unseen"
    performance_hs = evaluator_hs.getMetrics()
    performance_hs["set"] = "hard seen"
    performance_hu = evaluator_hu.getMetrics()
    performance_hu["set"] = "hard unseen"

    logger.info(f"Easy seen: {performance_es}")
    logger.info(f"Easy unseen: {performance_eu}")
    logger.info(f"Hard seen: {performance_hs}")
    logger.info(f"Hard unseen: {performance_hu}")

    wandb.log(performance_es)
    wandb.log(performance_eu)
    wandb.log(performance_hs)
    wandb.log(performance_hu)

    es_frame = evaluator_es.get_dice_as_df()
    eu_frame = evaluator_eu.get_dice_as_df()
    hs_frame = evaluator_hs.get_dice_as_df()
    hu_frame = evaluator_hu.get_dice_as_df()

    full_frame = pd.concat([es_frame, eu_frame, hs_frame, hu_frame], axis=0)
    full_frame.rename(columns={"Dice": model_name})
    full_frame.to_csv(Path(PRED_DIR, model_name+".csv"), index=False)

@app.command()
def get_mean_std_dev():
    mean, std = calculate_mean_std(get_datasets())
    print(f"Means over all images: {mean}")
    print(f"Standard deviation over all images: {std}")

if __name__ == "__main__":
    app()
