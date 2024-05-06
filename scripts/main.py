from pathlib import Path
import copy
import numpy as np
import pandas as pd
import typer
from medcam import medcam
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from config.config import CHECKPOINT_DIR, LOGS_DIR, logger, args, device
from scripts.dataloader import get_dataloaders, get_datasets
from scripts.loss import L2ReconstructionLoss, get_loss
from scripts.test import test_loop, Evaluator
from scripts.train import pre_train_loop, train_loop
from scripts.utils import EarlyStopping, calculate_mean_std, make_reproducible, get_optimizer, adjust_args
from scripts.models.PVS.pvs import *
#from scripts.models.PNSPlus.PNSPlusNetwork import PNSPlusNet
#from scripts.models.PNSPlus.PNS_Network import PNSNet
from scripts.models.PraNet.PraNet_Res2Net import get_PraNet
from scripts.models.SANet.model import get_SANet
from scripts.models.CASCADE.networks import get_CASCADE
from scripts.models.SSTAN.vacs import VACSNet
#from scripts.models.Hybrid2d3d.network import get_HybridNet
from scripts.models.COSNet.siamese_model_conf import CoattentionNet
from scripts.models.TransFuse.TransFuse import get_TransFuse
from scripts.models.DeepLabV3.deeplab import get_DeepLab

app = typer.Typer(pretty_exceptions_show_locals=False)
make_reproducible(42)
loss = get_loss(args["loss"])(smooth=0.0)
optimizer_module = get_optimizer(args["optimizer"])
pt_loss = L2ReconstructionLoss(args)
np.float = float # np.float deprecated but is needed for medcam ...
np.bool = bool # np.bool deprecated but needed for calculation of hausdorff distance.

# Model dict so a model can be selected by passing its name to the train/evaluation functions
model_dict = {
    "Conv_base3": ConvNext_base3,
    "Conv_base": ConvNext_base,
    "Swin_base3": PolypSwin_base3,
    "Swin_base": PolypSwin_base,
    "Conv_simple": ConvNext_simple,
    "Conv_simple_skip": ConvNext_simple_skip,
    "Conv_simple_enc": ConvNext_simple_enc,
    "Conv_3D": ConvNext_3D,
    "Conv_3D_skip": ConvNext_3D_skip,
    "Conv_3D_enc": ConvNext_3D_enc,
    "Conv_LSTM": ConvNext_LSTM,
    "Conv_LSTM_skip": ConvNext_LSTM_skip,
    "Conv_LSTM_enc": ConvNext_LSTM_enc,
    "Conv_Attention": ConvNext_Attention,
    "Conv_Attention_skip": ConvNext_Attention_skip,
    "Conv_Attention_enc": ConvNext_Attention_enc,
    "Conv_NSA": ConvNext_NSA,
    "Conv_NSA_skip": ConvNext_NSA_skip,
    "Conv_NSA_enc": ConvNext_NSA_enc,
    "Conv_LSTM_single": ConvNext_LSTM_single,
    "DeepLab": get_DeepLab,
    "SANet": get_SANet,
    "TransFuse": get_TransFuse,
    "PraNet": get_PraNet,
    "CASCADE": get_CASCADE,
    "COSNet": CoattentionNet,
    #"HybridNet": get_HybridNet,
    #"PNSNet": PNSNet,
    #"PNSPlusNet": PNSPlusNet,
    "VACSNet": VACSNet,
    "Conv_LSTM_unpruned": ConvNext_LSTM_unpruned
}

# pretraining was not used in our experiments
@app.command()
def pretrain(model_name:str = "Conv_base3", file_name:str = "model.pt", run_name:str = "run"):
    """Pretrain a model

    Args:
        model_name (str, optional): Name of the model. Defaults to "Conv_base3".
        file_name (str, optional): File name for the weight file. Defaults to "model.pt".
        run_name (str, optional): Name of the WandB run. Defaults to "run".
    """
    # adjust args to specific model
    global args
    args = adjust_args(model_name, args)
    # initialize wandb run
    if args["use_wandb"]:
        run = wandb.init(project="pvs", config=args, name=run_name)
    # get dataloaders and model
    dataloaders = get_dataloaders(args)
    model = model_dict[model_name]()
    # adjust output layer for inpainting task
    model.out_layer = torch.nn.Conv2d(24, 3, 1)
    model.to(device)
    # define optimizer, scheduler and early stopper
    optimizer = optimizer_module(params=model.parameters(), lr=args["inpainting_learning_rate"], weight_decay=args["weight_decay"])
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args["scheduler_gamma"])
    pretrain_stopper = EarlyStopping(
        patience=args["patience"],
        verbose=True,
        path=Path(CHECKPOINT_DIR, file_name),
        trace_func=logger.info,
    )
    # run pretraining
    pre_train_loop(dataloaders["unmasked_vid_train"], dataloaders["unmasked_vid_valid"], model, pt_loss, optimizer, args, pretrain_stopper, scheduler=scheduler)


@app.command()
def train(model_name:str = "Conv_base3", file_name:str = "model.pt", run_name:str = "run", pretrained_file: str = None):
    """Train a model

    Args:
        model_name (str, optional): Model name. Defaults to "Conv_base3".
        file_name (str, optional): Weight file name. Defaults to "model.pt".
        run_name (str, optional): WandB run name. Defaults to "run".
        pretrained_file (str, optional): Name of the pretrained weight file, if there exists one. Defaults to None.
    """
    # Adjust args to specific model
    global args
    args = adjust_args(model_name, args)
    # initialize WandB
    if args["use_wandb"]:
        run = wandb.init(project="pvs", config=args, name=run_name)
    # get dataloader and model
    dataloaders = get_dataloaders(args)
    model = model_dict[model_name]()
    # if there exists a pretrained file, load it
    if pretrained_file is not None:
        model.out_layer = torch.nn.Conv2d(24, 3, 1)
        model.to(device)
        model.load_state_dict(torch.load(Path(CHECKPOINT_DIR, pretrained_file)))
        model.out_layer = torch.nn.Conv2d(24, 1, 1)
    model.to(device)
    # define optimizer, scheduler and early stopper
    optimizer = optimizer_module(params=model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args["scheduler_gamma"])
    train_stopper = EarlyStopping(
        patience=args["patience"],
        verbose=True,
        path=Path(CHECKPOINT_DIR, file_name),
        trace_func=logger.info,
    )
    # run training loop
    train_loop(dataloaders["masked_vid_train"], dataloaders["masked_vid_valid"], model, loss, optimizer, args, stopper=train_stopper, scheduler=scheduler)

# multi threshold evaluation was not used in our experiments
@app.command()
def multiThresholdEvaluation(model_name:str = "Conv_base3", file_name: str = "model.pt", test_set: str = "masked_vid_test_easy_seen"):
    """Evaluate a model with different thresholds

    Args:
        model_name (str, optional): Model name. Defaults to "Conv_base3".
        file_name (str, optional): Name of the weight file. Defaults to "model.pt".
        test_set (str, optional): Name of the test set that should be used for evaluation. Defaults to "masked_vid_test_easy_seen".
    """
    # adjust args to specific model
    global args
    args = adjust_args(model_name, args)
    # initialize wandb
    if args["use_wandb"]:
        run = wandb.init(project="pvs", config=args)
    # get dataloaders and model
    dataloaders = get_dataloaders(args)
    model = model_dict[model_name]().to(device)
    model.load_state_dict(torch.load(Path(CHECKPOINT_DIR, file_name)))

    # create a list of evaluators with different thresholds
    evaluator_list = []

    # threshold = None means that the model outputs are not converted into binary, but floating point values between 0.0 and 1.0 are used
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

    # run the test loop
    test_loop(dataloaders[test_set], model, loss, args, evaluator_list, model_name, test_set)

    # write results to csv
    results_list = []
    for e in evaluator_list:
        results = e.getMetrics()
        results["threshold"] = e.threshold
        results_list.append(results)

    df = pd.DataFrame(results_list)
    df.to_csv(Path(LOGS_DIR, "results.csv"), index=False)


@app.command()
def evaluate(model_name: str = "Conv_base3", file_name:str = "model.pt", run_name:str = "run"):
    """Evaluate a model on all SUN-SEG test sets

    Args:
        model_name (str, optional): Model Name. Defaults to "Conv_base3".
        file_name (str, optional): Name of the weight file. Defaults to "model.pt".
        run_name (str, optional): Name of the WandB run. Defaults to "run".
    """
    # Adjust args to specific model
    global args
    args = adjust_args(model_name, args)
    # initialize wandb
    if args["use_wandb"]:
        run = wandb.init(project="pvs", config=args, name = run_name)
    # get dataloaders and model
    dataloaders = get_dataloaders(args)
    model = model_dict[model_name]().to(device)
    model.load_state_dict(torch.load(Path(CHECKPOINT_DIR, file_name), map_location=device))
    # initialize medcam
    if args["save_attention_maps"]:
        model = medcam.inject(model, backend="gcam", return_attention=True)

    # define the evaluators
    evaluator_es = Evaluator(
        accuracy=False,
        precision=False,
        recall=True,
        specificity=False,
        IOU=True,
        dice=True,
        F1=False,
        F2=False,
        MAE=False,
        hd95=True,
        threshold=args["threshold"],
    )
    evaluator_eu = copy.deepcopy(evaluator_es)
    evaluator_hs = copy.deepcopy(evaluator_es)
    evaluator_hu = copy.deepcopy(evaluator_es)

    # run test loops
    test_loop(dataloaders["masked_vid_test_easy_seen"], model, loss, args, evaluator_es, model_name, "ES")
    test_loop(dataloaders["masked_vid_test_easy_unseen"], model, loss, args, evaluator_eu, model_name, "EU")
    test_loop(dataloaders["masked_vid_test_hard_seen"], model, loss, args, evaluator_hs, model_name, "HS")
    test_loop(dataloaders["masked_vid_test_hard_unseen"], model, loss, args, evaluator_hu, model_name, "HU")

    # get performance metrics
    performance_es = evaluator_es.getMetrics()
    performance_es["set"] = "easy seen"
    performance_eu = evaluator_eu.getMetrics()
    performance_eu["set"] = "easy unseen"
    performance_hs = evaluator_hs.getMetrics()
    performance_hs["set"] = "hard seen"
    performance_hu = evaluator_hu.getMetrics()
    performance_hu["set"] = "hard unseen"

    # log performance in wandb
    if args["use_wandb"]:
        wandb.log(performance_es)
        wandb.log(performance_eu)
        wandb.log(performance_hs)
        wandb.log(performance_hu)

    # log in console
    logger.info(f"Easy seen: {performance_es}")
    logger.info(f"Easy unseen: {performance_eu}")
    logger.info(f"Hard seen: {performance_hs}")
    logger.info(f"Hard unseen: {performance_hu}")

@app.command()
def get_mean_std_dev():
    """Calculate mean and standard deviation values over all datasets
    """
    mean, std = calculate_mean_std(get_datasets())
    print(f"Means over all images: {mean}")
    print(f"Standard deviation over all images: {std}")

if __name__ == "__main__":
    app()
