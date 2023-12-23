import cv2
import torch
from torch.utils.data import DataLoader

import wandb
from config.config import device, logger, args
from scripts.transforms import RandomRectDropout
from scripts.utils import EarlyStopping

scaler = torch.cuda.amp.GradScaler()

def train_step_masked(dataloader: DataLoader, model: any, loss_fn: any, optimizer: any, amp = True) -> float:
    """Function which trains the given model with all the data from the given dataloader.

    Args:
        dataloader (dataloader): Dataloader
        model (pytorch model): Model to be trained.
        loss_fn (loss function): Function to calculate the training loss.
        optimizer (optimizer): Optimizer for backpropagation.

    Returns:
        float: average loss
    """
    # set model in train mode
    model.train()
    num_batches = len(dataloader)
    train_loss = 0

    #logger.warning("Loss calculation is configured for TransFuse!")

    # iterate through the dataloader
    for img, mask, _ in dataloader:
        img = img.to(device)

        ## #Input images different from #output images
        # if "output_frames" is negative, take the last -output_frames images
        # if "output_frames" is positive, only take the frame with that specific index
        if args["output_frames"] < 0:
            mask = mask[:, args["output_frames"]:, :, :, :]
        else:
            mask = mask[:, args["output_frames"], :, :, :]
            mask = mask.unsqueeze(1)

        mask = mask.to(device)

        if amp:
            with torch.autocast(device_type=device.type):
                pred = model(img)
                if type(pred) == list:
                    losses = [loss_fn(p, mask) for p in pred]
                    loss = 0
                    assert len(losses) == len(args["loss_factors"]), f"Number of calculated losses ({len(losses)}) and number of loss factors ({len(args['loss_factors'])}) are not equal!"
                    for factor, l in zip(args["loss_factors"], losses):
                        loss += factor*l
                else:
                    loss = loss_fn(pred, mask)
                train_loss += loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            pred = model(img)
            if type(pred) == list:
                losses = [loss_fn(p, mask) for p in pred]
                loss = sum(losses)
            else:
                loss = loss_fn(pred, mask)
            train_loss += loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


    return train_loss.item() / num_batches


def validation_step_masked(dataloader: DataLoader, model: any, loss_fn: any, amp = True) -> float:
    """Evaluate the model with the data from the dataloader

    Args:
        dataloader (DataLoader): Dataloader
        model (pytorch model): model to evaluate
        loss_fn (loss function): Function to calculate the validation loss.

    Returns:
        float: average loss
    """
    # Set the model into evaluation mode
    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0

    # iterate through the dataset and acumulate the loss
    with torch.no_grad():
        for img, mask, _ in dataloader:
            img = img.to(device)

            # #input images different from # output images
            if args["output_frames"] < 0:
                mask = mask[:, args["output_frames"]:, :, :, :]
            else:
                mask = mask[:, args["output_frames"], :, :, :]
                mask = mask.unsqueeze(1)
            mask = mask.to(device)

            if amp:
                with torch.autocast(device_type=device.type):
                    pred = model(img)
                    if type(pred) == list:
                        pred = pred[-1]
                    loss = loss_fn(pred, mask)
                    validation_loss += loss

            else:
                pred = model(img)
                if type(pred) == list:
                    pred = pred[-1]
                loss = loss_fn(pred, mask)
                validation_loss += loss


    return validation_loss.item() / num_batches


def train_step_inpainting(
    dataloader: DataLoader, model: any, loss_fn: any, optimizer: any, amp = True
) -> float:
    """Training step for pretraining with inpainting.

    Args:
        dataloader (DataLoader): Dataloader
        model (any): Model to be trained.
        loss_fn (any): Loss function. Will be called with three arguments: loss_fn(pred, target, mask)
        optimizer (any): Optimizter.

    Returns:
        float: average loss
    """
    # set model in train mode
    model.train()
    num_batches = len(dataloader)
    train_loss = 0
    # iterate through the dataloader
    for img, mask in dataloader:
        img = img.to(device)
        # expand mask in channel dimension
        if len(mask.shape) == 5:
            mask = mask.expand(-1, -1, 3, -1, -1)
        else:
            mask = mask.expand(-1, 3, -1, -1)
        mask = mask.to(device)
        # generate image with cutout regions
        img_cut = torch.sub(img, torch.mul(img, mask))
        img_cut = img_cut.to(device)
        # generate output and calculate the loss

        if amp:
            with torch.autocast(device_type=device.type):
                pred = model(img_cut)
                loss = loss_fn(pred, img, mask)
                train_loss += loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            pred = model(img_cut)
            loss = loss_fn(pred, img, mask)
            train_loss += loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return train_loss.item() / num_batches


def validation_step_inpainting(dataloader: DataLoader, model: any, loss_fn: any, amp = True) -> float:
    """_summary_

    Args:
        dataloader (DataLoader): Dataloader
        model (any): Model to evaluate
        loss_fn (any): Function for calculating the loss

    Returns:
        float: average loss
    """
    # Set the model into evaluation mode
    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0

    # iterate through the dataset and accumulate the loss
    with torch.no_grad():
        for img, mask in dataloader:
            img = img.to(device)
            # expand mask in channel dimension
            if len(mask.shape) == 5:
                mask = mask.expand(-1, -1, 3, -1, -1)
            else:
                mask = mask.expand(-1, 3, -1, -1)
            mask = mask.to(device)
            # generate image with cutouts
            img_cut = torch.sub(img, torch.mul(img, mask))
            img_cut = img_cut.to(device)
            # generate detection and calculate loss
            if amp:
                with torch.autocast(device_type=device.type):
                    pred = model(img_cut)
                    loss = loss_fn(pred, img, mask)
                    validation_loss += loss

            else:
                pred = model(img_cut)
                loss = loss_fn(pred, img, mask)
                validation_loss += loss

    return validation_loss.item() / num_batches


def train_loop(
    train_dataloader: DataLoader, validation_dataloader: DataLoader, model: any, loss_fn: any, optimizer: any, epochs: int, stopper: EarlyStopping = None, amp: bool = True, scheduler: any = None) -> None:
    """Performs training steps for the given number of epochs

    Args:
        train_dataloader (DataLoader): Dataloader for the train set
        validation_dataloader (DataLoader): Dataloader for the validation set
        model (any): Model to be trained
        loss_fn (any): Loss function
        optimizer (any): Optimizer
        epochs (int): Maximum number of epochs
        stopper (EarlyStopping, optional): EarlyStopper to prevent overfitting. Defaults to None.
        amp (bool, optional): If true, automatic mixed precision is used. Defaults to True.
        scheduler (any, optional): Learning rate scheduler. Defaults to None.
    """
    logger.info(f"Training with {len(train_dataloader.dataset)} images and validation with {len(validation_dataloader.dataset)} images.")
    # for each epoch perform a train and a validation step
    for e in range(epochs):
        train_loss = train_step_masked(train_dataloader, model, loss_fn, optimizer, amp = amp)
        validation_loss = validation_step_masked(validation_dataloader, model, loss_fn, amp = amp)
        logger.info(
            f"Train epoch {e}: train_loss: {train_loss}, validation_loss: {validation_loss}"
        )
        wandb.log({"train_loss": train_loss, "validation_loss": validation_loss, "train_epoch": e})
        if scheduler:
            scheduler.step()
        # check if the training should be stopped
        if stopper is not None:
            stopper(validation_loss, model)
            if stopper.early_stop:
                break


def pre_train_loop(
    train_dataloader: DataLoader, validation_dataloader: DataLoader, model: any, loss_fn: any, optimizer: any, epochs: int, stopper: EarlyStopping =None, amp: bool = True, scheduler = None) -> None:
    """Performs pre training steps for the given number of epochs

    Args:
        train_dataloader (DataLoader): Dataloader for the train set
        validation_dataloader (DataLoader): Dataloader for the validation set
        model (any): Model to be trained
        loss_fn (any): Loss function
        optimizer (any): Optimizer
        epochs (int): Maximum number of epochs
        stopper (EarlyStopping, optional): EarlyStopper to prevent overfitting. Defaults to None.
        amp (bool, optional): If true, automatic mixed precision is used. Defaults to True.
        scheduler (any, optional): Learning rate scheduler. Defaults to None.
    """
    logger.info(f"Pretraining with {len(train_dataloader.dataset)} images and validation with {len(validation_dataloader.dataset)} images.")
    for e in range(epochs):
        # for each epoch perform a train and a validation step
        train_loss = train_step_inpainting(train_dataloader, model, loss_fn, optimizer, amp = amp)
        validation_loss = validation_step_inpainting(validation_dataloader, model, loss_fn, amp = amp)
        logger.info(
            f"Pretrain epoch {e}: train_loss: {train_loss}, validation_loss: {validation_loss}"
        )
        wandb.log({"pretrain_loss": train_loss, "pt_validation_loss": validation_loss, "pt_epoch": e})
        # check if the training should be stopped
        if scheduler:
            scheduler.step()
        if stopper is not None:
            stopper(validation_loss, model)
            if stopper.early_stop:
                break
