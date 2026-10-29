"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate from an optimizer.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
import json
from tqdm import tqdm
import torch

def train(dataloaders, model, criterion, optimizer, scheduler, device, optim_model_wts_dir,
          n_epochs=30, *, logger=None, checkpoint_info=None, start_epoch=0, best_val_acc=-1.0,
          freeze_epochs=0):
    """
    Train and validate the model over a given number of epochs.
    
    This function performs the training loop for a video classification model.
    It iterates through the specified number of epochs, updates model weights
    using backpropagation, and evaluates model performance on a validation set.
    The model with the best validation accuracy is saved to disk.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model to be trained.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler, which adjusts the learning rate based on validation loss.
        device (torch.device): Device (CPU or GPU) on which to perform training.
        optim_model_wts_dir (str): Directory to save the best model weights.
        n_epochs (int, optional): Number of training epochs. Default is 30.
        logger (callable, optional): Receives loss and accuracy after each epoch.
        checkpoint_info (dict, optional): Model settings saved with the weights.
        start_epoch (int, optional): Number of epochs already completed.
        best_val_acc (float, optional): Best accuracy from the loaded checkpoint.
        freeze_epochs (int, optional): Train only the LSTM and classifier first.

    Returns:
        tuple: (model, loss_hist, acc_hist)
            - model (torch.nn.Module): The trained model loaded with the best validation weights.
            - loss_hist (dict): Dictionary containing lists of training and validation losses for each epoch.
            - acc_hist (dict): Dictionary containing lists of training and validation accuracies for each epoch.
    """
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    os.makedirs(optim_model_wts_dir, exist_ok=True)

    for epoch in range(start_epoch, n_epochs):
        current_lr = get_learning_rate(optimizer)
        print('Epoch {}/{}; Current learning rate {}'.format(epoch+1, n_epochs, current_lr))

        # Training phase
        model.train()
        if freeze_epochs:
            for name, parameter in model.base_model.named_parameters():
                parameter.requires_grad = epoch >= freeze_epochs and name.startswith('layer4.')
            # Keep the CNN's BatchNorm statistics fixed in both stages.
            model.base_model.eval()
            if epoch == start_epoch or epoch == freeze_epochs:
                print('CNN frozen; training LSTM and classifier.' if epoch < freeze_epochs
                      else 'Fine-tuning the last CNN block, LSTM, and classifier.')
        train_loss, train_accuracy = get_epoch_loss(model, criterion, dataloaders['train'], device, optimizer)
        loss_hist['train'].append(train_loss)
        acc_hist['train'].append(train_accuracy)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(model, criterion, dataloaders['val'], device)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_name = 'best_model_wts.pt'
            best_model_path = os.path.join(optim_model_wts_dir, best_model_name)
            checkpoint = best_model_wts
            if checkpoint_info is not None:
                checkpoint = dict(checkpoint_info)
                checkpoint.update(model_state_dict=best_model_wts, epoch=epoch + 1,
                                  val_accuracy=val_accuracy)
            torch.save(checkpoint, best_model_path)
            print('Best model weights are updated at epoch {}!'.format(epoch+1))
        loss_hist['val'].append(val_loss)
        acc_hist['val'].append(val_accuracy)

        metrics = {'epoch': epoch + 1, 'train/loss': train_loss,
                   'train/accuracy': train_accuracy, 'val/loss': val_loss,
                   'val/accuracy': val_accuracy}
        with open(os.path.join(optim_model_wts_dir, 'metrics.jsonl'), 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics) + '\n')
        if logger is not None:
            logger(metrics)

        # Update learning rate based on validation loss
        plateau_schedule = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        if plateau_schedule:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print("train loss: {:.6f}, val loss: {:.6f}, accuracy: {:.2f}".format(train_loss, val_loss, 100*val_accuracy))
        print("-" * 60)
        print()

    # Load the best model weights before returning the model
    model.load_state_dict(best_model_wts)
    return model, loss_hist, acc_hist

def get_learning_rate(optimizer):
    """
    Retrieve the current learning rate from the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rate.
    
    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch_correct_preds(output, target, return_tensor=False):
    """
    Compute the number of correct predictions for a mini-batch.
    
    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).
    
    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum()
    return correct_preds if return_tensor else correct_preds.item()

def video_loss_and_scores(output, target, criterion, lengths=None, video_indices=None):
    """Mask padded steps and give every video equal weight across its clips."""
    if video_indices is None and output.ndim == 2:
        return criterion(output, target), output
    if video_indices is None:
        video_indices = torch.arange(len(target), device=output.device)
    else:
        video_indices = video_indices.to(output.device, non_blocking=True)
    clip_targets = target[video_indices]
    if output.ndim == 3:
        lengths = lengths.to(output.device, non_blocking=True)
        mask = torch.arange(output.shape[1], device=output.device)[None, :] < lengths[:, None]
        step_loss = torch.nn.functional.cross_entropy(
            output.transpose(1, 2), clip_targets[:, None].expand(-1, output.shape[1]),
            reduction='none')
        clip_loss = (step_loss * mask).sum(1) / lengths
        clip_scores = (output.softmax(-1) * mask[:, :, None]).sum(1) / lengths[:, None]
    else:
        clip_loss = torch.nn.functional.cross_entropy(output, clip_targets, reduction='none')
        clip_scores = output.softmax(-1)
    counts = torch.bincount(video_indices, minlength=len(target)).to(output.dtype)
    losses = output.new_zeros(len(target)).index_add(0, video_indices, clip_loss)
    scores = output.new_zeros(len(target), output.shape[-1]).index_add(0, video_indices, clip_scores)
    return (losses / counts).sum(), scores / counts[:, None]


def get_batch_loss(criterion, output, target, optimizer=None, return_tensors=False,
                   lengths=None, video_indices=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).
    
    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights. If None, no backpropagation is performed.
        lengths (Tensor, optional): Real frame counts for masking step predictions.
        video_indices (Tensor, optional): Video index for each clip in the batch.
    
    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss, scores = video_loss_and_scores(output, target, criterion, lengths, video_indices)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(scores, target, return_tensor=return_tensors)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (loss.detach() if return_tensors else loss.item()), n_batch_correct_preds

def get_epoch_loss(model, criterion, dataloader, device, optimizer=None):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to update model weights during training.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss = torch.zeros((), dtype=torch.float64, device=device)
    running_total_correct_preds = torch.zeros((), dtype=torch.long, device=device)
    evaluated_samples = 0

    for batch in tqdm(dataloader):
        x_batch, y_batch = batch[:2]
        if x_batch is None:
            continue
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        lengths = batch[2] if len(batch) >= 3 else None
        video_indices = batch[3] if len(batch) == 4 else None
        output = model(x_batch, lengths) if lengths is not None else model(x_batch)
        batch_loss, n_batch_correct_preds = get_batch_loss(
            criterion, output, y_batch, optimizer, return_tensors=True,
            lengths=lengths, video_indices=video_indices
        )

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds
        evaluated_samples += y_batch.size(0)

    if evaluated_samples == 0:
        raise ValueError('No valid videos were available for this epoch.')
    loss = running_loss.item() / float(evaluated_samples)
    accuracy = running_total_correct_preds.item() / float(evaluated_samples)
    return loss, accuracy
