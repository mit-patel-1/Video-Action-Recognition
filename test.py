"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from train import video_loss_and_scores

def test(model, dataloader, device, return_loss=False):
    """
    Evaluate the model on the test dataset and compute overall accuracy.
    
    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
        return_loss (bool): Also return mean test loss when True.
    
    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        total_correct_preds = torch.zeros((), dtype=torch.long, device=device)
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        evaluated_samples = 0
        targets, outputs = [], []
        for batch in tqdm(dataloader):
            x_batch, y_batch = batch[:2]
            if x_batch is None:
                continue
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            lengths = batch[2] if len(batch) >= 3 else None
            video_indices = batch[3] if len(batch) == 4 else None
            output = model(x_batch, lengths) if lengths is not None else model(x_batch)
            loss, scores = video_loss_and_scores(output, y_batch, criterion, lengths, video_indices)
            total_loss += loss
            pred = scores.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum()
            total_correct_preds += correct_preds
            outputs.append(pred.view(-1).detach())
            targets.append(y_batch.view(-1).detach())
            evaluated_samples += y_batch.size(0)
        
        if evaluated_samples == 0:
            raise ValueError('No valid videos were available for testing.')
        accuracy = total_correct_preds.item() / float(evaluated_samples)
        outputs = torch.cat(outputs).cpu().tolist()
        targets = torch.cat(targets).cpu().tolist()
    
    if return_loss:
        return targets, outputs, accuracy, total_loss.item() / evaluated_samples
    return targets, outputs, accuracy

def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label: mat for label, mat in zip(all_cats, confusion_mat)}
