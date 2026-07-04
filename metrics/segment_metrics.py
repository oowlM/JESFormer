import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score


def tensor_detach(img):
    if isinstance(img, torch.Tensor) and len(img.shape) == 4:
        img = img.squeeze().detach().cpu().numpy()
    return img.flatten().astype(np.int64)


def calculate_confusion_matrix(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f'Image shapes are differnet: {pred.shape}, {target.shape}.')
    pred, target = tensor_detach(pred), tensor_detach(target)
    confusion = confusion_matrix(target, pred)
    accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1]) if float(confusion[0, 0] + confusion[0, 1]) != 0 else 0
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0]) if float(confusion[1, 1] + confusion[1, 0]) != 0 else 0
    precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1]) if float(confusion[1, 1] + confusion[0, 1]) != 0 else 0
    return accuracy, specificity, sensitivity, precision


def calculate_accuracy(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f'Image shapes are differnet: {pred.shape}, {target.shape}.')
    pred, target = tensor_detach(pred), tensor_detach(target)
    return accuracy_score(target, pred)


def calculate_precision(pred, target, label=255):
    if pred.shape != target.shape:
        raise ValueError(f'Image shapes are differnet: {pred.shape}, {target.shape}.')
    pred, target = tensor_detach(pred), tensor_detach(target)
    return precision_score(target, pred, pos_label=label, average='binary', zero_division=0)


def calculate_recall(pred, target, label=255):
    if pred.shape != target.shape:
        raise ValueError(f'Image shapes are differnet: {pred.shape}, {target.shape}.')
    pred, target = tensor_detach(pred), tensor_detach(target)
    return recall_score(target, pred, pos_label=label)


def calculate_auc_roc(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f'Image shapes are differnet: {pred.shape}, {target.shape}.')
    pred, target = tensor_detach(pred), tensor_detach(target)
    return roc_auc_score(target, pred)


def calculate_dice(pred, target, label=255, eps=1e-8):
    if pred.shape != target.shape:
        raise ValueError(f'Image shapes are differnet: {pred.shape}, {target.shape}.')
    pred, target = tensor_detach(pred), tensor_detach(target)
    pred = pred == label
    target = target == label
    intersection = np.logical_and(pred, target).sum(dtype=np.float64)
    denominator = pred.sum(dtype=np.float64) + target.sum(dtype=np.float64)
    return float((2.0 * intersection + eps) / (denominator + eps))


def calculate_miou(pred, target, label=255, eps=1e-8):
    if pred.shape != target.shape:
        raise ValueError(f'Image shapes are differnet: {pred.shape}, {target.shape}.')
    pred, target = tensor_detach(pred), tensor_detach(target)
    pred = pred == label
    target = target == label
    intersection = np.logical_and(pred, target).sum(dtype=np.float64)
    union = np.logical_or(pred, target).sum(dtype=np.float64)
    return float((intersection + eps) / (union + eps))
