import numpy as np
import torch

def compute_confusion_matrix(preds, labels, num_classes, ignore_index=255):
    mask = (labels != ignore_index) & (labels < num_classes)
    preds = preds[mask]
    labels = labels[mask]

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, l in zip(preds, labels):
        if l < num_classes and p < num_classes:
            confusion_matrix[l, p] += 1
    return confusion_matrix


def compute_miou(preds, labels, num_classes, ignore_index=255):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu()

    confusion_matrix = compute_confusion_matrix(preds, labels, num_classes, ignore_index)

    intersection = torch.diag(confusion_matrix)
    ground_truth = confusion_matrix.sum(dim=1)
    predicted = confusion_matrix.sum(dim=0)
    union = ground_truth + predicted - intersection

    iou = intersection.float() / union.float().clamp(min=1)

    miou = iou.mean().item()

    return miou, iou.tolist()
