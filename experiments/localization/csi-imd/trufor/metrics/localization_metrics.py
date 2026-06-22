from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np


def compute_iou(pred_mask, gt_mask):
    """
    IoU = intersection / union
    """

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0

    return intersection / union


def compute_accuracy(pred_mask, gt_mask):
    """
    Standard pixel accuracy.
    """

    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size

    return correct / total


def compute_precision(pred_mask, gt_mask):

    return precision_score(
        gt_mask.flatten(),
        pred_mask.flatten(),
        zero_division=0
    )


def compute_recall(pred_mask, gt_mask):

    return recall_score(
        gt_mask.flatten(),
        pred_mask.flatten(),
        zero_division=0
    )


def compute_f1(pred_mask, gt_mask):

    return f1_score(
        gt_mask.flatten(),
        pred_mask.flatten(),
        zero_division=0
    )