# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/recall """

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


__all__ = ['recall']


def recall(input_: torch.Tensor, target: torch.Tensor, per_class: bool = False) -> torch.Tensor:
    """
    Calculates and returns the average recall of the provided masks

    RECALL = \frac{TP}{TP+FN}

    Args:
        input_ <torch.Tensor>: predicted masks [batch_size, channels, ...]
        target <torch.Tensor>: ground truth masks [batch_size, channels, ...]
        per_class    <bool>: Whether or not return recall values per class

    Returns:
        recall <torch.Tensor>
    """
    assert isinstance(input_, torch.Tensor), type(input_)
    assert isinstance(target, torch.Tensor), type(target)
    assert isinstance(per_class, bool), type(per_class)

    mgr = ConfusionMatrixMGR(input_, target)
    tp = mgr.true_positives

    if per_class:
        tp = tp.sum(0)

        return tp / (tp + mgr.false_negatives.sum(0) + EPSILON)

    tp = tp.sum(1)
    result = tp / (tp + mgr.false_negatives.sum(1) + EPSILON)

    return result.sum() / input_.size(0)
