# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/recall """

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


def recall(input_, target, per_channel=False):
    """
    Calculates and returns the average recall of the provided masks

    RECALL = \frac{TP}{TP+FN}

    Args:
        input_ <torch.Tensor>: predicted masks [batch_size, channels, ...]
        target <torch.Tensor>: ground truth masks [batch_size, channels, ...]
        per_channel    <bool>: Whether or not return recall values per channel

    Returns:
        avg_recall <torch.Tensor>
    """
    assert isinstance(input_, torch.Tensor), type(input_)
    assert isinstance(target, torch.Tensor), type(target)
    assert isinstance(per_channel, bool), type(per_channel)

    mgr = ConfusionMatrixMGR(input_, target)
    tp = mgr.true_positives

    if per_channel:
        result = tp / (tp + mgr.false_negatives + EPSILON)

        return result.sum(0) / input_.size(0)

    tp = tp.sum(1)
    result = tp / (tp + mgr.false_negatives.sum(1) + EPSILON)

    return result.sum() / input_.size(0)
