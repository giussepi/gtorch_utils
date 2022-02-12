# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/fp_rate """

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


def fpr(input_: torch.Tensor, targets: torch.Tensor, per_channel: bool = False) -> torch.Tensor:
    """
    Calculates and returns the average False Positive Rate (FPR) score of the
    provided masks

    FPR = \frac{FP}{TN+FP}

    Args:
        input_  <torch.Tensor>: predicted masks [batch_size, channels, ...]. It will be
                                reshaped to [batch_size, channels, -1]
        target  <torch.Tensor>: ground truth masks [batch_size, channels, ...]. It will be
                                reshaped to [batch_size, channels, -1]
        per_channel     <bool>: Whether or not return recall values per channel

    Returns:
        avg_fpr <torch.Tensor>
    """
    assert isinstance(input_, torch.Tensor), type(input_)
    assert isinstance(targets, torch.Tensor), type(targets)
    assert isinstance(per_channel, bool), type(per_channel)

    mgr = ConfusionMatrixMGR(input_, targets)
    fp = mgr.false_positives

    if per_channel:
        result = fp / (mgr.true_negatives + fp + EPSILON)

        return result.sum(0) / input_.size(0)

    fp = fp.sum(1)
    result = fp / (mgr.true_negatives.sum(1) + fp + EPSILON)

    return result.sum() / input_.size(0)
