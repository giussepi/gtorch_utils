# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/fp_pred_val """

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


def fppv(input_: torch.Tensor, targets: torch.Tensor, per_channel: bool = False) -> torch.Tensor:
    """
    Calculates and returns the average False Positive Predictive Value (FPPV) of the
    provided masks

    FPPV = \frac{FP}{FP+TP}

    Args:
        input_   <torch.Tensor>: predicted masks [batch_size, channels, ...]
        targets  <torch.Tensor>: ground truth masks [batch_size, channels, ...]
        per_channel      <bool>: Whether or not return recall values per channel

    Returns:
        avg_fppv <torch.Tensor>
    """
    assert isinstance(input_, torch.Tensor), type(input_)
    assert isinstance(targets, torch.Tensor), type(targets)
    assert isinstance(per_channel, bool), type(per_channel)

    mgr = ConfusionMatrixMGR(input_, targets)
    fp = mgr.false_positives

    if per_channel:
        result = fp / (fp + mgr.true_positives + EPSILON)

        return result.sum(0) / input_.size(0)

    fp = fp.sum(1)
    result = fp / (fp + mgr.true_positives.sum(1) + EPSILON)

    return result.sum() / input_.size(0)
