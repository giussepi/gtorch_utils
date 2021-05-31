# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/neg_pred_val """

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


def npv(input_, target, per_channel=False):
    """
    Calculates and returns the average negative predictive value of the provided masks

    NPV = \frac{TN}{TN+FN}

    Args:
        inputs <torch.Tensor>: predicted masks [batch_size, channels, ...]
        target <torch.Tensor>: ground truth masks [batch_size, channels, ...]
        per_channel    <bool>: Whether or not return values per channel

    Returns:
        avg_npv <torch.Tensor>
    """
    assert isinstance(input_, torch.Tensor), type(input_)
    assert isinstance(target, torch.Tensor), type(target)
    assert isinstance(per_channel, bool), type(per_channel)

    # TODO: find an efficient way of implementing the special case when
    # both the prediction and the ground truth are white

    mgr = ConfusionMatrixMGR(input_, target)
    tn = mgr.true_negatives

    if per_channel:
        result = tn / (tn + mgr.false_negatives + EPSILON)

        return result.sum(0) / input_.size(0)

    tn = tn.sum(1)
    result = tn / (tn + mgr.false_negatives.sum(1) + EPSILON)

    return result.sum() / input_.size(0)
