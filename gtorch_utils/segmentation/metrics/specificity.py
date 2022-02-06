# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/specificity """

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


__all__ = ['Specificity']


class Specificity(torch.nn.Module):
    """
    Module based Specificity with logits support

    Usage:
        Specificity()(predictions, ground_truth)
    """

    def __init__(self, *, with_logits=False, per_channel=False):
        """
        Initializes the object instance

        with_logits <bool>: Set to True when working with logits to apply sigmoid
        per_channel <bool>: Set it to True to calculate specificity values per channel
        """
        super().__init__()
        assert isinstance(with_logits, bool), type(with_logits)
        assert isinstance(per_channel, bool), type(per_channel)

        self.with_logits = with_logits
        self.per_channel = per_channel

    def forward(self, preds, targets):
        """
        Calculates and returns the average specificity (true negative rate) of the
        provided masks

        specificity = \frac{TN}{TN + FP}

        kwargs:
            preds   <torch.Tensor>: predicted masks [batch_size, channels, ...]. It will be
                                    reshaped to [batch_size, channels, -1]
            targets <torch.Tensor>: ground truth masks [batch_size, channels, ...]. It will be
                                    reshaped to [batch_size, channels, -1]

        Returns:
            specificity <torch.Tensor>
        """
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(targets, torch.Tensor), type(targets)

        if self.with_logits:
            preds = torch.sigmoid(preds)

        mgr = ConfusionMatrixMGR(preds, targets)
        tn = mgr.true_negatives

        if self.per_channel:
            result = tn / (tn + mgr.false_positives + EPSILON)

            return result.sum(0) / preds.size(0)

        tn = tn.sum(1)
        result = tn / (tn + mgr.false_positives.sum(1) + EPSILON)

        return result.sum() / preds.size(0)
