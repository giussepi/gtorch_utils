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

    def __init__(self, *, with_logits=False, per_class=False):
        """
        Initializes the object instance

        with_logits <bool>: Set to True when working with logits to apply sigmoid
        per_class <bool>: Set it to True to calculate specificity values per class
        """
        super().__init__()
        assert isinstance(with_logits, bool), type(with_logits)
        assert isinstance(per_class, bool), type(per_class)

        self.with_logits = with_logits
        self.per_class = per_class

    @staticmethod
    def _calculate_specificity(tn: torch.Tensor, fp: torch.Tensor):
        """
        Calculates and returns the specificity

        Kwargs:
            tn <torch.Tensor>: tensor with true negative values
            fp <torch.Tensor>: tensor with false positive values

        returns:
            recall <torch.Tensor>
        """
        assert isinstance(tn, torch.Tensor), type(tn)
        assert isinstance(fp, torch.Tensor), type(fp)

        return tn / (tn + fp + EPSILON)

    def forward(self, preds, targets):
        """
        Calculates and returns the average specificity (true negative rate) of the
        provided masks

        specificity = \frac{TN}{TN + FP}

        kwargs:
            preds   <torch.Tensor>: predicted masks [batch_size, classes, ...]. It will be
                                    reshaped to [batch_size, classes, -1]
            targets <torch.Tensor>: ground truth masks [batch_size, classes, ...]. It will be
                                    reshaped to [batch_size, classes, -1]

        Returns:
            specificity <torch.Tensor>
        """
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(targets, torch.Tensor), type(targets)

        if self.with_logits:
            preds = torch.sigmoid(preds)

        mgr = ConfusionMatrixMGR(preds, targets)
        tn = mgr.true_negatives

        if self.per_class:
            return self._calculate_specificity(tn.sum(0), mgr.false_positives.sum(0))

        result = self._calculate_specificity(tn.sum(1), mgr.false_positives.sum(1))

        return result.sum() / preds.size(0)
