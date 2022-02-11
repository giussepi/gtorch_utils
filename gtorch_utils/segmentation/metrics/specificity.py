# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/specificity """

from typing import Callable

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

    def __init__(
            self, *, with_logits: bool = False, per_class: bool = False,
            logits_transform: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid
    ):
        """
        Initializes the object instance

        with_logits          <bool>: Set to True when working with logits to apply sigmoid
        per_class            <bool>: Set it to True to calculate specificity values per class
        logits_transform <callable>: Function to be applied to the logits in preds. Default torch.sigmoid
        """
        super().__init__()
        assert isinstance(with_logits, bool), type(with_logits)
        assert isinstance(per_class, bool), type(per_class)
        assert callable(logits_transform), f'{logits_transform} must be a callable'

        self.with_logits = with_logits
        self.per_class = per_class
        self.logits_transform = logits_transform

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

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
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
            preds = self.logits_transform(preds)

        mgr = ConfusionMatrixMGR(preds, targets)
        tn = mgr.true_negatives

        if self.per_class:
            return self._calculate_specificity(tn.sum(0), mgr.false_positives.sum(0))

        result = self._calculate_specificity(tn.sum(1), mgr.false_positives.sum(1))

        return result.sum() / preds.size(0)
