# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/specificity """

from typing import Callable

import torch
from torchmetrics import Metric

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


__all__ = ['Specificity']


class Specificity(Metric):
    r"""
    TorchMetric based specificity with logits support

    Calculates and returns the average specificity of the provided masks or the specificity per class

    SPECIFICITY = \frac{TN}{TN + FP}

    Usage:
        Specificity()(preds, target)
    """

    def __init__(
            self, *, per_class: bool = False, with_logits: bool = False,
            logits_transform: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            **kwargs
    ):
        """
        Initializes the object instance

        per_class   <bool>: Set it to True to calculate specificity values per class
        with_logits <bool>: Set to True when working with logits to apply logits_transform
        logits_transform <callable>: function to be applied to the logits in preds. Default torch.sigmoid
        """
        super().__init__(**kwargs)

        assert isinstance(per_class, bool), type(per_class)
        assert isinstance(with_logits, bool), type(with_logits)
        assert callable(logits_transform), f'{logits_transform} must be a callable'

        self.per_class = per_class
        self.with_logits = with_logits
        self.logits_transform = logits_transform
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")

        if not self.per_class:
            self.add_state('batch_size', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates the state given the inputs

        Kwargs:
            preds  <torch.Tensor>: predicted masks [batch_size, classes, ...]
            target <torch.Tensor>: ground truth masks [batch_size, classes, ...]
        """
        assert preds.shape == target.shape
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(target, torch.Tensor), type(target)

        if self.with_logits:
            preds = self.logits_transform(preds)

        mgr = ConfusionMatrixMGR(preds, target)
        tn = mgr.true_negatives
        fp = mgr.false_positives

        if self.tn.shape != tn.shape:
            self.tn = torch.zeros_like(tn, device=preds.device)

        if self.fp.shape != fp.shape:
            self.fp = torch.zeros_like(fp, device=preds.device)

        self.tn += tn
        self.fp += fp

        if not self.per_class:
            self.batch_size += torch.tensor(preds.size(0))

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

    def compute(self):
        """
        Computes and returns recall

        Returns:
            recall <torch.Tensor>
        """
        if self.per_class:
            return self._calculate_specificity(self.tn.sum(0), self.fp.sum(0))

        result = self._calculate_specificity(self.tn.sum(1), self.fp.sum(1))

        return result.sum() / self.batch_size
