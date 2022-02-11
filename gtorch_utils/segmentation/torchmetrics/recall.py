# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/recall """

from typing import Callable

import torch
from torchmetrics import Metric

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


__all__ = ['Recall']


class Recall(Metric):
    r"""
    TorchMetric based recall with logits support

    Calculates and returns the average recall of the provided masks or the recall per class

    RECALL = \frac{TP}{TP+FN}

    Usage:
        Recall()(preds, target)
    """

    def __init__(
            self, *, per_class: bool = False, with_logits: bool = False,
            logits_transform: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            **kwargs
    ):
        """
        Initializes the object instance

        Kwargs:
            per_class            <bool>: Whether or not return recall values per class. Default False
            with_logits          <bool>: Set to True when working with logits to apply logits_transform
            logits_transform <callable>: function to be applied to the logits in preds. Default torch.sigmoid
        """
        super().__init__(**kwargs)

        assert isinstance(per_class, bool), type(per_class)
        assert isinstance(with_logits, bool), type(with_logits)
        assert callable(logits_transform), f'{logits_transform} must be a callable'
        self.with_logits = with_logits
        self.logits_transform = logits_transform
        self.per_class = per_class
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

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
        tp = mgr.true_positives
        fn = mgr.false_negatives

        if self.tp.shape != tp.shape:
            self.tp = torch.zeros_like(tp, device=preds.device)

        if self.fn.shape != fn.shape:
            self.fn = torch.zeros_like(fn, device=preds.device)

        self.tp += tp
        self.fn += fn

        if not self.per_class:
            self.batch_size += torch.tensor(preds.size(0))

    @staticmethod
    def _calculate_recall(tp: torch.Tensor, fn: torch.Tensor):
        """
        Calculates and returns the recall

        Kwargs:
            tp <torch.Tensor>: tensor with true positive values
            fn <torch.Tensor>: tensor with false negative values

        returns:
            recall <torch.Tensor>
        """
        assert isinstance(tp, torch.Tensor), type(tp)
        assert isinstance(fn, torch.Tensor), type(fn)

        return tp / (tp + fn + EPSILON)

    def compute(self):
        """
        Computes and returns recall

        Returns:
            recall <torch.Tensor>
        """
        if self.per_class:
            return self._calculate_recall(self.tp.sum(0), self.fn.sum(0))

        result = self._calculate_recall(self.tp.sum(1), self.fn.sum(1))

        return result.sum() / self.batch_size
