# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/balanced_accuracy """

from typing import Callable

import torch
from torchmetrics import Metric

from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


class BalancedAccuracy(Metric):
    r"""
    torchmetrics based balanced accuracy with logits support

    Calculates and returns the balanced accuracy of the provided masks or the
    balanced accuracy per class

    BALANCED_PIXEL_ACCURACY = \frac{RECALL + SPECIFICITY}{2} = \frac{TPR + TNR}{2}

    Usage:
        BalancedAccuracy()(preds, target)
    """

    def __init__(
            self, *, per_class: bool = False, with_logits: bool = False,
            logits_transform: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            **kwargs
    ):
        """
        Initializes the object instance

        Kwargs:
            per_class            <bool>: Whether or not return balanced accuracy values per class.
                                         Default False
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
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

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

        tp, fp, tn, fn = ConfusionMatrixMGR(preds, target)()

        if self.tp.shape != tp.shape:
            self.tp = torch.zeros_like(tp, device=preds.device)

        if self.fp.shape != fp.shape:
            self.fp = torch.zeros_like(fp, device=preds.device)

        if self.tn.shape != tn.shape:
            self.tn = torch.zeros_like(tn, device=preds.device)

        if self.fn.shape != fn.shape:
            self.fn = torch.zeros_like(fn, device=preds.device)

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    @staticmethod
    def _calculate_balanced_accuracy(
            tp: torch.Tensor, fp: torch.Tensor, tn: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the balanced accuracy

        Kwargs:
            tp <torch.Tensor>: tensor with true positive values
            fp <torch.Tensor>: tensor with false positive values
            tn <torch.Tensor>: tensor with true negative values
            fn <torch.Tensor>: tensor with false negative values

        returns:
            pixel_accuracy <torch.Tensor>
        """
        assert isinstance(tp, torch.Tensor), type(tp)
        assert isinstance(fp, torch.Tensor), type(fp)
        assert isinstance(tn, torch.Tensor), type(tn)
        assert isinstance(fn, torch.Tensor), type(fn)

        # avoiding circular references
        from gtorch_utils.segmentation.torchmetrics import Recall, Specificity

        return (Recall._calculate_recall(tp, fn) + Specificity._calculate_specificity(tn, fp)) / 2

    def compute(self) -> torch.Tensor:
        """
        Computes and returns the balanced accuracy

        Returns:
            balanced_accuracy <torch.Tensor>
        """
        if self.per_class:
            return self._calculate_balanced_accuracy(self.tp.sum(0), self.fp.sum(0), self.tn.sum(0), self.fn.sum(0))

        return self._calculate_balanced_accuracy(self.tp.sum(), self.fp.sum(), self.tn.sum(), self.fn.sum())
