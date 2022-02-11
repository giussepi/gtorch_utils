# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/dice """

from typing import Callable

import torch
from torchmetrics import Metric

from gtorch_utils.constants import EPSILON


__all__ = ['DiceCoefficient',  'DiceCoefficientPerImage']


class DiceMixin:
    """
    Holds basics methods related to computing the Dice coefficient

    Usage:
        class MyDiceMetric(DiceMixin):
            <some code goes here>
    """

    @staticmethod
    def _calculate_dice(intersection: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the Dice coefficient

        Kwargs:
            intersection <torch.Tensor>: tensor with intersection values
            union        <torch.Tensor>: tensor with union values

        returns:
            recall <torch.Tensor>
        """
        assert isinstance(intersection, torch.Tensor), type(intersection)
        assert isinstance(union, torch.Tensor), type(union)

        return (2 * intersection.float() + EPSILON) / (union.float() + EPSILON)


class DiceCoefficient(Metric, DiceMixin):
    r"""
    TorchMetric based Dice coefficient with logits support

    DICECOEFFICIENT = \frac{ 2 \left| X \cap Y \right| }{ \left| X \right| + \left| Y \right| }

    DICECOEFFICIENT = \frac{ 2TP }{ 2TP + FP + FN}

    Usage:
        DiceCoefficient()(preds, target)
    """

    def __init__(
            self, *, per_class: bool = False,  with_logits: bool = False,
            logits_transform: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            **kwargs
    ):
        """
        Initializes the object instance

        per_class <bool>: Set it to True to calculate specificity values per class
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
        self.add_state('intersection', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('union', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates the state given the inputs

        Kwargs:
            preds  <torch.Tensor>: predicted masks [batch_size, classes, ...]
            target <torch.Tensor>: ground truth masks [batch_size, classes, ...]
        """
        assert preds.shape == target.shape, f'{preds.shape} != {target.shape}'
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(target, torch.Tensor), type(target)

        if self.with_logits:
            preds = self.logits_transform(preds)

        if self.intersection.shape != preds.shape[:2]:
            self.intersection = torch.zeros(preds.shape[:2], device=preds.device)

        if self.union.shape != preds.shape[:2]:
            self.union = torch.zeros(preds.shape[:2], device=preds.device)

        preds_ = preds.view(*preds.shape[:2], -1)
        target_ = target.view(*target.shape[:2], -1)
        self.intersection += (preds_ * target_).sum(2)
        self.union += (preds_ + target_).sum(2)

    def compute(self) -> torch.Tensor:
        """
        Computes and returns Dice coefficient

        Returns:
            recall <torch.Tensor>
        """
        if self.per_class:
            return self._calculate_dice(self.intersection.sum(0), self.union.sum(0))

        return self._calculate_dice(self.intersection.sum(), self.union.sum())


class DiceCoefficientPerImage(Metric, DiceMixin):
    r"""
    TorchMetric based Dice coefficient with logits support

    DICECOEFFICIENT = \frac{ 2 \left| X \cap Y \right| }{ \left| X \right| + \left| Y \right| }

    DICECOEFFICIENT = \frac{ 2TP }{ 2TP + FP + FN}

    Usage:
        DiceCoefficient()(preds, target)
    """

    def __init__(
            self, *, per_class: bool = False, with_logits: bool = False,
            logits_transform: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            **kwargs
    ):
        """
        Initializes the object instance

        per_class <bool>: Set it to True to calculate specificity values per class
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
        self.add_state('score', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('batch_size', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates the state given the inputs

        Kwargs:
            preds  <torch.Tensor>: predicted masks [batch_size, classes, ...]
            target <torch.Tensor>: ground truth masks [batch_size, classes, ...]
        """
        assert preds.shape == target.shape, f'{preds.shape} != {target.shape}'
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(target, torch.Tensor), type(target)

        if self.with_logits:
            preds = self.logits_transform(preds)

        if self.per_class and self.score.shape != preds.size(1):
            self.score = torch.zeros(preds.size(1), device=preds.device, dtype=preds.dtype)

        preds_ = preds.view(*preds.shape[:2], -1)
        target_ = target.view(*target.shape[:2], -1)
        intersection = (preds_ * target_).sum(2)
        union = (preds_ + target_).sum(2)

        if self.per_class:
            self.score += self._calculate_dice(intersection, union).sum(0)
        else:
            self.score += self._calculate_dice(intersection.sum(1), union.sum(1)).sum()

        self.batch_size += torch.tensor(preds.size(0))

    def compute(self) -> torch.Tensor:
        """
        Computes and returns the dice coefficient

        Returns:
            recall <torch.Tensor>
        """
        return self.score / self.batch_size
