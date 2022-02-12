# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/specificity """

from typing import Callable

import torch

from gtorch_utils.segmentation.metrics import Specificity


__all__ = ['SpecificityLoss']


class SpecificityLoss(torch.nn.Module):
    """
    Module based Specificity Loss with logits support

    Usage:
        Specificity()(predictions, ground_truth)
    """

    def __init__(
            self, *, with_logits: bool = False, per_class: bool = False,
            logits_transform: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid
    ):
        """
        Initializes the object instance

        with_logits          <bool>: set to True when working with logits to apply sigmoid
        per_class            <bool>: Set it to True to calculate specificity values per class
        logits_transform <callable>: function to be applied to the logits in preds. Default torch.sigmoid
        """
        super().__init__()
        assert isinstance(with_logits, bool), type(with_logits)
        assert isinstance(per_class, bool), type(per_class)
        assert callable(logits_transform), f'{logits_transform} must be a callable'

        self.with_logits = with_logits
        self.per_class = per_class
        self.logits_transform = logits_transform
        self.specificity = Specificity(
            per_class=self.per_class, with_logits=self.with_logits, logits_transform=self.logits_transform)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the specificity loss (true negative rate)

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

        return 1 - self.specificity(preds, targets)
