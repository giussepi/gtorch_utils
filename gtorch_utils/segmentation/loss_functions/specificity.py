# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/specificity """

import torch

from gtorch_utils.segmentation.metrics import Specificity


__all__ = ['SpecificityLoss']


class SpecificityLoss(torch.nn.Module):
    """
    Module based Specificity Loss with logits support

    Usage:
        Specificity()(predictions, ground_truth)
    """

    def __init__(self, *, with_logits=False, per_channel=False):
        """
        Initializes the object instance

        with_logits <bool>: set to True when working with logits to apply sigmoid
        per_channel <bool>: Set it to True to calculate specificity values per channel
        """
        super().__init__()
        assert isinstance(with_logits, bool), type(with_logits)
        assert isinstance(per_channel, bool), type(per_channel)

        self.with_logits = with_logits
        self.per_channel = per_channel
        self.specificity = Specificity(with_logits=self.with_logits, per_channel=self.per_channel)

    def forward(self, preds, targets):
        """
        Calculates and returns the specificity loss (true negative rate)

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

        return 1 - self.specificity(preds, targets)
