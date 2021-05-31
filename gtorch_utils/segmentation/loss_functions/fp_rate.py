# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/fp_rate """

import torch

from gtorch_utils.segmentation.metrics.fp_rate import fpr


class FPR_Loss(torch.nn.Module):
    """
    Calculates and returns the False Positve Rate (FPR) loss

    Usage:
        loss = FPR_Loss()(predictions, ground_truth)
    """

    def __init__(self, per_channel=False):
        """
        initializes the object instance

        Args:
            per_channel <bool>: Whether or not return FPR per channel

        """
        super().__init__()
        assert isinstance(per_channel, bool), type(per_channel)

        self.per_channel = per_channel

    def forward(self, preds, targets):
        """
        Calculates and returns the FPR

        Args:
            preds  <torch.Tensor>: predicted masks [batch_size, channels, ...]
            target <torch.Tensor>: ground truth masks [batch_size, channels, ...]

        Returns:
            loss <torch.Tensor>
        """
        return 1 - fpr(preds, targets, self.per_channel)
