# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/recall """

import torch

from gtorch_utils.segmentation.metrics.recall import recall


class Recall_Loss(torch.nn.Module):
    """
    Calculates and returns the recall loss

    Usage:
        loss = Recall_Loss()(predictions, ground_truth)
    """

    def __init__(self, per_channel=False):
        """
        initializes the object instance

        Args:
            per_channel <bool>: Whether or not return recall per channel

        """
        super().__init__()
        assert isinstance(per_channel, bool), type(per_channel)

        self.per_channel = per_channel

    def forward(self, preds, targets):
        """
        Calculates and returns the recall_loss

        Args:
            preds  <torch.Tensor>: predicted masks [batch_size, channels, ...]
            target <torch.Tensor>: ground truth masks [batch_size, channels, ...]

        Returns:
            loss <torch.Tensor>
        """
        return 1 - recall(preds, targets, self.per_channel)
